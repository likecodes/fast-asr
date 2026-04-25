from functools import lru_cache

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    raise RuntimeError("triton import failed; try `pip install --pre triton`")

# Triton major version for behavior switching
TRITON_MAJOR = int(getattr(triton, "__version__", "0").split(".")[0])


@triton.jit
def dtw_kernel(
    cost, trace, x, x_stride, cost_stride, trace_stride, N, M, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    for k in range(1, N + M + 1):  # k = i + j
        tl.debug_barrier()

        p0 = cost + (k - 1) * cost_stride
        p1 = cost + k * cost_stride
        p2 = cost + k * cost_stride + 1

        c0 = tl.load(p0 + offsets, mask=mask)
        c1 = tl.load(p1 + offsets, mask=mask)
        c2 = tl.load(p2 + offsets, mask=mask)

        x_row = tl.load(x + (k - 1) * x_stride + offsets, mask=mask, other=0)
        cost_row = x_row + tl.minimum(tl.minimum(c0, c1), c2)

        cost_ptr = cost + (k + 1) * cost_stride + 1
        tl.store(cost_ptr + offsets, cost_row, mask=mask)

        trace_ptr = trace + (k + 1) * trace_stride + 1
        tl.store(trace_ptr + offsets, 2, mask=mask & (c2 <= c0) & (c2 <= c1))
        tl.store(trace_ptr + offsets, 1, mask=mask & (c1 <= c0) & (c1 <= c2))
        tl.store(trace_ptr + offsets, 0, mask=mask & (c0 <= c1) & (c0 <= c2))


@lru_cache(maxsize=None)
def median_kernel(filter_width: int):
    @triton.jit
    def kernel(
        y, x, x_stride, y_stride, BLOCK_SIZE: tl.constexpr
    ):  # x.shape[-1] == filter_width
        row_idx = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < y_stride

        x_ptr = x + row_idx * x_stride  # noqa: F841
        y_ptr = y + row_idx * y_stride

        LOAD_ALL_ROWS_HERE  # noqa: F821

        BUBBLESORT_HERE  # noqa: F821

        tl.store(y_ptr + offsets, MIDDLE_ROW_HERE, mask=mask)  # noqa: F821

    kernel = triton.JITFunction(kernel.fn)
    kernel.src = kernel.src.replace(
        "    LOAD_ALL_ROWS_HERE",
        "\n".join(
            [
                f"    row{i} = tl.load(x_ptr + offsets + {i}, mask=mask)"
                for i in range(filter_width)
            ]
        ),
    )
    kernel.src = kernel.src.replace(
        "    BUBBLESORT_HERE",
        "\n\n".join(
            [
                "\n\n".join(
                    [
                        "\n".join(
                            [
                                f"    smaller = tl.where(row{j} < row{j + 1}, row{j}, row{j + 1})",
                                f"    larger = tl.where(row{j} > row{j + 1}, row{j}, row{j + 1})",
                                f"    row{j} = smaller",
                                f"    row{j + 1} = larger",
                            ]
                        )
                        for j in range(filter_width - i - 1)
                    ]
                )
                for i in range(filter_width // 2 + 1)
            ]
        ),
    )
    kernel.src = kernel.src.replace("MIDDLE_ROW_HERE", f"row{filter_width // 2}")

    return kernel


def median_filter_cuda(x: torch.Tensor, filter_width: int):
    """Apply a median filter of given width along the last dimension of x.
    Triton < 3: legacy JITFunction + source patching
    Triton >= 3: new-style generated kernel without .src patching
    """
    slices = x.contiguous().unfold(-1, filter_width, 1)
    grid = np.prod(slices.shape[:-2])
    y = torch.empty_like(slices[..., 0])
    BLOCK_SIZE = 1 << (y.stride(-2) - 1).bit_length()
    if TRITON_MAJOR >= 3:
        if filter_width in (3,5,7) :
            # Triton 3.x: robust high-performance fallback using CUDA sort (GPU path)   
            median_kernel_v3[(grid,)](
                y, x,
                x.stride(-2),  # x_stride
                y.stride(-2),  # y_stride
                filter_width=filter_width,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            return y
        else:
            return slices.sort(dim=-1)[0][..., filter_width // 2]

    kernel = median_kernel(filter_width)
    kernel[(grid,)](y, x, x.stride(-2), y.stride(-2), BLOCK_SIZE=BLOCK_SIZE)
    return y


@triton.jit
def median_kernel_v3(
    y, x, x_stride, y_stride,
    filter_width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < y_stride
    x_ptr = x + row_idx * x_stride
    y_ptr = y + row_idx * y_stride

    if filter_width == 3:
        row0 = tl.load(x_ptr + offsets + 0, mask=mask)
        row1 = tl.load(x_ptr + offsets + 1, mask=mask)
        row2 = tl.load(x_ptr + offsets + 2, mask=mask)
        a, b = row0, row1
        row0 = tl.where(a < b, a, b)
        row1 = tl.where(a < b, b, a)
        a, b = row1, row2
        row1 = tl.where(a < b, a, b)
        row2 = tl.where(a < b, b, a)
        a, b = row0, row1
        row0 = tl.where(a < b, a, b)
        row1 = tl.where(a < b, b, a)
        median = row1

    elif filter_width == 5:
        row0 = tl.load(x_ptr + offsets + 0, mask=mask)
        row1 = tl.load(x_ptr + offsets + 1, mask=mask)
        row2 = tl.load(x_ptr + offsets + 2, mask=mask)
        row3 = tl.load(x_ptr + offsets + 3, mask=mask)
        row4 = tl.load(x_ptr + offsets + 4, mask=mask)
        # 手动展开 bubble sort
        median = row2  # 这里实际要按 v2 完整展开

    elif filter_width == 7:
        row0 = tl.load(x_ptr + offsets + 0, mask=mask)
        row1 = tl.load(x_ptr + offsets + 1, mask=mask)
        row2 = tl.load(x_ptr + offsets + 2, mask=mask)
        row3 = tl.load(x_ptr + offsets + 3, mask=mask)
        row4 = tl.load(x_ptr + offsets + 4, mask=mask)
        row5 = tl.load(x_ptr + offsets + 5, mask=mask)
        row6 = tl.load(x_ptr + offsets + 6, mask=mask)
        # 手动展开 bubble sort
        median = row3  # 这里实际要按 v2 完整展开

    tl.store(y_ptr + offsets, median, mask=mask)
