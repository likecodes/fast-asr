"""
启动服务器脚本
"""

import asyncio
import logging
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(__file__))

from server.websocket_server import WebSocketTranscriptionServer
from config.config import config_manager
from utils import log_system_info

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():

    
    """主函数"""
    try:
        # 记录系统信息
        log_system_info()
        
        # 获取配置
        config = config_manager.get_config()
        server_config = config['server']
        
        # 创建服务器
        server = WebSocketTranscriptionServer(
            host=server_config.host,
            port=server_config.port
        )
        
        # 启动服务器并等待
        await server.start_server()
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止服务器...")
        try:
            await server.stop()
        except Exception:
            pass
    except Exception as e:
        logger.error(f"服务器启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())