import json
from typing import List
import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["WebSocket"])

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("ws_client_connected", total_clients=len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("ws_client_disconnected", total_clients=len(self.active_connections))

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        payload = json.dumps(message)
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception:
                dead_connections.append(connection)
        for dead in dead_connections:
            self.disconnect(dead)

manager = ConnectionManager()
_main_loop = None

def set_main_loop(loop):
    global _main_loop
    _main_loop = loop
    logger.info("ws_main_loop_registered")

def broadcast_sync(message: dict):
    global _main_loop
    try:
        import asyncio
        if _main_loop is not None and _main_loop.is_running():
            asyncio.run_coroutine_threadsafe(manager.broadcast(message), _main_loop)
            return

        # Fallback to current thread loop
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                asyncio.create_task(manager.broadcast(message))
                return
        except RuntimeError:
            pass

        loop = asyncio.new_event_loop()
        loop.run_until_complete(manager.broadcast(message))
        loop.close()
    except Exception as e:
        logger.warning("broadcast_sync_error", error=str(e))

@router.websocket("/ws/scrutiny")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"event": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.debug("ws_client_closed", error=str(e))
        manager.disconnect(websocket)
