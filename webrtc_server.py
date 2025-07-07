import asyncio
import json
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription

async def webrtc_server(websocket, path):
    pc = RTCPeerConnection()
    # 这里添加音频轨道（示例中省略，实际需添加音频源）
    # 例如：pc.addTrack(AudioStreamTrack())
    
    @pc.on("icecandidate")
    async def on_icecandidate(candidate):
        if candidate:
            await websocket.send(json.dumps(candidate))

    # 接收客户端的 offer
    offer = json.loads(await websocket.recv())
    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"], type=offer["type"]))
    
    # 创建并发送 answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await websocket.send(json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}))

    # 保持连接
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    start_server = websockets.serve(webrtc_server, "localhost", 8765)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


# 在 webrtc_rvc.py 的 AudioStreamTrack 类中添加 ontrack 处理
class AudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.audio_buffer = asyncio.Queue()

    async def recv(self):
        return await self.audio_buffer.get()

    def add_audio(self, data):
        self.audio_buffer.put_nowait(data)
