import asyncio
import json
import numpy as np
import sounddevice as sd
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer
from aiortc.contrib.media import MediaStreamTrack

# 音频配置
SAMPLE_RATE = 44100
CHANNELS = 2
BLOCK_SIZE = int(SAMPLE_RATE * 0.25)

# 自定义音频轨道
class MicrophoneAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.audio_buffer = asyncio.Queue()
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=self.audio_callback
        )

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")
        self.audio_buffer.put_nowait(indata.copy().tobytes())

    async def recv(self):
        return await self.audio_buffer.get()

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()

async def webrtc_server(websocket, path):
    # 配置 ICE 服务器
    ice_servers = [
        RTCIceServer(urls="stun:stun.l.google.com:19302"),  # 公共 STUN 服务器
        # 可添加 TURN 服务器，例如：
        # RTCIceServer(urls="turn:your.turn.server", username="username", credential="password")
    ]
    pc = RTCPeerConnection(iceServers=ice_servers)
    audio_track = MicrophoneAudioTrack()

    try:
        pc.addTrack(audio_track)

        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate:
                await websocket.send(json.dumps({
                    "type": "candidate",
                    "candidate": candidate.sdp,
                    "sdpMid": candidate.sdpMid,
                    "sdpMLineIndex": candidate.sdpMLineIndex
                }))

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Connection state: {pc.connectionState}")
            if pc.connectionState == "failed":
                audio_track.stop()
                await pc.close()

        # 启动麦克风音频捕获
        audio_track.start()

        # 接收客户端的 offer
        try:
            message = json.loads(await asyncio.wait_for(websocket.recv(), timeout=10))
            if message.get("type") != "offer":
                raise ValueError("Expected SDP offer")
            await pc.setRemoteDescription(RTCSessionDescription(sdp=message["sdp"], type=message["type"]))
        except Exception as e:
            print(f"Error receiving offer: {e}")
            return

        # 创建并发送 answer
        try:
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await websocket.send(json.dumps({
                "type": "answer",
                "sdp": pc.localDescription.sdp
            }))
        except Exception as e:
            print(f"Error creating/sending answer: {e}")
            return

        # 处理 ICE 候选者
        while True:
            try:
                message = json.loads(await asyncio.wait_for(websocket.recv(), timeout=30))
                if message.get("type") == "candidate" and message.get("candidate"):
                    await pc.addIceCandidate(RTCSessionDescription(
                        sdp=message["candidate"],
                        type="candidate",
                        sdpMid=message["sdpMid"],
                        sdpMLineIndex=message["sdpMLineIndex"]
                    ))
            except asyncio.TimeoutError:
                print("No ICE candidates received, closing connection")
                break
            except Exception as e:
                print(f"Error processing ICE candidate: {e}")
                break

    except Exception as e:
        print(f"WebRTC server error: {e}")
    finally:
        audio_track.stop()
        await pc.close()

if __name__ == "__main__":
    start_server = websockets.serve(webrtc_server, "localhost", 8765)
    asyncio.get_event_loop().run_until_complete(start_server)
    print("WebRTC server running on ws://localhost:8765")
    asyncio.get_event_loop().run_forever()