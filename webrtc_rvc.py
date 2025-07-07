import asyncio
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat
import librosa
from pulsectl_asyncio import PulseAsync
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack
from tools.torchgate import TorchGate
from infer.lib import rvc_for_realtime
from configs.config import Config
import multiprocessing
from multiprocessing import Queue
import pyworld
import time
import json
import websockets

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 设置环境变量
os.environ["OMP_NUM_THREADS"] = "4"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 工作目录
now_dir = os.getcwd()
sys.path.append(now_dir)

# 自定义打印函数
def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)

# 相位声码器
def phase_vocoder(a, b, fade_out, fade_in):
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    result = (
        a * (fade_out**2)
        + b * (fade_in**2)
        + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
    )
    return result

# Harvest 进程用于音高提取
class Harvest(multiprocessing.Process):
    def __init__(self, inp_q, opt_q):
        multiprocessing.Process.__init__(self)
        self.inp_q = inp_q
        self.opt_q = opt_q

    def run(self):
        import numpy as np
        import pyworld

        while True:
            idx, x, res_f0, n_cpu, ts = self.inp_q.get()
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=16000,
                f0_ceil=1100,
                f0_floor=50,
                frame_period=10,
            )
            res_f0[idx] = f0
            if len(res_f0.keys()) >= n_cpu:
                self.opt_q.put(ts)

# 配置类
class AudioConfig:
    def __init__(self):
        self.pth_path = "path/to/your/model.pth"  # 替换为实际模型路径
        self.index_path = "path/to/your/index.index"  # 替换为实际索引路径
        self.pitch = 0
        self.formant = 0.0
        self.sr_type = "sr_model"
        self.block_time = 0.25  # 块时间（秒）
        self.threhold = -60  # 响应阈值（dB）
        self.crossfade_time = 0.05  # 淡入淡出时间（秒）
        self.extra_time = 2.5  # 额外推理时间（秒）
        self.I_noise_reduce = False  # 输入降噪
        self.O_noise_reduce = False  # 输出降噪
        self.use_pv = False  # 是否启用相位声码器
        self.rms_mix_rate = 0.0  # 响度混合比率
        self.index_rate = 0.0  # 索引比率
        self.n_cpu = min(multiprocessing.cpu_count(), 4)  # CPU 核心数
        self.f0method = "fcpe"  # 音高检测方法
        self.samplerate = 44100  # 默认采样率，可根据服务端调整
        self.channels = 2  # 通道数（立体声）

# 自定义音频轨道
class AudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.audio_buffer = asyncio.Queue()

    async def recv(self):
        """模拟接收音频数据"""
        data = await self.audio_buffer.get()
        return data

    def add_audio(self, data):
        """将接收到的音频数据添加到缓冲区"""
        self.audio_buffer.put_nowait(data)

# 主音频处理类
class WebRTCRVC:
    def __init__(self, signaling_server="ws://localhost:8765"):
        self.config = Config()
        self.audio_config = AudioConfig()
        self.inp_q = Queue()
        self.opt_q = Queue()
        self.n_cpu = self.audio_config.n_cpu
        for _ in range(self.n_cpu):
            p = Harvest(self.inp_q, self.opt_q)
            p.daemon = True
            p.start()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.rvc = None
        self.stream_active = False
        self.pc = None
        self.audio_track = AudioStreamTrack()
        self.signaling_server = signaling_server

    async def initialize(self):
        # 初始化 RVC 模型
        self.rvc = rvc_for_realtime.RVC(
            self.audio_config.pitch,
            self.audio_config.formant,
            self.audio_config.pth_path,
            self.audio_config.index_path,
            self.audio_config.index_rate,
            self.audio_config.n_cpu,
            self.inp_q,
            self.opt_q,
            self.config,
            None
        )
        self.audio_config.samplerate = (
            self.rvc.tgt_sr if self.audio_config.sr_type == "sr_model" else self.audio_config.samplerate
        )
        self.zc = self.audio_config.samplerate // 100
        self.block_frame = int(np.round(self.audio_config.block_time * self.audio_config.samplerate / self.zc)) * self.zc
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = int(np.round(self.audio_config.crossfade_time * self.audio_config.samplerate / self.zc)) * self.zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = int(np.round(self.audio_config.extra_time * self.audio_config.samplerate / self.zc)) * self.zc
        self.input_wav = torch.zeros(
            self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame,
            device=self.device,
            dtype=torch.float32,
        )
        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.device,
            dtype=torch.float32,
        )
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=self.device, dtype=torch.float32)
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc
        self.fade_in_window = torch.sin(
            0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.sola_buffer_frame, device=self.device, dtype=torch.float32)
        ) ** 2
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(orig_freq=self.audio_config.samplerate, new_freq=16000, dtype=torch.float32).to(self.device)
        self.resampler2 = tat.Resample(orig_freq=self.rvc.tgt_sr, new_freq=self.audio_config.samplerate, dtype=torch.float32).to(self.device) if self.rvc.tgt_sr != self.audio_config.samplerate else None
        self.tg = TorchGate(sr=self.audio_config.samplerate, n_fft=4 * self.zc, prop_decrease=0.9).to(self.device)

    async def audio_callback(self, pulse, sink):
        """处理音频流"""
        while self.stream_active:
            # 从 WebRTC 接收音频
            data = await self.audio_track.recv()
            indata = np.frombuffer(data, dtype=np.float32).reshape(-1, self.audio_config.channels)
            indata = librosa.to_mono(indata.T)  # 转换为单声道

            # 阈值处理
            if self.audio_config.threhold > -60:
                indata = np.append(self.rms_buffer, indata)
                rms = librosa.feature.rms(y=indata, frame_length=4 * self.zc, hop_length=self.zc)[:, 2:]
                self.rms_buffer[:] = indata[-4 * self.zc:]
                indata = indata[2 * self.zc - self.zc // 2:]
                db_threhold = librosa.amplitude_to_db(rms, ref=1.0)[0] < self.audio_config.threhold
                for i in range(db_threhold.shape[0]):
                    if db_threhold[i]:
                        indata[i * self.zc:(i + 1) * self.zc] = 0
                indata = indata[self.zc // 2:]

            # 更新输入缓冲区
            self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame:].clone()
            self.input_wav[-indata.shape[0]:] = torch.from_numpy(indata).to(self.device)
            self.input_wav_res[:-self.block_frame_16k] = self.input_wav_res[self.block_frame_16k:].clone()

            # 输入降噪和重采样
            if self.audio_config.I_noise_reduce:
                self.input_wav_denoise[:-self.block_frame] = self.input_wav_denoise[self.block_frame:].clone()
                input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame:]
                input_wav = self.tg(input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)).squeeze(0)
                input_wav[:self.sola_buffer_frame] *= self.fade_in_window
                input_wav[:self.sola_buffer_frame] += self.nr_buffer * self.fade_out_window
                self.input_wav_denoise[-self.block_frame:] = input_wav[:self.block_frame]
                self.nr_buffer[:] = input_wav[self.block_frame:]
                self.input_wav_res[-self.block_frame_16k - 160:] = self.resampler(
                    self.input_wav_denoise[-self.block_frame - 2 * self.zc:]
                )[160:]
            else:
                self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1):] = self.resampler(
                    self.input_wav[-indata.shape[0] - 2 * self.zc:]
                )[160:]

            # 变声处理
            start_time = time.perf_counter()
            infer_wav = self.rvc.infer(
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.return_length,
                self.audio_config.f0method,
            )
            if self.resampler2 is not None:
                infer_wav = self.resampler2(infer_wav)

            # 输出降噪
            if self.audio_config.O_noise_reduce:
                self.output_buffer[:-self.block_frame] = self.output_buffer[self.block_frame:].clone()
                self.output_buffer[-self.block_frame:] = infer_wav[-self.block_frame:]
                infer_wav = self.tg(infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)).squeeze(0)

            # 响度混合
            if self.audio_config.rms_mix_rate < 1:
                input_wav = self.input_wav_denoise[self.extra_frame:] if self.audio_config.I_noise_reduce else self.input_wav[self.extra_frame:]
                rms1 = librosa.feature.rms(y=input_wav[:infer_wav.shape[0]].cpu().numpy(), frame_length=4 * self.zc, hop_length=self.zc)
                rms1 = torch.from_numpy(rms1).to(self.device)
                rms1 = F.interpolate(rms1.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
                rms2 = librosa.feature.rms(y=infer_wav.cpu().numpy(), frame_length=4 * self.zc, hop_length=self.zc)
                rms2 = torch.from_numpy(rms2).to(self.device)
                rms2 = F.interpolate(rms2.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
                rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
                infer_wav *= torch.pow(rms1 / rms2, torch.tensor(1 - self.audio_config.rms_mix_rate))

            # SOLA 算法
            conv_input = infer_wav[None, None, :self.sola_buffer_frame + self.sola_search_frame]
            cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
            cor_den = torch.sqrt(F.conv1d(conv_input**2, torch.ones(1, 1, self.sola_buffer_frame, device=self.device)) + 1e-8)
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
            infer_wav = infer_wav[sola_offset:]
            if not self.audio_config.use_pv:
                infer_wav[:self.sola_buffer_frame] *= self.fade_in_window
                infer_wav[:self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
            else:
                infer_wav[:self.sola_buffer_frame] = phase_vocoder(
                    self.sola_buffer,
                    infer_wav[:self.sola_buffer_frame],
                    self.fade_out_window,
                    self.fade_in_window,
                )
            self.sola_buffer[:] = infer_wav[self.block_frame:self.block_frame + self.sola_buffer_frame]

            # 输出音频到 PulseAudio
            outdata = infer_wav[:self.block_frame].repeat(self.audio_config.channels, 1).t().cpu().numpy()
            await sink.write_async(outdata.tobytes())

            # 记录推理时间
            total_time = time.perf_counter() - start_time
            printt("Infer time: %.2f ms", total_time * 1000)

    async def start_stream(self):
        """启动 WebRTC 和 PulseAudio 流"""
        self.stream_active = True
        self.pc = RTCPeerConnection()
        self.pc.addTrack(self.audio_track)

        async with PulseAsync('rvc-client') as pulse:
            # 创建 PulseAudio 播放流
            sink = await pulse.sink_input_new(
                sink_name=None,  # 使用默认接收器
                stream_name='rvc-output',
                rate=self.audio_config.samplerate,
                channels=self.audio_config.channels,
                format='float32le'
            )

            # WebRTC 信令
            async with websockets.connect(self.signaling_server) as websocket:
                # 创建并发送 offer
                await self.pc.setLocalDescription(await self.pc.createOffer())
                await websocket.send(json.dumps({"sdp": self.pc.localDescription.sdp, "type": self.pc.localDescription.type}))

                # 接收 answer
                response = json.loads(await websocket.recv())
                await self.pc.setRemoteDescription(RTCSessionDescription(sdp=response["sdp"], type=response["type"]))

                # 处理 ICE 候选者
                async def handle_ice():
                    while True:
                        candidate = await websocket.recv()
                        candidate = json.loads(candidate)
                        if candidate:
                            await self.pc.addIceCandidate(candidate)

                # 模拟接收音频数据（实际由服务端发送）
                async def simulate_audio():
                    # 假设服务端发送 PCM 数据
                    while self.stream_active:
                        # 示例：模拟从服务端接收的音频数据
                        # 替换为实际从 WebRTC 数据通道或媒体流接收的逻辑
                        data = np.random.randn(self.block_frame * self.audio_config.channels).astype(np.float32).tobytes()
                        self.audio_track.add_audio(data)
                        await asyncio.sleep(self.audio_config.block_time)

                # 启动音频处理和 ICE 候选者处理
                await asyncio.gather(
                    self.audio_callback(pulse, sink),
                    handle_ice(),
                    simulate_audio()
                )

    async def stop_stream(self):
        """停止音频流"""
        self.stream_active = False
        if self.pc:
            await self.pc.close()

    def run(self):
        """运行主程序"""
        asyncio.run(self.initialize())
        asyncio.run(self.start_stream())

if __name__ == "__main__":
    webrtc_rvc = WebRTCRVC(signaling_server="ws://localhost:8765")  # 替换为实际信令服务器地址
    try:
        webrtc_rvc.run()
    except KeyboardInterrupt:
        asyncio.run(webrtc_rvc.stop_stream())