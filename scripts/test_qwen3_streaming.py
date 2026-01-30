#!/usr/bin/env python3
"""
Qwen3-ASR WebSocket 流式识别测试脚本 (POC)

使用方法:
1. 启动服务: python -m app.main
2. 运行测试: python scripts/test_qwen3_streaming.py --audio /path/to/audio.wav

注意：需要使用 Qwen3-ASR 模型才能测试此端点
"""

import argparse
import asyncio
import json
import sys
import wave
import numpy as np

import websockets


async def test_qwen3_streaming(audio_path: str, server_url: str = "ws://localhost:8000/ws/v1/qwen3/asr"):
    """测试 Qwen3-ASR 流式识别"""

    print(f"连接到 {server_url}...")

    async with websockets.connect(server_url) as websocket:
        print("连接成功!")

        # 1. 发送开始识别消息
        start_msg = {
            "type": "start",
            "payload": {
                "format": "pcm",
                "sample_rate": 16000,
                "language": None,  # 自动检测
                "context": "",
                "chunk_size_sec": 2.0,
            }
        }
        await websocket.send(json.dumps(start_msg))
        print(f"发送: {start_msg}")

        # 等待 started 响应
        response = await websocket.recv()
        data = json.loads(response)
        print(f"收到: {data}")

        if data.get("type") != "started":
            print("启动识别失败!")
            return

        # 2. 读取并发送音频数据
        print(f"\n读取音频文件: {audio_path}")

        # 尝试读取 WAV 文件
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                num_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                num_frames = wav_file.getnframes()

                print(f"  采样率: {sample_rate} Hz")
                print(f"  声道数: {num_channels}")
                print(f"  采样宽度: {sample_width} bytes")
                print(f"  总帧数: {num_frames}")

                # 读取音频数据
                audio_data = wav_file.readframes(num_frames)

                # 如果是立体声，转换为单声道
                if num_channels == 2:
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
                    audio_data = audio_array.tobytes()

        except Exception as e:
            print(f"读取音频文件失败: {e}")
            return

        # 分块发送音频（每块 0.5 秒）
        chunk_size = int(16000 * 2 * 0.5)  # 0.5秒，16bit = 2 bytes/sample
        total_chunks = 0

        print(f"\n开始发送音频数据（每块 {chunk_size} bytes = 0.5秒）...")

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            await websocket.send(chunk)
            total_chunks += 1

            # 尝试接收中间结果
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                data = json.loads(response)
                if data.get("type") == "result":
                    for result in data.get("results", []):
                        print(f"  [Chunk {result.get('chunk_id')}] "
                              f"Language: {result.get('language')} | "
                              f"Text: {result.get('text')}")
            except asyncio.TimeoutError:
                pass

            # 模拟实时音频流
            await asyncio.sleep(0.5)

        print(f"\n音频发送完成，共 {total_chunks} 块")

        # 3. 发送停止识别消息
        stop_msg = {"type": "stop"}
        await websocket.send(json.dumps(stop_msg))
        print(f"发送: {stop_msg}")

        # 等待最终结果
        response = await websocket.recv()
        data = json.loads(response)
        print(f"\n收到最终结果: {json.dumps(data, indent=2, ensure_ascii=False)}")

        if data.get("type") == "final":
            result = data.get("result", {})
            print(f"\n识别完成!")
            print(f"  语言: {result.get('language')}")
            print(f"  文本: {result.get('text')}")
            print(f"  总块数: {result.get('total_chunks')}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR WebSocket 流式识别测试")
    parser.add_argument(
        "--audio", "-a",
        required=True,
        help="测试音频文件路径 (WAV格式, 16kHz或自动重采样)"
    )
    parser.add_argument(
        "--server", "-s",
        default="ws://localhost:8000/ws/v1/qwen3/asr",
        help="WebSocket 服务器地址"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3-ASR WebSocket 流式识别测试 (POC)")
    print("=" * 60)

    try:
        asyncio.run(test_qwen3_streaming(args.audio, args.server))
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
