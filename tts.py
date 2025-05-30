import asyncio
import pygame
import os
import tempfile
from datetime import datetime
from pydub import AudioSegment
from edge_tts import Communicate

OUTPUT_DIR = "audio_output"
VOICE = "zh-CN-YunyangNeural"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

async def generate_speech(text: str) -> str:
    if not text.strip():
        raise ValueError("输入文本不能为空")
    timestamp = get_timestamp()
    output_file = os.path.join(OUTPUT_DIR, f"{timestamp}.wav")
    temp_mp3_path = None
    try:
        # 生成MP3临时文件
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_mp3_path = temp_file.name
        communicate = Communicate(text, VOICE)
        await communicate.save(temp_mp3_path)
        # 转为16kHz单声道WAV
        audio = AudioSegment.from_file(temp_mp3_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_file, format="wav")
        # 确保文件完全写入磁盘
        with open(output_file, "ab") as f:
            f.flush()
            os.fsync(f.fileno())
        os.remove(temp_mp3_path)
        # 再次确保文件完整
        if not os.path.exists(output_file) or os.path.getsize(output_file) < 800:
            raise RuntimeError("生成的WAV文件无效或过小")
        return output_file
    except Exception as e:
        print(f"❌ 语音生成失败: {str(e)}")
        if temp_mp3_path and os.path.exists(temp_mp3_path):
            os.remove(temp_mp3_path)
        raise

def play_audio(file_path: str) -> None:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"音频文件不存在: {file_path}")
    try:
        pygame.mixer.init(frequency=16000)
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
    except Exception as e:
        print(f"❌ 播放失败: {str(e)}")
        raise
    finally:
        pygame.mixer.quit()

if __name__ == "__main__":
    test_text = "您好，这是一段微软EdgeTTS老年男声的演示。"
    try:
        wav_path = asyncio.run(generate_speech(test_text))
        print(f"语音已生成: {wav_path}")
        play_audio(wav_path)
    except Exception as e:
        print(f"调试失败: {e}")