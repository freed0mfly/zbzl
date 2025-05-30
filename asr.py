import time
import sounddevice as sd
import numpy as np
from funasr import AutoModel
import torch

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.3
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
FORMAT = "int16"
BUFFER_DURATION = 2.0
wake_word = "开"
exit_word = "关机"
exit_chunks =200

vad = AutoModel(
    model="fsmn-vad",
    model_revision="v2.0.4",
    device="cuda" if torch.cuda.is_available() else "cpu",
    quantize=True,
    disable_update=True
)
asr = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_itn=True,
    disable_update=True
)

class VadAsrProcessor:
    def __init__(self, on_text=None):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.sample_points = int(BUFFER_DURATION * SAMPLE_RATE)
        self.silence_count = 0
        self.exit_count = 0
        self.is_wake = False
        self.is_exit = True
        self.txt = ""
        self.on_text = on_text  # 回调

    def process_chunk(self, indata):
        chunk = indata[:, 0].astype(np.float32) / 32768.0
        try:
            result = vad.generate(
                input=chunk,
                batch_size_s=50,
                max_single_segment_length=3000,
            )
            if result and result[0] and 'value' in result[0]:
                vad_segments = result[0]['value']
                if vad_segments:
                    total_duration = sum((end - start) / 1000 for start, end in vad_segments)
                    activation_ratio = total_duration / (len(chunk) / SAMPLE_RATE)
                    return activation_ratio > 0.6, chunk
            return False, chunk
        except Exception as e:
            print(f"VAD处理异常: {str(e)}")
            return False, chunk

    def parse_result(self, indata):
        try:
            is_voice, chunk = self.process_chunk(indata)
        except TypeError:
            return

        if is_voice:
            self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
            self.silence_count = 0
            self.exit_count = 0
        else:
            self.silence_count += 1
            self.exit_count += 1
            if self.silence_count == 2 and len(self.audio_buffer) > 0:
                try:
                    self.txt = asr.generate(input=self.audio_buffer)[0]['text']
                    print(f"识别结果: {self.txt}")

                    # 唤醒逻辑
                    if wake_word in self.txt and not self.is_wake:
                        self.is_wake = True
                        self.is_exit = False
                        print("--------------------------------\n系统已唤醒\n")
                        if self.on_text:
                            self.on_text("", True)  # 仅唤醒状态改变
                        return
                    # 退出逻辑
                    if exit_word in self.txt:
                        self.is_exit = True
                        print("--------------------------------\n检测到退出词，系统退出唤醒\n")
                        if self.on_text:
                            self.on_text("", False)
                        return
                    # 唤醒状态下处理内容
                    if self.is_wake and not self.is_exit:
                        print(f"输出: {self.txt}")
                        if self.on_text:
                            self.on_text(self.txt, True)

                    if self.is_wake and self.is_exit:
                        self.is_wake = False
                        print("--------------------------------\n系统已退出\n")
                        if self.on_text:
                            self.on_text("", False)
                except Exception as e:
                    print(f"ASR错误: {str(e)}")
                finally:
                    self.audio_buffer = np.array([], dtype=np.float32)
            if self.exit_count >= exit_chunks:
                if self.is_wake:
                    self.is_wake = False
                    self.is_exit = True
                    print("--------------------------------\n长时间静音，系统已自动退出\n")
                    if self.on_text:
                        self.on_text("", False)
                self.exit_count = 0

def run_asr_thread(asr_callback, run_flag):
    """
    asr_callback: lambda text, is_wake
    run_flag: lambda -> bool, 控制是否退出
    """
    processor = VadAsrProcessor(on_text=asr_callback)
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            channels=1,
            dtype=FORMAT,
            callback=lambda indata, *_: processor.parse_result(indata)
        ):
            print("实时语音检测已启动 (CTRL+C 停止)")
            while run_flag():
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n检测已停止")
    except Exception as e:
        print(f"\033[31m设备错误: {str(e)}\033[0m")
        print("建议检查：1.麦克风权限 2.其他音频程序占用 3.更新声卡驱动")