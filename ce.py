import sys
import threading
import asyncio
import time
import numpy as np
import cv2
import os
from PyQt5 import QtWidgets, QtCore, QtGui, QtMultimedia

from config import (
    API_KEY, BASE_URL, WAV2LIP_MODEL_PATH, IDLE_VIDEO_PATH, FACE_IMAGE_PATH, DEVICE,
    CHAT_FONT_SIZE, STAGE_FONT_SIZE
)
from llm import ChatBot
from tts import generate_speech
from face import load_model, preprocess_image, prepare_audio_batches, LipSyncPlayer
from asr import run_asr_thread

# ------------- 气泡文本和消息窗口定义 -------------

class BubbleTextEdit(QtWidgets.QTextEdit):
    def __init__(self, *args, font_size=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.setReadOnly(True)
        self.setFont(QtGui.QFont("微软雅黑", font_size))
        self.setStyleSheet(f"""
            QTextEdit {{
                background: #fff;
                border-radius: 18px;
                padding: 12px 10px 12px 10px;
                font-size: {font_size}px;
                color: #222;
                border: 1.5px solid #e2e2e2;
            }}
            QScrollBar:vertical {{
                background: #f5f6fa;
                width: 10px;
                margin: 2px 0 2px 0;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: #e2e2e2;
                border-radius: 5px;
                min-height: 20px;
            }}
        """)

    def append_bubble(self, text, speaker="用户"):
        if speaker == "用户":
            self.append(
                "<div style='margin:14px 0; text-align:right;'>"
                "<span style=\""
                "background: #fff;"
                "color:#b27b00;"
                "border-radius:32px;"
                "border: 2.5px solid #ff3030;"
                "box-shadow: 0 4px 18px rgba(255,48,48,0.10);"
                "padding:14px 28px;"
                "font-weight:600;"
                "display:inline-block;"
                "max-width:67%;"
                "line-height:1.8;"
                "word-break:break-all;"
                "\">"
                f"{text}"
                "</span>"
                "</div>")
        else:
            self.append(
                "<div style='margin:14px 0; text-align:left;'>"
                "<span style=\""
                "background: #fff;"
                "color:#2176ff;"
                "border-radius:32px;"
                "border: 2.5px solid #ff3030;"
                "box-shadow: 0 4px 18px rgba(255,48,48,0.08);"
                "padding:14px 28px;"
                "font-weight:600;"
                "display:inline-block;"
                "max-width:67%;"
                "line-height:1.8;"
                "word-break:break-all;"
                "\">"
                f"{text}"
                "</span>"
                "</div>")

class MessageWindow(QtWidgets.QWidget):
    """左：主对话（用户/助手），右：阶段/系统消息"""
    def __init__(self, font_size=17):
        super().__init__()
        self.setWindowTitle("消息面板")
        self.resize(900, 600)
        self.setStyleSheet("background: #fff; border: none;")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        self.left_bubble = BubbleTextEdit(font_size=font_size)
        self.right_bubble = BubbleTextEdit(font_size=font_size)
        self.left_bubble.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.right_bubble.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        layout.addWidget(self.left_bubble, 1)
        layout.addWidget(self.right_bubble, 1)
        self.setLayout(layout)

    def append_dialogue(self, text, speaker):
        self.left_bubble.append_bubble(text, speaker)

    def append_stage(self, text):
        self.right_bubble.append_bubble(text, "助手")

# ------------- 主窗口/人像窗口定义 -------------

class FaceDisplayWidget(QtWidgets.QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("background:transparent;")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setScaledContents(True)

class AudioPlayer(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = QtMultimedia.QMediaPlayer()
        self.player.setVolume(100)
        self.player.mediaStatusChanged.connect(self.handle_status)

    def play(self, audio_path):
        if self.player.state() == QtMultimedia.QMediaPlayer.PlayingState:
            self.player.stop()
        url = QtCore.QUrl.fromLocalFile(os.path.abspath(audio_path))
        self.player.setMedia(QtMultimedia.QMediaContent(url))
        self.player.play()

    def handle_status(self, status):
        if status in (QtMultimedia.QMediaPlayer.EndOfMedia, QtMultimedia.QMediaPlayer.InvalidMedia):
            self.finished.emit()

class DigitalHumanUI(QtWidgets.QWidget):
    append_history_signal = QtCore.pyqtSignal(str, str)
    show_frame_signal = QtCore.pyqtSignal(QtGui.QPixmap)
    idle_signal = QtCore.pyqtSignal()
    stage_signal = QtCore.pyqtSignal(str)
    asr_text_signal = QtCore.pyqtSignal(str)
    asr_status_signal = QtCore.pyqtSignal(bool, bool)
    play_video_frames_signal = QtCore.pyqtSignal(list, str, float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("数字人问答演示")
        self.resize(1200, 800)
        self.setStyleSheet("background: #fff;")
        self.asr_running = False
        self.asr_thread = None
        self._asr_wake = False
        self.busy = False
        self.idle_video_thread = None
        self.idle_video_running = threading.Event()
        self._last_pixmap = None
        self._last_face_size = (0, 0)
        self.audio_player = AudioPlayer()
        self.sync_timer = QtCore.QTimer(self)
        self.sync_timer.timeout.connect(self._sync_frame_with_audio)
        self.video_frames = []
        self.video_frame_count = 0
        self.target_fps = 25
        self.audio_total_ms = 0

        # 消息窗口
        self.message_window = MessageWindow(font_size=CHAT_FONT_SIZE)
        self.message_window.show()

        self.init_ui()
        self.init_resources()
        self.append_history_signal.connect(self.append_history)
        self.show_frame_signal.connect(self._show_pixmap_mainthread)
        self.idle_signal.connect(self.show_idle)
        self.stage_signal.connect(self.show_stage)
        self.asr_text_signal.connect(self.on_asr_text)
        self.asr_status_signal.connect(self.update_asr_status)
        self.play_video_frames_signal.connect(self.play_video_frames)
        self.show_idle()
        self.show_stage("系统待命...")

    def init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 中间人像区（居中1/3宽，最大高度0.8）
        self.center_panel = QtWidgets.QWidget(self)
        self.center_panel.setStyleSheet("background:transparent;")
        center_layout = QtWidgets.QHBoxLayout(self.center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        center_layout.addStretch(1)
        self.face_label = FaceDisplayWidget(self.center_panel)
        center_layout.addWidget(self.face_label, 3)
        center_layout.addStretch(1)
        main_layout.addWidget(self.center_panel, stretch=10)

        # 输入区（底部）
        self.input_panel = QtWidgets.QWidget(self)
        input_layout = QtWidgets.QHBoxLayout(self.input_panel)
        input_layout.setContentsMargins(48, 0, 48, 18)
        input_layout.setSpacing(16)
        self.input_box = QtWidgets.QLineEdit(self.input_panel)
        self.input_box.setPlaceholderText("请输入你的问题或说话（支持语音唤醒）...")
        self.input_box.setStyleSheet("""
            font-size:17px; border-radius:12px; padding:10px; background:#f9f9f9;
            border:2px solid #e2e2e2; color:#222;
        """)
        self.send_btn = QtWidgets.QPushButton("发送", self.input_panel)
        self.send_btn.setStyleSheet("""
            font-size:17px; padding:10px 28px; background:#4faaff; color:#15181a; border-radius:10px;
            font-weight:bold;
        """)
        self.send_btn.clicked.connect(self.on_submit)
        self.asr_btn = QtWidgets.QPushButton("🎤", self.input_panel)
        self.asr_btn.setStyleSheet("""
            font-size:24px; padding:10px 16px; background:#e2e2e2; color:#4faaff; border-radius:50%;
            border:2px solid #e2e2e2;
        """)
        self.asr_btn.setCheckable(True)
        self.asr_btn.clicked.connect(self.on_toggle_asr)

        input_layout.addWidget(self.input_box, 10)
        input_layout.addWidget(self.send_btn, 2)
        input_layout.addWidget(self.asr_btn, 1)
        self.input_panel.setStyleSheet("background:transparent;")
        main_layout.addWidget(self.input_panel, stretch=1)

        self.asr_status_label = QtWidgets.QLabel("语音识别：关闭 | 唤醒：未唤醒", self)
        self.asr_status_label.setStyleSheet("color:#4faaff; font-size:15px; font-weight:bold; background:transparent;")
        self.asr_status_label.setAlignment(QtCore.Qt.AlignRight)
        self.asr_status_label.setFixedHeight(24)
        main_layout.addWidget(self.asr_status_label, alignment=QtCore.Qt.AlignRight)

    def resizeEvent(self, event):
        # 保证中间人像区1/3宽，最大高度0.8
        total_w = self.width()
        total_h = self.height()
        face_w = max(int(total_w / 3), 1)
        face_h = min(int(total_h * 0.8), int(self.center_panel.height()))
        if face_w < 1: face_w = 300
        if face_h < 1: face_h = 400
        self.face_label.setMinimumSize(face_w, face_h)
        self.face_label.setMaximumSize(face_w, face_h)
        self._last_face_size = (face_w, face_h)
        if self._last_pixmap is not None:
            self._show_pixmap_mainthread(self._last_pixmap)
        event.accept()

    def update_center_panel_geometry(self):
        self.resizeEvent(QtGui.QResizeEvent(self.size(), self.size()))

    def init_resources(self):
        self.bot = ChatBot(
            api_key=API_KEY,
            base_url=BASE_URL,
            log_dir="logs",
            default_background="你是一个知识渊博的助手，能够简洁地回答问题。",
            default_prefix="请简洁地回答下述问题："
        )
        self.model = load_model(WAV2LIP_MODEL_PATH, DEVICE)
        self.face_img, self.face_coords, self.orig_image = preprocess_image(FACE_IMAGE_PATH, device=DEVICE)
        self.idle_video_path = IDLE_VIDEO_PATH
        self.lip_player = LipSyncPlayer(self.model, DEVICE, self.orig_image, self.face_coords, fps=25)

    def show_idle(self):
        self.stop_sync()
        self.idle_video_running.set()
        if self.idle_video_thread is None or not self.idle_video_thread.is_alive():
            self.idle_video_thread = threading.Thread(target=self.play_idle_video, daemon=True)
            self.idle_video_thread.start()

    def play_idle_video(self):
        while self.idle_video_running.is_set():
            cap = cv2.VideoCapture(self.idle_video_path)
            if not cap.isOpened():
                print(f"无法打开idle视频：{self.idle_video_path}")
                return
            while self.idle_video_running.is_set():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                w, h = self._last_face_size
                if w is None or h is None or w < 1 or h < 1:
                    w, h = 300, 400
                frame = cv2.resize(frame, (w, h))
                qtimg = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888)
                pix = QtGui.QPixmap.fromImage(qtimg)
                self._last_pixmap = pix
                self.show_frame_signal.emit(pix)
                time.sleep(1.0 / 25)
            cap.release()

    def stop_idle_video(self):
        self.idle_video_running.clear()

    def stop_sync(self):
        if self.sync_timer.isActive():
            self.sync_timer.stop()
        self.video_frames = []
        self.video_frame_count = 0
        self.audio_total_ms = 0

    @QtCore.pyqtSlot(QtGui.QPixmap)
    def _show_pixmap_mainthread(self, pix):
        w, h = self._last_face_size
        if w is None or h is None or w < 1 or h < 1:
            w, h = 300, 400
        scaled = pix.scaled(w, h, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
        self.face_label.setPixmap(scaled)
        self._last_pixmap = pix

    @QtCore.pyqtSlot(str, str)
    def append_history(self, speaker, text):
        # 只把主对话（用户/助手）写到左侧
        self.message_window.append_dialogue(text, speaker)

    @QtCore.pyqtSlot(str)
    def show_stage(self, text):
        # 阶段信息写到右侧
        self.message_window.append_stage(text)

    @QtCore.pyqtSlot(np.ndarray)
    def show_video_frame(self, frame):
        w, h = self._last_face_size
        if w is None or h is None or w < 1 or h < 1:
            w, h = 300, 400
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (w, h))
        qtimg = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qtimg)
        self.face_label.setPixmap(pix)
        self._last_pixmap = pix

    @QtCore.pyqtSlot(list, str, float)
    def play_video_frames(self, frames, audio_path, audio_duration):
        self.stop_idle_video()
        self.stop_sync()
        self.video_frames = frames
        self.video_frame_count = len(frames)
        self.target_fps = 25
        self.audio_total_ms = int(audio_duration * 1000)
        self.audio_player.play(audio_path)
        self.sync_timer.start(20)

    def _sync_frame_with_audio(self):
        ms = self.audio_player.player.position()
        if ms <= 0:
            return
        idx = int(ms * self.target_fps / 1000)
        idx = min(idx, self.video_frame_count - 1)
        if 0 <= idx < self.video_frame_count:
            frame = self.video_frames[idx]
            self.show_video_frame(frame)
        if ms >= self.audio_total_ms - 20 or idx >= self.video_frame_count - 1:
            self.sync_timer.stop()
            self.idle_signal.emit()

    @QtCore.pyqtSlot(str)
    def on_asr_text(self, text):
        if self.busy:
            self.stage_signal.emit("正在播报回答，请稍后再提问。")
            return
        text = text.strip()
        if text:
            self.input_box.setText(text)
            self.on_submit()

    @QtCore.pyqtSlot(bool, bool)
    def update_asr_status(self, running, wake):
        self.asr_running = running
        self._asr_wake = wake
        s = f"语音识别：{'开启' if running else '关闭'} | 唤醒：{'已唤醒' if wake else '未唤醒'}"
        color = "#4faaff" if running else "#555"
        wcolor = "#ff5050" if wake else "#4faaff"
        self.asr_status_label.setText(s)
        self.asr_status_label.setStyleSheet(f"color:{wcolor if wake else color}; font-size:15px; font-weight:bold; background:transparent;")
        if not running:
            self.asr_btn.setChecked(False)
            self.asr_btn.setText("🎤")
        else:
            self.asr_btn.setChecked(True)
            self.asr_btn.setText("⏹")

    def on_toggle_asr(self):
        if self.asr_running:
            self._stop_asr()
        else:
            self._start_asr()

    def _start_asr(self):
        if self.asr_running:
            return
        self.asr_running = True
        self.asr_btn.setText("⏹")
        self.asr_status_signal.emit(True, False)
        self.asr_thread = threading.Thread(target=self.start_asr, daemon=True)
        self.asr_thread.start()

    def _stop_asr(self):
        if not self.asr_running:
            return
        self.asr_running = False
        self.asr_btn.setText("🎤")
        self.asr_status_signal.emit(False, False)

    def start_asr(self):
        def asr_callback(text, wake_state):
            self.asr_status_signal.emit(True, wake_state)
            if text and wake_state and not self.busy:
                self.asr_text_signal.emit(text)
        try:
            run_asr_thread(asr_callback, lambda: self.asr_running)
        except Exception as e:
            self.asr_status_signal.emit(False, False)

    def on_submit(self):
        if self.busy:
            self.stage_signal.emit("正在播报上一个回答，请稍后...")
            return
        question = self.input_box.text().strip()
        if not question:
            return
        self.input_box.setText("")
        self.append_history("用户", question)
        self.show_stage("开始处理...")
        self.busy = True
        threading.Thread(target=self.process_conversation, args=(question,)).start()

    def process_conversation(self, question):
        import soundfile as sf
        t0 = time.perf_counter()
        self.stage_signal.emit("等待大模型回复...")
        t1 = time.perf_counter()
        answer = self.bot.chat(question)
        t2 = time.perf_counter()
        self.append_history_signal.emit("助手", answer)
        self.stage_signal.emit(f"大模型回复完成，耗时：{t2-t1:.2f}s")
        self.stage_signal.emit("正在合成语音...")

        if not answer or len(answer.strip()) < 2:
            self.stage_signal.emit("回答内容过短，跳过语音与口型合成。")
            time.sleep(0.5)
            self.idle_signal.emit()
            self.busy = False
            return

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        t3 = time.perf_counter()
        try:
            audio_path = loop.run_until_complete(generate_speech(answer))
        except Exception as e:
            self.stage_signal.emit(f"语音合成失败：{e}")
            self.idle_signal.emit()
            self.busy = False
            return
        t4 = time.perf_counter()
        if not audio_path or not os.path.exists(audio_path) or os.path.getsize(audio_path) < 800:
            self.stage_signal.emit("语音文件生成失败或内容太短，跳过口型合成。")
            self.idle_signal.emit()
            self.busy = False
            return
        audio_info = sf.info(audio_path)
        audio_duration = float(audio_info.duration)
        self.stage_signal.emit(f"语音合成完成，耗时：{t4-t3:.2f}s")
        self.stage_signal.emit("正在生成嘴型动画...")

        t5 = time.perf_counter()
        gen = prepare_audio_batches(audio_path, self.face_img, self.face_coords)
        infer_time, all_frames = self.lip_player.infer_frames(gen)
        self.stage_signal.emit(f"视频帧推理完成，耗时{infer_time:.2f}s")
        self.stage_signal.emit("正在播放语音与动画...")

        self.play_video_frames_signal.emit(all_frames, audio_path, audio_duration)
        t6 = time.perf_counter()
        self.busy = False

if __name__ == "__main__":
    os.environ["QT_FONT_DPI"] = "96"
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("digitalhuman.app")
    sys.stdout.reconfigure(encoding='utf-8')
    app = QtWidgets.QApplication(sys.argv)
    win = DigitalHumanUI()
    win.show()
    sys.exit(app.exec_())