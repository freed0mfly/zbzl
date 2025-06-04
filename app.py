import sys
import os
import threading

import cv2
import time
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui, QtMultimedia


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


class DigitalHumanPreview(QtWidgets.QWidget):
    show_frame_signal = QtCore.pyqtSignal(QtGui.QPixmap)
    idle_signal = QtCore.pyqtSignal()
    stage_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("数字人界面预览工具")
        self.resize(1200, 800)
        self.setStyleSheet("background: #DFDFDF;")
        self.idle_video_thread = None
        self.idle_video_running = threading.Event()
        self._last_pixmap = None
        self._last_face_size = (0, 0)
        self.audio_player = AudioPlayer()
        self.video_frames = []
        self.video_frame_count = 0
        self.target_fps = 25
        self.audio_total_ms = 0
        self.current_face_image = None
        self.current_idle_video = None

        # 默认资源路径
        self.default_face_image = "images/default_face.jpg"
        self.default_idle_video = "videos/default_idle.mp4"

        # 消息窗口
        self.message_window = MessageWindow(font_size=17)
        self.message_window.show()

        self.init_ui()
        self.show_stage("系统就绪，请选择图片和视频...")
        self.show_frame_signal.connect(self._show_pixmap_mainthread)
        self.idle_signal.connect(self.show_idle)
        self.stage_signal.connect(self.show_stage)
        self.audio_player.finished.connect(self.on_audio_play_finished)

    def init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # 资源选择区
        resource_layout = QtWidgets.QHBoxLayout()
        resource_layout.setSpacing(12)

        self.face_image_btn = QtWidgets.QPushButton("选择人脸图片", self)
        self.face_image_btn.setStyleSheet("padding:8px 16px; font-size:15px;")
        self.face_image_btn.clicked.connect(self.select_face_image)

        self.idle_video_btn = QtWidgets.QPushButton("选择待机视频", self)
        self.idle_video_btn.setStyleSheet("padding:8px 16px; font-size:15px;")
        self.idle_video_btn.clicked.connect(self.select_idle_video)

        self.show_face_btn = QtWidgets.QPushButton("显示人脸", self)
        self.show_face_btn.setStyleSheet("padding:8px 16px; font-size:15px;")
        self.show_face_btn.clicked.connect(self.display_face_image)

        self.play_idle_btn = QtWidgets.QPushButton("播放待机视频", self)
        self.play_idle_btn.setStyleSheet("padding:8px 16px; font-size:15px;")
        self.play_idle_btn.clicked.connect(self.play_idle_video_manual)

        resource_layout.addWidget(self.face_image_btn)
        resource_layout.addWidget(self.idle_video_btn)
        resource_layout.addWidget(self.show_face_btn)
        resource_layout.addWidget(self.play_idle_btn)
        main_layout.addLayout(resource_layout)

        # 中间人像区
        self.center_panel = QtWidgets.QWidget(self)
        self.center_panel.setStyleSheet("background:transparent; border: 2px dashed #ccc; border-radius: 8px;")
        center_layout = QtWidgets.QHBoxLayout(self.center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        center_layout.addStretch(1)
        self.face_label = FaceDisplayWidget(self.center_panel)
        center_layout.addWidget(self.face_label, 3)
        center_layout.addStretch(1)
        main_layout.addWidget(self.center_panel, stretch=10)

        # 控制区
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setSpacing(16)

        self.simulate_btn = QtWidgets.QPushButton("模拟对话", self)
        self.simulate_btn.setStyleSheet(
            "padding:10px 28px; font-size:17px; background:#4faaff; color:#15181a; border-radius:10px; font-weight:bold;")
        self.simulate_btn.clicked.connect(self.simulate_conversation)

        self.audio_btn = QtWidgets.QPushButton("选择音频", self)
        self.audio_btn.setStyleSheet("padding:10px 28px; font-size:17px;")
        self.audio_btn.clicked.connect(self.select_audio)

        self.play_audio_btn = QtWidgets.QPushButton("播放音频", self)
        self.play_audio_btn.setStyleSheet("padding:10px 28px; font-size:17px;")
        self.play_audio_btn.clicked.connect(self.play_audio)
        self.play_audio_btn.setEnabled(False)

        control_layout.addWidget(self.simulate_btn)
        control_layout.addWidget(self.audio_btn)
        control_layout.addWidget(self.play_audio_btn)
        main_layout.addLayout(control_layout)

        # 状态显示
        self.status_label = QtWidgets.QLabel("状态：就绪", self)
        self.status_label.setStyleSheet("color:#4faaff; font-size:15px; font-weight:bold; background:transparent;")
        self.status_label.setAlignment(QtCore.Qt.AlignLeft)
        main_layout.addWidget(self.status_label)

        self.current_audio = None

    def resizeEvent(self, event):
        total_w = self.width()
        total_h = self.height()
        face_w = max(int(total_w / 3), 1)
        face_h = min(int(total_h * 0.6), int(self.center_panel.height()))
        if face_w < 1: face_w = 300
        if face_h < 1: face_h = 400
        self.face_label.setMinimumSize(face_w, face_h)
        self.face_label.setMaximumSize(face_w, face_h)
        self._last_face_size = (face_w, face_h)
        if self._last_pixmap is not None:
            self._show_pixmap_mainthread(self._last_pixmap)
        event.accept()

    def select_face_image(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择人脸图片", "", "图片文件 (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.current_face_image = file_path
            self.status_label.setText(f"状态：已选择人脸图片 - {os.path.basename(file_path)}")
            self.show_stage(f"已选择人脸图片: {os.path.basename(file_path)}")

    def select_idle_video(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择待机视频", "", "视频文件 (*.mp4 *.avi *.mov)"
        )
        if file_path:
            self.current_idle_video = file_path
            self.status_label.setText(f"状态：已选择待机视频 - {os.path.basename(file_path)}")
            self.show_stage(f"已选择待机视频: {os.path.basename(file_path)}")

    def select_audio(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择音频文件", "", "音频文件 (*.wav *.mp3 *.ogg)"
        )
        if file_path:
            self.current_audio = file_path
            self.status_label.setText(f"状态：已选择音频 - {os.path.basename(file_path)}")
            self.play_audio_btn.setEnabled(True)
            self.show_stage(f"已选择音频: {os.path.basename(file_path)}")

    def display_face_image(self):
        if not self.current_face_image:
            self.show_stage("请先选择人脸图片")
            return

        self.stop_idle_video()
        try:
            image = cv2.imread(self.current_face_image)
            if image is None:
                self.show_stage(f"无法加载图片: {self.current_face_image}")
                return

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            w, h = self._last_face_size
            if w < 1 or h < 1:
                w, h = 300, 400
            image = cv2.resize(image, (w, h))
            qtimg = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.strides[0],
                                 QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qtimg)
            self._last_pixmap = pix
            self.show_frame_signal.emit(pix)
            self.show_stage(f"已显示人脸图片: {os.path.basename(self.current_face_image)}")
        except Exception as e:
            self.show_stage(f"显示图片时出错: {str(e)}")

    def show_idle(self):
        self.stop_sync()
        if not self.current_idle_video:
            self.show_stage("请先选择待机视频")
            return

        self.idle_video_running.set()
        if self.idle_video_thread is None or not self.idle_video_thread.is_alive():
            self.idle_video_thread = threading.Thread(target=self.play_idle_video, daemon=True)
            self.idle_video_thread.start()

    def play_idle_video(self):
        if not self.current_idle_video:
            return

        while self.idle_video_running.is_set():
            cap = cv2.VideoCapture(self.current_idle_video)
            if not cap.isOpened():
                self.show_stage(f"无法打开视频: {self.current_idle_video}")
                break

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25

            while self.idle_video_running.is_set():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                w, h = self._last_face_size
                if w < 1 or h < 1:
                    w, h = 300, 400
                frame = cv2.resize(frame, (w, h))
                qtimg = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0],
                                     QtGui.QImage.Format_RGB888)
                pix = QtGui.QPixmap.fromImage(qtimg)
                self._last_pixmap = pix
                self.show_frame_signal.emit(pix)
                time.sleep(1.0 / fps)

            cap.release()

    def play_idle_video_manual(self):
        self.show_idle()
        self.show_stage(f"正在播放待机视频: {os.path.basename(self.current_idle_video)}")

    def stop_idle_video(self):
        self.idle_video_running.clear()

    def stop_sync(self):
        self.video_frames = []
        self.video_frame_count = 0
        self.audio_total_ms = 0

    @QtCore.pyqtSlot(QtGui.QPixmap)
    def _show_pixmap_mainthread(self, pix):
        w, h = self._last_face_size
        if w < 1 or h < 1:
            w, h = 300, 400
        scaled = pix.scaled(w, h, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
        self.face_label.setPixmap(scaled)
        self._last_pixmap = pix

    def show_stage(self, text):
        self.message_window.append_stage(text)
        self.status_label.setText(f"状态：{text}")

    def play_audio(self):
        if not self.current_audio:
            self.show_stage("请先选择音频文件")
            return

        self.show_stage(f"正在播放音频: {os.path.basename(self.current_audio)}")
        self.audio_player.play(self.current_audio)

    def on_audio_play_finished(self):
        self.show_stage("音频播放完成")

    def simulate_conversation(self):
        self.message_window.append_dialogue("这是一个模拟的用户问题", "用户")
        self.show_stage("模拟大模型回复中...")

        # 模拟回复延迟
        QtCore.QTimer.singleShot(1500, self.show_simulated_response)

    def show_simulated_response(self):
        response = (
            "这是一个模拟的数字人回复。在实际应用中，"
            "这里会是大语言模型生成的回答，并通过语音合成和唇形同步技术展示。"
            "你可以使用这个工具来测试不同图片和视频的显示效果。"
        )
        self.message_window.append_dialogue(response, "助手")
        self.show_stage("模拟回复完成")


if __name__ == "__main__":
    os.environ["QT_FONT_DPI"] = "96"
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("digitalhuman.preview")
    sys.stdout.reconfigure(encoding='utf-8')
    app = QtWidgets.QApplication(sys.argv)
    win = DigitalHumanPreview()
    win.show()
    sys.exit(app.exec_())