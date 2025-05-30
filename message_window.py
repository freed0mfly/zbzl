from PyQt5 import QtWidgets

class BubbleTextEdit(QtWidgets.QTextEdit):
    def __init__(self, *args, font_size=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.setReadOnly(True)
        self.setFontFamily("微软雅黑")
        self.setFontPointSize(font_size)
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
    """左右分栏显示用户和助手消息的独立窗口"""
    def __init__(self, font_size=17):
        super().__init__()
        self.setWindowTitle("消息面板")
        self.resize(800, 600)
        self.setStyleSheet("background: #fff; border: none;")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        self.left_bubble = BubbleTextEdit(font_size=font_size)
        self.right_bubble = BubbleTextEdit(font_size=font_size)
        self.left_bubble.setVerticalScrollBarPolicy(QtWidgets.QScrollBarAlwaysOn)
        self.right_bubble.setVerticalScrollBarPolicy(QtWidgets.QScrollBarAlwaysOn)
        layout.addWidget(self.left_bubble, 1)
        layout.addWidget(self.right_bubble, 1)
        self.setLayout(layout)

    def append_user(self, text):
        self.left_bubble.append_bubble(text, "用户")

    def append_assistant(self, text):
        self.right_bubble.append_bubble(text, "助手")