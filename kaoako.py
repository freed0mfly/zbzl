import cv2

# 输入视频路径
input_video_path = r"D:\coding\projects\Python\human\Wav2Lip\input\33.mp4"
# 输出视频路径
output_video_path = r"D:\coding\projects\Python\human\Wav2Lip\input\44.mp4"
import cv2

# 保存第一帧图片的路径
first_frame_path = r"D:\coding\projects\Python\human\Wav2Lip\input/44.png"

import cv2


# 读取视频
cap = cv2.VideoCapture(input_video_path)

# 获取视频的帧率和尺寸
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 设置视频编解码器并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 读取并保存第一帧
ret, first_frame = cap.read()
if ret:
    cv2.imwrite(first_frame_path, first_frame)

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    out.write(frame)

# 重新定位到视频开头
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()