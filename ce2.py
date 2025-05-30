import os
import time
import threading

import numpy as np
import cv2
import torch
import Wav2Lip.audio as audio
import subprocess
import sounddevice as sd
import soundfile as sf
import Wav2Lip.face_detection as face_detection
from Wav2Lip.models import Wav2Lip

mel_step_size = 16

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print(f"从 {path} 加载模型")
    checkpoint = _load(path)
    new_s = {}
    for k, v in checkpoint["state_dict"].items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    model = model.to(device)
    return model.eval()

def preprocess_image(
    image_path,
    pads=[0,10,0,0],
    box=[-1,-1,-1,-1],
    img_size=96,
    face_det_batch_size=16,
    nosmooth=False,
    device='cuda'
):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    if box[0] == -1:
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)
        predictions = detector.get_detections_for_batch(np.array([image]))
        rect = predictions[0]
        if rect is None:
            raise RuntimeError("未检测到人脸")
        pady1, pady2, padx1, padx2 = pads
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
    else:
        y1, y2, x1, x2 = box
    face = image[y1:y2, x1:x2]
    face = cv2.resize(face, (img_size, img_size))
    coords = (y1, y2, x1, x2)
    return face, coords, image

def prepare_audio_batches(
    audio_path,
    face_img,
    face_coords,
    static=True,
    fps=25,
    mel_step_size=16,
    wav2lip_batch_size=128,
    img_size=96
):
    if not audio_path.endswith('.wav'):
        print('提取音频...')
        command = f'ffmpeg -y -i "{audio_path}" -strict -2 temp/temp.wav'
        subprocess.call(command, shell=True)
        audio_path = 'temp/temp.wav'
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('梅尔频谱包含NaN值，请检查音频质量')
    mel_idx_multiplier = 80. / fps
    mel_chunks = []
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    for i, m in enumerate(mel_chunks):
        frame_to_save = face_img.copy()
        face = face_img.copy()
        coords = face_coords
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)
        if len(img_batch) >= wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
            img_masked = img_batch.copy()
            img_masked[:, img_size // 2:] = 0
            img_input = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_input = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            yield img_input, mel_input, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
        img_masked = img_batch.copy()
        img_masked[:, img_size // 2:] = 0
        img_input = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_input = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        yield img_input, mel_input, frame_batch, coords_batch

def play_audio_stream(audio_path, start_event, stop_event):
    data, samplerate = sf.read(audio_path)
    if len(data.shape) == 1:
        data = data[:, None]
    def callback(outdata, frames, time_, status):
        start_event.wait()
        if stop_event.is_set():
            raise sd.CallbackAbort
        chunk = data[callback.idx:callback.idx+frames]
        if len(chunk) < frames:
            outdata[:len(chunk)] = chunk
            outdata[len(chunk):] = 0
            stop_event.set()
            raise sd.CallbackStop
        else:
            outdata[:] = chunk
        callback.idx += frames
    callback.idx = 0
    with sd.OutputStream(channels=data.shape[1], samplerate=samplerate, callback=callback):
        start_event.wait()
        while not stop_event.is_set() and callback.idx < len(data):
            sd.sleep(100)

import queue

def wav2lip_sync_play(
    model,
    gen,
    device,
    orig_image,
    coords,
    audio_path,
    fps=25,
    window_size=(224, 336),
    show_window=True,
    window_name="Wav2Lip Result"
):
    import threading
    import time

    frame_interval = 1.0 / fps
    frame_count = 0

    # 1. 先推理出第一批嘴型帧，保证至少一帧嘴型可用
    frame_cache = []
    gen_iter = iter(gen)
    try:
        img_batch, mel_batch, frames, coords_batch = next(gen_iter)
        import torch
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        with torch.no_grad():
            pred = model(mel_batch, img_batch)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        pred = pred.astype(np.uint8)
        for p in pred:
            y1, y2, x1, x2 = coords
            h, w = y2 - y1, x2 - x1
            if h <= 0 or w <= 0:
                continue
            p_resized = cv2.resize(p, (w, h))
            show_img = orig_image.copy()
            show_img[y1:y2, x1:x2] = p_resized
            frame_cache.append(show_img.copy())
    except StopIteration:
        print("音频太短，没有帧可推理。")
        return

    # 2. 准备好第一帧后再启动音频播放
    start_event = threading.Event()
    stop_event = threading.Event()
    def play_audio_thread_func():
        from soundfile import read
        import sounddevice as sd
        data, samplerate = sf.read(audio_path)
        if len(data.shape) == 1:
            data = data[:, None]
        def callback(outdata, frames, time_, status):
            start_event.wait()
            if stop_event.is_set():
                raise sd.CallbackAbort
            chunk = data[callback.idx:callback.idx+frames]
            if len(chunk) < frames:
                outdata[:len(chunk)] = chunk
                outdata[len(chunk):] = 0
                stop_event.set()
                raise sd.CallbackStop
            else:
                outdata[:] = chunk
            callback.idx += frames
        callback.idx = 0
        with sd.OutputStream(channels=data.shape[1], samplerate=samplerate, callback=callback):
            start_event.wait()
            while not stop_event.is_set() and callback.idx < len(data):
                sd.sleep(100)

    audio_thread = threading.Thread(target=play_audio_thread_func)
    audio_thread.start()

    # 3. 播放口型帧（第一批用 frame_cache，后续用gen）
    start_time = time.time()
    start_event.set()  # 现在才开始音频播放！

    def show_cv(img):
        show_img_disp = cv2.resize(img, window_size)
        cv2.imshow(window_name, show_img_disp)
        cv2.waitKey(1)

    # 先显示缓存帧
    for img in frame_cache:
        # 音视频时间对齐
        target_time = start_time + frame_count * frame_interval
        now = time.time()
        if now < target_time:
            time.sleep(target_time - now)
        show_cv(img)
        frame_count += 1

    # 后面继续推理并播放
    for img_batch, mel_batch, frames, coords_batch in gen_iter:
        import torch
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        with torch.no_grad():
            pred = model(mel_batch, img_batch)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        pred = pred.astype(np.uint8)
        for p in pred:
            y1, y2, x1, x2 = coords
            h, w = y2 - y1, x2 - x1
            if h <= 0 or w <= 0:
                continue
            p_resized = cv2.resize(p, (w, h))
            show_img = orig_image.copy()
            show_img[y1:y2, x1:x2] = p_resized
            # 时间对齐
            target_time = start_time + frame_count * frame_interval
            now = time.time()
            if now < target_time:
                time.sleep(target_time - now)
            show_cv(show_img)
            frame_count += 1
            if stop_event.is_set():
                break
        if stop_event.is_set():
            break

    stop_event.set()
    audio_thread.join()
    cv2.destroyAllWindows()

def show_image_idle(orig_image, window_size=(224, 336), window_name="Wav2Lip"):
    img = cv2.resize(orig_image, window_size)
    cv2.imshow(window_name, img)
    # 只刷新，不阻塞主线程，需外部循环配合
    cv2.waitKey(1)

def wait_for_audio_path():
    audio_path = input("请输入要讲解的音频文件路径（或输入 q 退出）：").strip()
    if audio_path.lower() == "q":
        return None
    if not os.path.exists(audio_path):
        print("音频文件不存在，请重新输入。")
        return ""
    return audio_path

if __name__ == '__main__':
    check_path = r"D:\coding\projects\Python\human\Wav2Lip\checkpoints\wav2lip_gan.pth"
    img_path = r"D:\coding\projects\Python\human\Wav2Lip\input\1.png"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(check_path)
    face_img, face_coords, orig_image = preprocess_image(img_path, device=device)
    window_name = "Wav2Lip 自助讲解"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("==== 自助讲解员系统 ====")
    print("系统空闲时自动展示图片，收到音频地址后自动讲解并口型同步。按 'q' 退出。")

    while True:
        # 持续展示图片，直到有音频输入
        print("空闲展示中，等待输入讲解音频路径 ...")
        while True:
            show_image_idle(orig_image, window_name=window_name)
            # 检查窗口是否被关闭/按下q
            key = cv2.waitKey(100)
            if key == ord('q'):
                print("用户退出。")
                cv2.destroyAllWindows()
                exit(0)
            # 检查命令行输入是否就绪（非阻塞方式，提示用户切到命令行输入）
            # 推荐只在命令行输入后进入下一步
            break

        # 等待用户输入音频路径
        audio_path = wait_for_audio_path()
        if audio_path is None:
            break
        if not audio_path:
            continue

        try:
            gen = prepare_audio_batches(audio_path, face_img, face_coords)
            wav2lip_sync_play(
                model, gen, device,
                orig_image=orig_image, coords=face_coords,
                audio_path=audio_path,
                window_name=window_name
            )
        except Exception as e:
            print(f"音频处理或合成异常: {e}")

    cv2.destroyAllWindows()
    print("讲解系统已关闭。")