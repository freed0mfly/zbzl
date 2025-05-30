#face
import os
import numpy as np
import cv2
import torch
import threading
import time
import Wav2Lip.audio as audio
import subprocess
import sounddevice as sd
import soundfile as sf
import Wav2Lip.face_detection as face_detection
from Wav2Lip.models import Wav2Lip

mel_step_size = 16

def _load(checkpoint_path, device):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path, device):
    model = Wav2Lip()
    print(f"从 {path} 加载模型")
    checkpoint = _load(path, device)
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
    # 补帧：确保嘴型帧不会太短
    min_frames = int((len(mel[0]) / 80) * fps / (16000 / 80))
    if len(mel_chunks) < min_frames and len(mel_chunks) > 0:
        repeat = int(np.ceil(min_frames / len(mel_chunks)))
        mel_chunks = (mel_chunks * repeat)[:min_frames]

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

class LipSyncPlayer:
    def __init__(self, model, device, orig_image, face_coords, fps=25):
        self.model = model
        self.device = device
        self.orig_image = orig_image
        self.face_coords = face_coords
        self.fps = fps

    def infer_frames(self, batch_gen):
        """
        只做推理，返回(耗时, all_frames)
        """
        t_infer_start = time.perf_counter()
        all_frames = []
        gen_iter = iter(batch_gen)
        for img_batch, mel_batch, frames, coords_batch in gen_iter:
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)
            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            pred = pred.astype(np.uint8)
            for p in pred:
                y1, y2, x1, x2 = self.face_coords
                h, w = y2 - y1, x2 - x1
                if h <= 0 or w <= 0:
                    continue
                p_resized = cv2.resize(p, (w, h))
                show_img = self.orig_image.copy()
                show_img[y1:y2, x1:x2] = p_resized
                all_frames.append(show_img.copy())
        t_infer_end = time.perf_counter()
        infer_time = t_infer_end - t_infer_start
        return infer_time, all_frames

    def play_frames(self, audio_path, all_frames, frame_callback):
        """
        只做播放，返回耗时
        """
        # 加载音频数据
        data, samplerate = sf.read(audio_path)
        duration = len(data) / samplerate
        n_frames = len(all_frames)
        frame_interval = 1.0 / self.fps
        total_frames_by_audio = int(duration * self.fps)

        stop_event = threading.Event()
        def audio_thread_func():
            sd.play(data, samplerate)
            sd.wait()
            stop_event.set()
        audio_thread = threading.Thread(target=audio_thread_func)
        audio_thread.start()

        t_play_start = time.perf_counter()
        t_start = time.time()
        frame_idx = 0
        last_frame = None
        while not stop_event.is_set():
            now = time.time()
            expected_frame = int((now - t_start) * self.fps)
            if expected_frame >= total_frames_by_audio:
                break
            if frame_idx < n_frames:
                frame_callback(all_frames[frame_idx])
                last_frame = all_frames[frame_idx]
            else:
                if last_frame is not None:
                    frame_callback(last_frame)
            frame_idx += 1
            next_time = t_start + frame_idx * frame_interval
            time.sleep(max(0, next_time - time.time()))
        audio_thread.join()
        t_play_end = time.perf_counter()
        play_time = t_play_end - t_play_start
        return play_time