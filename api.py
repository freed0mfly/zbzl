from fastapi import FastAPI
import asyncio
from tts import generate_speech, play_audio
from llm import ChatBot
from face import load_model, preprocess_image, prepare_audio_batches, wav2lip_sync_play
import os
import torch
app = FastAPI()

# 初始化聊天机器人
api_key = "sk-kHwb21aa8Ef1ov5eBXMAVdo127bxWF8445QNqmwlbXm59lHG"
base_url = "https://api.hunyuan.cloud.tencent.com/v1"
bot = ChatBot(api_key, base_url)
bot.set_background("你是一个知识渊博的助手，能够简洁地回答问题。")
bot.set_prompt("请保持回答简短。")

# 加载Wav2Lip模型
check_path = r"D:\coding\projects\Python\human\Wav2Lip\checkpoints\wav2lip_gan.pth"
img_path = r"D:\coding\projects\Python\human\Wav2Lip\input\1.png"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model(check_path)
face_img, face_coords, orig_image = preprocess_image(img_path, device=device)

@app.post("/chat")
async def chat(user_input: str):
    # 获取AI回复
    response = bot.chat(user_input)
    # 生成语音
    audio_file = await generate_speech(response)
    # 准备音频批次
    gen = prepare_audio_batches(audio_file, face_img, face_coords)
    # 口型同步播放
    wav2lip_sync_play(
        model, gen, device,
        orig_image=orig_image, coords=face_coords,
        audio_path=audio_file,
        window_name="Wav2Lip 数字人对话系统"
    )
    return {"response": response, "audio_file": audio_file}