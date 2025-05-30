import gradio as gr
import asyncio
import os
from llm import ChatBot
from tts import generate_speech, play_audio
from face import load_model, preprocess_image, prepare_audio_batches, wav2lip_sync_play
import threading
import time
import torch
# 初始化LLM聊天机器人
api_key = "sk-kHwb21aa8Ef1ov5eBXMAVdo127bxWF8445QNqmwlbXm59lHG"
base_url = "https://api.hunyuan.cloud.tencent.com/v1"
bot = ChatBot(api_key, base_url)
bot.set_background("你是一个知识渊博的助手，能够简洁地回答问题。")
bot.set_prompt("请保持回答简短。")

# 初始化Wav2Lip模型
check_path = r"D:\coding\projects\Python\human\Wav2Lip\checkpoints\wav2lip_gan.pth"
img_path = r"D:\coding\projects\Python\human\Wav2Lip\input\1.png"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model(check_path)
face_img, face_coords, orig_image = preprocess_image(img_path, device=device)


def run_chatbot_and_tts(user_input):
    # 获取LLM的回答
    response = bot.chat(user_input)

    # 将回答转换为语音
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    audio_file = loop.run_until_complete(generate_speech(response))
    loop.close()

    # 启动Wav2Lip同步播放
    def run_wav2lip():
        gen = prepare_audio_batches(audio_file, face_img, face_coords)
        wav2lip_sync_play(
            model, gen, device,
            orig_image=orig_image, coords=face_coords,
            audio_path=audio_file
        )

    wav2lip_thread = threading.Thread(target=run_wav2lip)
    wav2lip_thread.start()

    return response


# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("### 对话数字人系统")
    input_text = gr.Textbox(label="输入问题")
    output_text = gr.Textbox(label="回答")
    submit_button = gr.Button("提交")

    submit_button.click(
        fn=run_chatbot_and_tts,
        inputs=input_text,
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()