import os
from openai import OpenAI
from datetime import datetime

class ChatBot:
    def __init__(
        self,
        api_key,
        base_url,
        model="hunyuan-turbos-latest",
        log_dir="logs_llm",
        default_background="你是一个知识渊博的助手。",
        default_prefix="请简洁地回答下述问题，并且不要带*#："
    ):
        # 创建日志目录并生成唯一日志文件名
        os.makedirs(log_dir, exist_ok=True)
        log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{log_time}.txt")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.messages = []
        self.system_prompt_set = False
        self.background = default_background
        self.prefix = default_prefix
        self._init_log()

    def _init_log(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("# LLM ChatBot 对话日志\n\n")

    def _log(self, entry):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {entry}\n")

    def set_background(self, background_text):
        self.background = background_text
        if not self.system_prompt_set:
            self.messages.append({"role": "system", "content": background_text})
            self.system_prompt_set = True
        else:
            self.messages[0]["content"] = background_text
        self._log(f"设置背景提示词：{background_text}")

    def set_prefix(self, prefix_text):
        self.prefix = prefix_text
        self._log(f"设置前缀提示词：{prefix_text}")

    def set_prompt(self, prompt_text):
        for i, msg in enumerate(self.messages):
            if msg["role"] == "user" and msg.get("prompt", False):
                self.messages[i]["content"] = prompt_text
                self._log(f"设置Prompt提示：{prompt_text}")
                return
        self.messages.append({"role": "user", "content": prompt_text, "prompt": True})
        self._log(f"新增Prompt提示：{prompt_text}")

    def chat(self, user_input, enable_enhancement=True):
        # 前缀提示词拼接
        full_input = f"{self.prefix}{user_input}" if self.prefix else user_input
        user_msg = {"role": "user", "content": full_input}
        self.messages = [msg for msg in self.messages if not msg.get("prompt", False)]
        self.messages.append(user_msg)
        self._log(f"用户: {user_input}")
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                extra_body={
                    "enable_enhancement": enable_enhancement
                }
            )
            response = completion.choices[0].message.content
            self.messages.append({"role": "assistant", "content": response})
            self._log(f"助手: {response}")
            return response
        except Exception as e:
            err_msg = f"API调用出错: {str(e)}"
            print(err_msg)
            self._log(f"助手发生错误: {err_msg}")
            return "抱歉，我遇到了一个错误，无法回答。"

    def get_history(self):
        return self.messages

    def clear_history(self):
        system_prompt = self.messages[0] if self.system_prompt_set else None
        self.messages = [system_prompt] if system_prompt else []
        self._log("对话历史已清空")