import google.generativeai as genai
import os

os.environ['http_proxy'] = 'http://127.0.0.1:10808'
os.environ['https_proxy'] = 'http://127.0.0.1:10808'

# 1. 亮出你的身份证：配置 API Key
genai.configure(api_key="AIzaSyBW5KEqzlTATYWfvxAOhVicAKziWrCrWZ8")

# 2. 唤醒大脑：选择你要使用的具体模型
model = genai.GenerativeModel('gemini-flash-latest')
# 3. 发送指令：让大模型干活
print("正在呼叫 Gemini 大脑，请稍候...")
response = model.generate_content("用一句话给我解释一下什么是API？并用一个幽默的比喻。")

# 4. 接收结果：打印出大模型的回答
print("\n--- 收到回复 ---")
print(response.text)