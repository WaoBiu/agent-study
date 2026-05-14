import os
import requests
import google.generativeai as genai

# 1. 保持网络畅通 (请确认端口号 10808 是否正确)
os.environ['http_proxy'] = 'http://127.0.0.1:10808'
os.environ['https_proxy'] = 'http://127.0.0.1:10808'

# 2. 填入你的 API Key
genai.configure(api_key="AIzaSyBW5KEqzlTATYWfvxAOhVicAKziWrCrWZ8") 

# ==========================================
# 核心能力一：真实的感知工具 (读取真实网页)
# ==========================================
def read_real_website(url: str) -> str:
    """
    读取真实网页内容的工具。
    当用户要求你总结、查看或提取某个网址(URL)的内容时，务必调用此工具。
    """
    print(f"\n[系统内部日志] 🌍 正在访问真实网页: {url} ...")
    try:
        # 简单伪装成浏览器，防止被某些网站拦截
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        response.encoding = 'utf-8' # 确保中文不乱码
        
        # 截取前 2000 个字符。因为真实网页源代码很长，直接全塞给大模型可能会超过字数限制
        return response.text[:2000] 
    except Exception as e:
        return f"网页访问失败: {e}"

# ==========================================
# 组装 Agent 
# ==========================================
model = genai.GenerativeModel(
    model_name='gemini-flash-latest',
    tools=[read_real_website] 
)

# ==========================================
# 核心能力二：开启“记忆”功能
# ==========================================
# 使用 start_chat 创建的 chat 对象，会自动在后台把你的每一次提问和它的每一次回答都记录下来，形成“记忆”。
print("初始化带有记忆和联网能力的 Agent...")
chat = model.start_chat(enable_automatic_function_calling=True)

# ---------------- 测试开始 ----------------

# 第一轮对话：测试真实网页读取
print("\n🧑 用户: 去看看百度首页 (https://www.baidu.com) 的源代码里，有没有提到'新闻'或者'贴吧'？")
response1 = chat.send_message("去看看百度首页 (https://www.baidu.com) 的源代码里，有没有提到'新闻'或者'贴吧'？")
print(f"🤖 Agent: {response1.text}")

print("-" * 40)

# 第二轮对话：测试记忆能力（注意，我没有再提百度的网址）
print("\n🧑 用户: 我刚才让你看了哪个网站？你觉得那个网站主要的作用是什么？")
response2 = chat.send_message("我刚才让你看了哪个网站？你觉得那个网站主要的作用是什么？")
print(f"🤖 Agent: {response2.text}")