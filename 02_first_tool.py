import os
import google.generativeai as genai

# 1. 保持网络畅通（使用你之前查到的 10808 端口）
os.environ['http_proxy'] = 'http://127.0.0.1:10808'
os.environ['https_proxy'] = 'http://127.0.0.1:10808'

# 2. 配置 API Key
genai.configure(api_key="AIzaSyBW5KEqzlTATYWfvxAOhVicAKziWrCrWZ8") 

# ==========================================
# 核心步骤 A：打造你的“工具” (一个普通的 Python 函数)
# ==========================================
# 【极其重要】函数的类型提示 (float) 和 注释 (Docstring) 绝对不能省！
# 大模型就是靠阅读这里的注释，来判断自己该不该用这个工具、以及怎么传参数的。
def multiply_numbers(a: float, b: float) -> float:
    """
    一个高精度计算器工具，专门用于计算两个数字的乘积。
    当用户要求你做乘法运算时，请务必调用此工具。
    """
    print(f"\n[系统内部日志] 🛠️ 触发工具调用！大模型把参数传过来了，正在后台偷偷帮你计算: {a} * {b} ...")
    return a * b

# ==========================================
# 核心步骤 B：带着工具箱唤醒大脑
# ==========================================
# 把我们写好的函数放在 tools 列表里，告诉大模型：“这是给你的工具包”
model = genai.GenerativeModel(
    model_name='gemini-flash-latest',
    tools=[multiply_numbers] 
)

# ==========================================
# 核心步骤 C：开始对话并观察魔法
# ==========================================
# enable_automatic_function_calling=True 这个参数非常贴心
# 它会让 SDK 自动帮我们执行大模型点名要用的函数，并把结果自动塞回给大模型
chat = model.start_chat(enable_automatic_function_calling=True)

print("正在给 Gemini 发送一个复杂的乘法难题...")
# 大模型本身做大数乘法很容易瞎编，所以它必须依赖你的工具
response = chat.send_message("请帮我计算一下 3.14159 乘以 9999 的精确结果是多少？")

print("\n--- 收到最终回复 ---")
print(response.text)