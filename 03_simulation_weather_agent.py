import os
import google.generativeai as genai

# 1. 保持网络畅通（请确保端口号与你之前配置的一致）
os.environ['http_proxy'] = 'http://127.0.0.1:10808'
os.environ['https_proxy'] = 'http://127.0.0.1:10808'

# 2. 填入你的 API Key
genai.configure(api_key="AIzaSyBW5KEqzlTATYWfvxAOhVicAKziWrCrWZ8") 

# ==========================================
# 打造新工具：模拟实时天气查询接口
# ==========================================
def get_current_weather(city: str) -> str:
    """
    查询指定城市的实时天气。
    当用户询问任何城市的天气、气温或出行建议时，请务必先调用此工具获取最新数据。
    """
    print(f"\n[系统内部日志] 🌍 正在联网搜索 {city} 的实时天气...")
    
    # 模拟一个外部数据库或 API 返回的结果
    weather_database = {
        "北京": "晴天，气温 28°C，紫外线极强",
        "东京": "暴雨，气温 18°C，航班大面积延误",
        "伦敦": "阴天，气温 15°C，伴有微风"
    }
    # 如果查不到，就告诉大模型找不到
    return weather_database.get(city, "抱歉，未找到该城市的实时天气数据。")

# ==========================================
# 组装 Agent：把天气工具交给它
# ==========================================
model = genai.GenerativeModel(
    model_name='gemini-flash-latest',
    tools=[get_current_weather] 
)

chat = model.start_chat(enable_automatic_function_calling=True)

# 发送一个复杂的复合任务
user_prompt = "我下周打算去旅游。请问今天北京和东京的天气分别怎么样？根据天气，我更适合去哪里？需要带什么装备？"
print(f"用户提问: {user_prompt}")
print("-" * 30)

response = chat.send_message(user_prompt)

print("\n--- Agent 最终回复 ---")
print(response.text)