import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
# 【跨时代更新】抛弃旧版 AgentExecutor，引入新一代 LangGraph 引擎
from langgraph.prebuilt import create_react_agent

# 1. 基础配置
os.environ['http_proxy'] = 'http://127.0.0.1:10808' # 确保是你 Mac 的端口
os.environ['https_proxy'] = 'http://127.0.0.1:10808'
os.environ["GOOGLE_API_KEY"] = "AIzaSyB5ZLe3z3FTLsm557vWkIjrIGVmIDekooY"
# print(f"🕵️ 当前实际使用的密钥前十位是: {os.environ.get('GOOGLE_API_KEY', '没找到')[:10]}...")

print("🔗 正在加载组件...")
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

# ==========================================
# 🛠️ 武器库锻造区：完全保持不变！这就是手写工具的巨大优势
# ==========================================

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
vectorstore = Chroma(
    persist_directory="./my_local_vectordb", 
    collection_name="company_rules",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

@tool
def search_company_rules(query: str) -> str:
    """当用户询问关于公司规章制度、报销流程、食堂通知等【公司内部规定】时，必须使用此工具。"""
    print(f"\n[系统偷窥] 👉 Agent 正在悄悄翻阅本地知识库，搜索: {query}...")
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

@tool
def get_weather(location: str) -> str:
    """查询指定城市的天气。当用户询问天气情况时使用此工具。"""
    print(f"\n[系统偷窥] 👉 Agent 正在悄悄调用天气工具，查询: {location}...")
    if "北京" in location:
        return "北京今天突降特大暴雨，街道积水严重，气温 15 度。"
    return "天气晴朗，适合出行。"

tools = [search_company_rules, get_weather]


# ==========================================
# 🧠 大脑组装区：LangGraph 降维打击！
# ==========================================
print("⚙️ 正在使用新一代 LangGraph 引擎构建 Agent...")
# 极其不可思议：不用写 Prompt，不用写草稿本，一句代码把大模型和工具绑在一起！
agent = create_react_agent(llm, tools)


# ==========================================
# 🎬 终极测试：见证 AI 的自主决策
# ==========================================
print("\n" + "="*50)
# user_query = "我是北京分部的员工，明天我想带我的小金毛去公司上班，可以吗？"
user_query = "我是北京分部的员工，明天天气怎样？"
print(f"🧑 用户刁钻提问: {user_query}")
print("="*50)

# 启动！LangGraph 采用的是标准的消息队列格式
# 我们把用户的话打包成一条 human(人类) 消息发给它
response = agent.invoke({"messages": [("human", user_query)]})

print("\n" + "="*50)
print("🤖 Agent 最终回复:")
# LangGraph 返回的是一个消息历史列表，最后一条就是它的最终回答
print(response["messages"][-1].content)
print("="*50)