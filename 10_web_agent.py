import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ==========================================
# 🎨 网页基础配置
# ==========================================
st.set_page_config(page_title="我的超级智能体", page_icon="🤖")
st.title("🤖 你的全能企业助理 Agent")
st.caption("基于 LangGraph 驱动，内置企业规章数据库与天气查询工具")

# ==========================================
# ⚙️ 后端大脑初始化 (使用缓存，避免每次对话重复加载)
# ==========================================
@st.cache_resource
def init_agent():
    # ⚠️ 请确保替换正确的端口和 Key
    os.environ['http_proxy'] = 'http://127.0.0.1:10808'
    os.environ['https_proxy'] = 'http://127.0.0.1:10808'
    os.environ["GOOGLE_API_KEY"] = "AIzaSyB5ZLe3z3FTLsm557vWkIjrIGVmIDekooY"

    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
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
        docs = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    @tool
    def get_weather(location: str) -> str:
        """查询指定城市的天气。当用户询问天气情况时使用此工具。"""
        if "北京" in location:
            return "北京今天突降特大暴雨，街道积水严重，气温 15 度。"
        return "天气晴朗，适合出行。"

    tools = [search_company_rules, get_weather]
    return create_react_agent(llm, tools)

# 获取组装好的 Agent
agent = init_agent()

# ==========================================
# 💬 聊天界面交互逻辑
# ==========================================

# 1. 记忆系统：如果刚打开网页，创建一个空的历史记录本
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. 渲染历史对话：把历史记录本里的对话一条条画在网页上
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. 接收用户提问：在网页底部画一个输入框
if prompt := st.chat_input("请向我提问 (例如：明天能带小金毛去北京分部吗？)"):
    
    # 立即把用户的问题画在屏幕上，并存入历史记录
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 让 Agent 开始思考并回答
    with st.chat_message("assistant"):
        
        # 1. 开启加载动画 (这里面的代码负责纯后台计算)
        with st.spinner("🧠 智能体正在自主思考并调度工具..."):
            agent_messages = [
                ("human" if m["role"] == "user" else "ai", m["content"]) 
                for m in st.session_state.messages
            ]
            response = agent.invoke({"messages": agent_messages})
            raw_content = response["messages"][-1].content
            
            # 【终极净水器】
            if isinstance(raw_content, list):
                final_answer = raw_content[0].get("text", "")
            else:
                final_answer = raw_content
            
        # 2. 【看这里！】往左退一格（4个空格）。
        # 它离开了 spinner(加载圈)，但依然在 assistant(机器头像) 的管辖内！
        st.markdown(final_answer)
            
            
    # 3. 【看这里！】再往左退一格，和最外层对齐。
    # 彻底离开网页渲染区，只在后台静默保存记忆。
    st.session_state.messages.append({"role": "assistant", "content": final_answer})