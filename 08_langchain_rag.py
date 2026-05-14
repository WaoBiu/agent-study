import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 配置网络与 API
os.environ['http_proxy'] = 'http://127.0.0.1:10808'
os.environ['https_proxy'] = 'http://127.0.0.1:10808'
os.environ["GOOGLE_API_KEY"] = "AIzaSyAYIGYa-mWetYw2yjSjmCgja7CH43h-OjM" # ⚠️ 记得换新的！

# ==========================================
# 模块 A：连接已有数据库 (Retriever)
# ==========================================
print("🔗 正在连接本地向量数据库...")
# 文字 -> 向量 模型
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")

vectorstore = Chroma(
    persist_directory="./my_local_vectordb", 
    collection_name="company_rules",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# ==========================================
# 模块 B：设定大脑与提示词 (LLM & Prompt)
# ==========================================
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的公司内部行政助手。请严格根据以下【参考资料】来回答用户的提问。\n\n【参考资料】:\n{context}"),
    ("human", "{input}"),
])

# 【辅助函数】：把数据库搜出来的文档对象，提取成纯文本字符串
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ==========================================
# 模块 C：核心魔法 —— 纯正 LCEL 流水线组装
# ==========================================
print("⚙️ 正在组装纯正的 LCEL RAG 流水线...")

# 看这里！使用 | 符号像接水管一样把所有组件连起来
rag_chain = (
    # 第一步：准备数据 (把提问透传给 input，同时用检索器去找资料给 context)
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    # 第二步：塞进提示词模板
    | prompt
    # 第三步：发给大模型大脑
    | llm
    # 第四步：把大模型复杂的返回格式，直接过滤成干净的文本
    | StrOutputParser()
)

# ==========================================
# 运行！一键触发流水线
# ==========================================
user_query = "我明天上班能带我的小金毛去公司吗？"
print(f"\n🧑 用户: {user_query}")
print("🤖 Agent 正在流水线上全自动处理...")

# 直接 invoke，因为我们加了 StrOutputParser，返回的直接就是字符串！
response = rag_chain.invoke(user_query)

print("\n--- 最终回复 ---")
print(response)