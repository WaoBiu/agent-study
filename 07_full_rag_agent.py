import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai

# ==========================================
# 1. 基础配置 (代理与 API 密钥)
# ==========================================
os.environ['http_proxy'] = 'http://127.0.0.1:10808'
os.environ['https_proxy'] = 'http://127.0.0.1:10808'
genai.configure(api_key="AIzaSyBW5KEqzlTATYWfvxAOhVicAKziWrCrWZ8") 

# ==========================================
# 2. 向量数据库准备 (翻译官与本地图书馆)
# ==========================================
class MyGeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input_texts: Documents) -> Embeddings:
        embeddings = []
        for text in input_texts:
            result = genai.embed_content(
                model="models/gemini-embedding-2", # 请确保替换为你查到的可用模型名
                content=text
            )
            embeddings.append(result['embedding'])
        return embeddings

client = chromadb.PersistentClient(path="./my_local_vectordb")
collection = client.get_or_create_collection(
    name="company_rules",
    embedding_function=MyGeminiEmbeddingFunction() 
)

# 确保数据库里有数据 (如果之前已经存过，这里会自动忽略或更新)
# collection.upsert(
#     documents=[
#         "公司规章制度：办公区域内严格禁止携带猫狗等一切宠物进入。",
#         "财务报销流程：所有出差打车费用，请找王会计报销。",
#         "食堂通知：本周五中午二楼食堂将免费提供红烧肉。"
#     ],
#     ids=["doc1", "doc2", "doc3"] 
# )

# ==========================================
# 3. 核心流程：RAG 检索增强生成
# ==========================================
# 用户的真实提问
user_query = "我明天上班能带我的小金毛去公司吗？"
print(f"🧑 用户: {user_query}")

# 第一步：检索 (Retrieval) - 从数据库找出最相关的 1 条规定
print("\n[系统日志] 🔍 正在检索本地知识库...")
results = collection.query(query_texts=[user_query], n_results=1)
retrieved_context = results['documents'][0][0]
print(f"[系统日志] 📄 查找到参考资料: {retrieved_context}")

# 第二步：组装提示词模板 (Prompt Engineering)
# 这是 RAG 最精髓的一步：我们要给大模型设定严格的答题规则
final_prompt = f"""
你是一个专业的公司内部行政助手。请严格根据以下【参考资料】来回答用户的【问题】。
要求：
1. 态度要温和、专业。
2. 如果参考资料中没有明确提及相关内容，请诚实地回答“抱歉，资料中未找到相关规定”，绝对不要自己胡编乱造。

【参考资料】：
{retrieved_context}

【问题】：
{user_query}
"""

# 第三步：生成 (Generation) - 将组装好的任务发给大模型大脑
print("\n[系统日志] 🧠 正在结合参考资料，思考最终回答...\n")
model = genai.GenerativeModel('gemini-flash-latest')
response = model.generate_content(final_prompt)

print("🤖 Agent 最终回复:")
print("-" * 40)
print(response.text)
print("-" * 40)