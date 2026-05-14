import os
import chromadb
# 引入 ChromaDB 自定义函数的基类
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai

# 1. 挂上代理 (回到 Windows 了，端口应该是 10808)
os.environ['http_proxy'] = 'http://127.0.0.1:10808'
os.environ['https_proxy'] = 'http://127.0.0.1:10808'

# 2. 填入你的 API Key
genai.configure(api_key="AIzaSyBW5KEqzlTATYWfvxAOhVicAKziWrCrWZ8") 

# ==========================================
# 核心魔法：自定义一个 ChromaDB 适配器
# 把我们在路线 A 学到的知识包装起来，完美避开官方 Bug
# ==========================================
class MyGeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input_texts: Documents) -> Embeddings:
        # 这个函数会接收 ChromaDB 传过来的一批文本，我们需要把它们变成向量返回去
        embeddings = []
        for text in input_texts:
            result = genai.embed_content(
                # 【注意】填入你之前查到的可用模型名，比如 "models/embedding-001"
                model="models/gemini-embedding-2", 
                content=text
            )
            embeddings.append(result['embedding'])
        return embeddings

# ==========================================
# 初始化 ChromaDB
# ==========================================
client = chromadb.PersistentClient(path="./my_local_vectordb")

# 创建集合时，使用我们【自己手搓的适配器】！
collection = client.get_or_create_collection(
    name="company_rules",
    embedding_function=MyGeminiEmbeddingFunction() 
)

print("📚 正在将文档存入 ChromaDB，并自动转换向量...")
collection.upsert(
    documents=[
        "公司规章制度：办公区域内严格禁止携带猫狗等一切宠物进入。",
        "财务报销流程：所有出差打车费用，请在当月月底统一贴票找王会计报销。",
        "食堂通知：为庆祝节日，本周五中午二楼食堂将免费提供红烧肉。"
    ],
    ids=["doc1", "doc2", "doc3"] 
)
print("✅ 存储完成！\n")

# ==========================================
# 体验检索
# ==========================================
query_text = "我明天上班能带我的小金毛去公司吗？"
print(f"🗣️ 用户提问: {query_text}")
print("🔍 ChromaDB 正在极速检索...")

results = collection.query(
    query_texts=[query_text],
    n_results=1 
)

print("\n🎯 检索成功！最匹配的文档是：")
print(f"👉 {results['documents'][0][0]}")