import os
import math
import google.generativeai as genai

# 1. 保持网络畅通 (请确认端口号 10808)
os.environ['http_proxy'] = 'http://127.0.0.1:10808'
os.environ['https_proxy'] = 'http://127.0.0.1:10808'

# 2. 配置 API Key
genai.configure(api_key="AIzaSyBW5KEqzlTATYWfvxAOhVicAKziWrCrWZ8") 

# ==========================================
# 准备我们的“迷你图书馆” (知识库)
# ==========================================
documents = [
    "公司规章制度：办公区域内严格禁止携带猫狗等一切宠物进入。",
    "财务报销流程：所有出差打车费用，请在当月月底统一贴票找王会计报销。",
    "食堂通知：为庆祝节日，本周五中午二楼食堂将免费提供红烧肉。"
]

# ==========================================
# 核心工具 1：将文字变成数字 (Embedding)
# ==========================================
def get_embedding(text):
    """调用大模型的 Embedding 接口，把文字变成一串浮点数向量"""
    result = genai.embed_content(
        model="models/gemini-embedding-2", # 这是专门负责变向量的模型
        content=text
    )
    return result['embedding']

# ==========================================
# 核心工具 2：纯数学计算距离 (余弦相似度)
# ==========================================
def cosine_similarity(vec1, vec2):
    """计算两个向量之间的余弦相似度 (值在 -1 到 1 之间，越接近 1 越相似)"""
    # 纯手搓的点积与长度公式
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

# ==========================================
# 开始 RAG 流程演示
# ==========================================
print("📚 第一步：图书管理员正在把知识库切块并变成向量 (这需要几秒钟)...")
doc_embeddings = [get_embedding(doc) for doc in documents]

# 用户的模糊提问
query = "我明天上班能带我的小金毛去公司吗？"
print(f"\n🗣️ 用户提问: {query}")

print("\n🧠 第二步：正在将用户的提问也转化为向量...")
query_embedding = get_embedding(query)

print("\n🔍 第三步：开始数学对比！计算提问与每一条文档的【余弦相似度】...")
similarities = []
for i in range(len(documents)):
    sim_score = cosine_similarity(query_embedding, doc_embeddings[i])
    similarities.append((documents[i], sim_score))
    print(f"对比文档 {i+1} 得分: {sim_score:.4f} -> {documents[i][:10]}...")

# 将得分从高到低排序，取出最高分的一项
similarities.sort(key=lambda x: x[1], reverse=True)
best_match_doc, highest_score = similarities[0]

print("\n=============================================")
print(f"🎯 检索成功！找到最相关的知识块 (最高分: {highest_score:.4f}):")
print(f"👉 {best_match_doc}")
print("=============================================")
print("第四步：现在可以把这段知识和提问一起发给 Gemini 生成最终回答了！")