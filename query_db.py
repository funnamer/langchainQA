from langchain_community.vectorstores import Chroma
from qwen3_embeddings import Qwen3EmbeddingAPI

embedding = Qwen3EmbeddingAPI()

persist_path='./data/medicalQA/vectordb/chroma'


vectordb=Chroma(
    persist_directory=persist_path,
    embedding_function=embedding,
)
print(f"加载的向量库中存储的数量：{vectordb._collection.count()}")


# base_prompt=""
question = "小孩脑袋发热"
sim_docs = vectordb.similarity_search(question,k=3)
print(f"检索到的内容数：{len(sim_docs)}")


for i, doc in enumerate(sim_docs):
    print(f"\n第{i}个文档内容----------------------------\n{doc.page_content}")