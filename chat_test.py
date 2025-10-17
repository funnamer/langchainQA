from langchain_community.vectorstores import Chroma
from qwen3_embeddings import Qwen3EmbeddingAPI


embedding = Qwen3EmbeddingAPI(
    api_url="http://localhost:8001/embed",
    normalize=True,
    dim=1024
)

# 加载向量库
persist_path = './data/medicalQA/vectordb/chroma'
vectordb = Chroma(
    persist_directory=persist_path,
    embedding_function=embedding,
)
print(f"加载的向量库中存储的数量：{vectordb._collection.count()}")

from langchain_qwen3 import Qwen3LLM
qwen_llm = Qwen3LLM(
    api_url="http://localhost:8000/qwen3/local/generate",
    max_new_tokens=512,
    temperature=0.1
)


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

output_parser = StrOutputParser()

template = """你是一名儿科医生，使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。尽量回答五句话以上。尽量使答案简洁。”。
{context}
问题: {input}
"""
prompt = PromptTemplate(
    template=template,
    input_variables=["context", "input"]
)

# 定义检索器和上下文格式化函数
retriever = vectordb.as_retriever(search_kwargs={"k": 5})  # 取前3个相关文档

def format_docs(docs):
    """将检索到的文档列表格式化为字符串"""
    return "\n\n".join(doc.page_content for doc in docs)

# 关键：定义「提取查询字符串」的函数（解决 dict 转 str 问题）
def get_query(inputs: dict) -> str:
    """从输入字典中提取 question 字段，返回纯字符串"""
    return inputs.get("question", "").strip()  # 处理空值，避免传入空字符串

# 定义完整的 QA 链（先提查询，再检索，最后生成回答）
retrieval_chain = get_query | retriever | format_docs  # 检索链：str → 文档 → 格式化上下文
qa_chain = (
    {
        "context": retrieval_chain, 
        "input": get_query           
    }
    | prompt
    | qwen_llm
    | output_parser
)

if __name__ == "__main__":
    try:
        result = qa_chain.invoke({"question": "小孩脑袋发热，怎么回事，可能由什么引起的"})
        print("回答结果：")
        print(result)
    except Exception as e:

        print(f"调用链失败：{str(e)}")
