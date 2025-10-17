from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
import requests
import os
from typing import List, Optional


class Qwen3EmbeddingAPI(Embeddings):

    def __init__(
            self,
            api_url: str = "http://localhost:8001/embed",
            dim: int = 512,
            normalize: bool = True
    ):
        self.api_url = api_url
        self.dim = dim
        self.normalize = normalize

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        try:
            response = requests.post(
                self.api_url,
                json={"texts": texts, "dim": self.dim, "normalize": self.normalize},
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

            if result.get("code") != 200:
                raise ValueError(f"API 错误: {result.get('message')}")

            return result["data"]["embeddings"]

        except Exception as e:
            raise RuntimeError(f"嵌入 API 调用失败: {str(e)}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = 50  # 控制批量大小，避免显存溢出
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            all_embeddings.extend(self._call_api(batch_texts))
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._call_api([text])[0]


def load_and_split_pdf(pdf_path: str) -> List:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    # 加载 PDF（保留页码元数据）
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    print(f"已加载 PDF，共 {len(documents)} 页")

    # 分割文档（中文优化）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    splits = text_splitter.split_documents(documents)

    # 添加元数据（来源和页码）
    for split in splits:
        split.metadata["source"] = pdf_path
        split.metadata["page"] = split.metadata.get("page_number", "未知")

    print(f"文档分割完成，共 {len(splits)} 个片段")
    return splits


def build_vector_db(pdf_path: str, db_path: str = "./chroma_db") -> Chroma:

    # 加载并分割 PDF
    splits = load_and_split_pdf(pdf_path)

    # 初始化嵌入模型
    embeddings = Qwen3EmbeddingAPI(api_url="http://localhost:8001/embed", dim=512)

    # 构建或加载向量库
    if os.path.exists(db_path):
        print(f"加载已存在的向量库: {db_path}")
        vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
    else:
        print(f"创建新向量库: {db_path}")
        vector_db = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=db_path
        )
        vector_db.persist()  # 持久化到本地

    return vector_db



def retrieve_from_db(vector_db: Chroma, query: str, top_k: int = 3) -> List:
    # 检索相似片段（返回 top_k 个）
    similar_docs = vector_db.similarity_search(
        query=query,
        k=top_k
    )

    # 格式化输出结果（含页码和内容预览）
    results = []
    for doc in similar_docs:
        results.append({
            "page": doc.metadata["page"],
            "content": doc.page_content,
            "source": doc.metadata["source"]
        })

    return results


if __name__ == "__main__":
    # 配置路径
    PDF_PATH = "./data/pumpkin_book.pdf"  
    DB_PATH = "./chroma_pumpkin_db" 

    # 构建向量数据库
    vector_db = build_vector_db(PDF_PATH, DB_PATH)

    # 检索测试
    while True:
        query = input("\n请输入检索关键词（输入 'exit' 退出）: ")
        if query.lower() == "exit":
            break

        # 执行检索
        similar_docs = retrieve_from_db(vector_db, query, top_k=3)

        # 输出结果
        print(f"\n===== 与 '{query}' 最相关的 {len(similar_docs)} 个片段 =====")
        for i, doc in enumerate(similar_docs, 1):
            print(f"\n【片段 {i}】页码: {doc['page']}")

            print(f"内容: {doc['content'][:300]}...")  # 预览前 300 字符
