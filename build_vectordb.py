from langchain_community.document_loaders import PyMuPDFLoader
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qwen3_embeddings import Qwen3EmbeddingAPI
from langchain_community.vectorstores import Chroma

# 加载PDF文档
loader = PyMuPDFLoader("./data/pediatrics.pdf")
pdf_pages = loader.load()
print(f"载入后的变量类型为：{type(pdf_pages)}，该 PDF 一共包含 {len(pdf_pages)} 页")

# 文本预处理（应用到所有页面）
pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
for page in pdf_pages:
    # 移除特定模式的换行
    page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), page.page_content)
    # 清理特殊字符
    page.page_content = page.page_content.replace('•', '').replace(' ', '').replace('• ', '')

# 文本分割
CHUNK_SIZE = 500
OVERLAP_SIZE = 150
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP_SIZE
)
split_docs = text_splitter.split_documents(pdf_pages)
print(f"切分后的文件数量：{len(split_docs)}")

# 初始化嵌入模型和向量库
embedding = Qwen3EmbeddingAPI()
persis_path = './data/medicalQA/vectordb/chroma'
vectordb = None  # 初始化向量库变量

# 分批处理文档（每批100个）
batch_size = 100
total_docs = len(split_docs)

for i in range(0, total_docs, batch_size):
    # 计算当前批次的起止索引
    start = i
    end = min(i + batch_size, total_docs)
    batch_docs = split_docs[start:end]

    print(f"开始处理第 {start + 1} 至 {end} 个文档（共 {total_docs} 个）")

    # 首次创建向量库，后续追加文档
    if vectordb is None:
        vectordb = Chroma.from_documents(
            documents=batch_docs,
            embedding=embedding,
            persist_directory=persis_path
        )
    else:
        vectordb.add_documents(batch_docs)

    print(f"已完成第 {start + 1} 至 {end} 个文档的处理，当前向量库总数：{vectordb._collection.count()}\n")

# 持久化向量库
vectordb.persist()
print(f"所有文档处理完成，最终向量库中存储的数量：{vectordb._collection.count()}")