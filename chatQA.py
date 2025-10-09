from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory  # 对话记忆（保存多轮历史）

from langchain_qwen3 import Qwen3LLM
from qwen3_embeddings import Qwen3EmbeddingAPI


def init_qwen3_rag_chroma(
        chroma_persist_dir: str = "./chroma_pumpkin_db",  # 已存在的 Chroma 向量库路径
        chroma_collection_name: str = "qwen3_embeddings",  # Chroma 集合名（需与你创建向量库时一致）
        api_url_llm: str = "http://localhost:8000/qwen3/local/generate",  # Qwen3 对话API地址
        api_url_emb: str = "http://localhost:8001/embed"  # Qwen3 嵌入API地址（用于检索时的查询向量化）
) -> ConversationalRetrievalChain:
    """
    初始化带记忆的 Qwen3 RAG 链（加载已有的 Chroma 向量库，无需重新向量化）
    :param chroma_persist_dir: 本地 Chroma 向量库的持久化路径
    :param chroma_collection_name: Chroma 中存储向量的集合名（需与创建时一致）
    :param api_url_llm: Qwen3 对话模型 API 地址
    :param api_url_emb: Qwen3 嵌入模型 API 地址（用于查询文本向量化）
    :return: 带对话记忆的 RAG 链
    """
    # 初始化 Qwen3 对话模型
    qwen3_llm = Qwen3LLM(
        api_url=api_url_llm,
        enable_thinking=False,  # 按需开启（思考模式会增加耗时，纯RAG推荐关闭）
        max_new_tokens=1500,  # 适配长回答（需覆盖“问题+历史+检索文档+答案”的长度）
        temperature=0.3
    )

    # 初始化 Qwen3 嵌入模型（用于将用户新问题向量化，与 Chroma 中的文档向量匹配）
    qwen3_embeddings = Qwen3EmbeddingAPI(
        api_url=api_url_emb
    )

    # 加载已持久化的 Chroma 库
    try:
        chroma_vector_store = Chroma(
            collection_name=chroma_collection_name,  # 必须与创建向量库时的集合名一致
            persist_directory=chroma_persist_dir,  # 向量库本地存储路径
            embedding_function=qwen3_embeddings  # 检索时用相同的嵌入模型处理查询
        )
        print(f"✅ 成功加载 Chroma 向量库！路径：{chroma_persist_dir}，集合名：{chroma_collection_name}")
        # 可选：验证向量库规模（查看文档片段总数）
        doc_count = chroma_vector_store._collection.count()
        print(f"✅ 向量库中包含 {doc_count} 个文档片段")
    except Exception as e:
        raise RuntimeError(
            f"加载 Chroma 向量库失败！请检查路径/集合名是否正确，或向量库是否已创建：{str(e)}"
        )

    # ConversationBufferMemory：保存完整对话历史（短轮次优先，长轮次可换 ConversationSummaryMemory）
    conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",  # 记忆键名（必须与 ConversationalRetrievalChain 对齐）
        return_messages=True,  # 返回 Message 对象（链需此格式处理历史）
        output_key="answer"  # 指定记忆中保存“回答”字段（避免与检索结果冲突）
    )

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=qwen3_llm,  # 对话模型（负责整合信息生成自然语言回答）
        retriever=chroma_vector_store.as_retriever(
            search_kwargs={"k": 3}  # 检索 top-3 最相关的文档片段（平衡相关性与效率）
            # 可选：检索策略优化（如 MMR 避免结果重复）
            # search_type="mmr",
            # search_kwargs={"k": 3, "fetch_k": 10}  # fetch_k：先获取10个候选，再选3个多样性高的
        ),
        memory=conversation_memory,  # 对话记忆（让模型记住上下文，如“上一个问题的答案是什么”）
        return_source_documents=True,  # 返回回答参考的源文档（便于溯源“答案来自哪段文档”）
        verbose=False,  # 调试时设为 True，可查看链的运行日志（如检索到的文档）
        chain_type="stuff"  # 文档整合方式（stuff：直接将检索片段塞入 prompt，高效）
    )

    return rag_chain


def run_rag_conversation(rag_chain: ConversationalRetrievalChain):
    """
    运行带记忆的 RAG 对话交互（每次对话仅查询 Chroma 向量库，不重新向量化）
    :param rag_chain: 初始化后的 RAG 链
    """
    print("📌 带记忆的 Qwen3 RAG 对话（Chroma 版）已启动！")
    print("💡 操作提示：输入问题获取答案，输入 'exit' 退出，输入 'clear' 清空对话历史\n")

    while True:
        user_query = input("你：")

        # 退出对话
        if user_query.lower() == "exit":
            print("系统：对话结束，再见！")
            break

        # 清空对话历史
        if user_query.lower() == "clear":
            rag_chain.memory.clear()  # 调用记忆组件的 clear 方法清空历史
            print("系统：已清空对话历史，可重新提问～\n")
            continue

        # 空输入过滤
        if not user_query.strip():
            print("系统：请输入有效的问题哦～\n")
            continue

        try:
            # 调用 RAG 链：自动完成“查询向量化→Chrom a检索→结合历史→生成答案”
            response = rag_chain({
                "question": user_query  # 仅传入新问题，历史由 memory 自动携带
            })

            # 解析结果（答案 + 参考的源文档）
            answer = response["answer"].strip()
            source_docs = response["source_documents"]  # 检索到的相关文档片段

            # 输出答案
            print(f"\n系统：{answer}\n")

            # 输出参考文档（可选，让用户知道答案来源，增强可信度）
            print("🔍 参考文档片段（前3条）：")
            for i, doc in enumerate(source_docs, 1):
                # 提取文档元数据（如来源路径、页码，需创建向量库时已存入）
                doc_source = doc.metadata.get("source", "未知来源")
                doc_page = doc.metadata.get("page", "未知页码")
                # 截取文档内容（避免过长，显示前200字符）
                doc_content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                print(f"  片段{i} | 来源：{doc_source} | 页码：{doc_page}")
                print(f"  内容：{doc_content}\n")

        except Exception as e:
            # 捕获异常（如 API 连接失败、检索错误等）
            print(f"系统：生成答案失败，错误信息：{str(e)}\n")


if __name__ == "__main__":
    # 初始化 RAG 链（需替换为你的 Chroma 向量库路径和集合名）
    rag_chain = init_qwen3_rag_chroma(
        chroma_persist_dir="./chroma_pumpkin_db",  # 你的 Chroma 向量库路径
        chroma_collection_name="qwen3_embeddings"  # 你的 Chroma 集合名（创建时的名称）
    )

    # 启动对话交互
    run_rag_conversation(rag_chain)