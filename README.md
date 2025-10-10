# medical_langchainQA
分别对已部署的Qwen3 0.6B 封装langchain_qwen3，以及对Qwen3 -0.6B-embedding 嵌入向量模型封装的qwen3_embeddings，经过封装后可直接在langchain中调用
对于Qwen3的部署可参考我另外一个项目`https://github.com/funnamer/EscalateIdea`

# 项目结构
```
langchainQA/
├── data/                  # 数据目录，存放用于 QA 系统的相关数据
│   ├── medicalQA/         # 向量数据库
│   └── pediatrics.pdf     # 儿科相关 PDF 文档，可能作为知识库来源
├── build_vectordb.py      # 构建向量数据库的脚本，用于将文档等数据转换为向量存储
├── chat_test.py           # 聊天测试脚本，用于测试 QA 系统的对话功能
├── langchain_qwen3.py     # 基于langchain的BaseLLM 对于Qwen3 api进行封装以支持langchain框架的使用
├── query_db.py            # 测试查询，查询向量数据库的脚本，用于从向量库中检索相关内容
├── qwen3_embeddings.py    # 基于langchain的Embeddings对Qwen3-embedding-向量化模型的api进行封装以支持langchain框架的使用
├── README.md              # 项目说明文档，介绍项目功能、结构、使用方法等
├── test_embedding.py      # 测试 Embeddings 功能的脚本，验证文本嵌入的效果
└── test_llm.py            # 测试大语言模型（LLM）功能的脚本，验证模型的生成等能力
```
