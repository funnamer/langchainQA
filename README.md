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
├── chat_test.py           # 聊天测试脚本，用于测试 QA 系统的对话功能(核心代码)
├── langchain_qwen3.py     # 基于langchain的BaseLLM 对于Qwen3 api进行封装以支持langchain框架的使用(核心代码)
├── query_db.py            # 测试查询，测试 build_vectordb.py 
├── qwen3_embeddings.py    # 基于langchain的Embeddings对Qwen3-embedding-向量化模型的api进行封装以支持langchain框架的使用(核心代码)
├── README.md              
├── test_embedding.py      # 测试qwen3_embeddings.py
└── test_llm.py            # 测试langchain_qwen3.py
```

# 结果
question:小孩脑袋发热，怎么回事，可能由什么引起的？
prompt:如果你不知道答案，就说你不知道，不要试图编造答案。尽量回答五句话以上。尽量使答案简洁。
使用test_llm.py询问LLM结果如下
```
小孩脑袋发热可能由多种原因引起，包括：  
1. 睡眠不足或过度疲劳；  
2. 环境温度过高；  
3. 感染性疾病（如感冒、流感）；  
4. 情绪波动或压力。  
如果你不确定，建议咨询医生或家长。
```
使用chat_test.py 通过RAG向量检索，匹配文本作为上下文，询问LLM结果如下
```
加载的向量库中存储的数量：2120

回答结果：
小孩脑袋发热可能由多种原因引起，包括感染、炎症或其他疾病。常见的原因包括：

1. **感染**：如细菌性、病毒性或化脓性感染，尤其是发热初期或体温快速上升期。
2. **炎症**：如脑膜炎、脑炎或脑积水等，可能导致发热伴随神经系统症状。
3. **热性惊厥**：在发热初期或体温上升期出现的惊厥，通常与中枢神经系统有关，但已被国际抗癫痫联盟分类为非癫痫。
4. **其他疾病**：如低血糖、低钙血症、颅内压增高等，也可能导致发热。

如果发热持续或伴随其他症状，建议及时就医进行详细检查。
```

可以看出 对于病情的分析和回答  明显详细了很多
