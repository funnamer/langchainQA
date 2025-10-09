from langchain_qwen3 import Qwen3LLM

qwen_llm = Qwen3LLM(
    api_url="http://localhost:8000/qwen3/local/generate",  # 你的 API 地址
    max_new_tokens=512,
    temperature=0.1
)

# 直接生成文本
response = qwen_llm("用 3 句话介绍 LangChain 的作用")
print("生成结果：")
print(response)