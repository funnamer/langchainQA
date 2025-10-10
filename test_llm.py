from langchain_qwen3 import Qwen3LLM

qwen_llm = Qwen3LLM(
    api_url="http://localhost:8000/qwen3/local/generate",  # 你的 API 地址
    max_new_tokens=512,
    temperature=0.1
)

# 直接生成文本
response = qwen_llm("如果你不知道答案，就说你不知道，不要试图编造答案。尽量回答五句话以上。尽量使答案简洁。小孩脑袋发热，怎么回事，可能由什么引起的？")
print("生成结果：")

print(response)
