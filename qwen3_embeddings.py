import requests
from langchain_core.embeddings import Embeddings
from typing import List, Optional


class Qwen3EmbeddingAPI(Embeddings):
    def __init__(
        self,
        api_url: str = "http://localhost:8001/embed",  # 嵌入服务地址（固定）
        normalize: bool = True,  # 按服务端默认，开启向量归一化
        dim: int = 1024  # 按服务端默认，嵌入维度1024（32-1024可调整）
    ):
        self.api_url = api_url
        self.normalize = normalize
        self.dim = dim
        # 验证服务地址是否合法
        if not self.api_url.startswith(("http://", "https://")):
            raise ValueError("api_url 必须以 http:// 或 https:// 开头")

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        try:
            # 构造请求体（参数名必须和服务端的 EmbeddingRequest 一致：texts、normalize、dim）
            payload = {
                "texts": texts,          # 服务端要求的「文本列表」参数名
                "normalize": self.normalize,  # 服务端可选参数
                "dim": self.dim          # 服务端可选参数（32-1024）
            }
            # 发送 POST 请求（Content-Type 必须是 application/json）
            response = requests.post(
                url=self.api_url,
                json=payload,  # 自动设置 Content-Type: application/json
                timeout=30  # 超时时间（避免服务卡死后一直等待）
            )
            # 处理响应（服务端返回格式是 {"code":200, "data":{"embeddings":...}}）
            response.raise_for_status()  # 抛出 HTTP 错误（4xx/5xx）
            result = response.json()

            # 校验服务端返回是否包含嵌入向量
            if result.get("code") != 200:
                raise RuntimeError(f"服务端返回错误：{result.get('message', '未知错误')}")
            if "embeddings" not in result.get("data", {}):
                raise RuntimeError("服务端返回结果中缺少 'embeddings' 字段")

            return result["data"]["embeddings"]  # 返回嵌入向量列表

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"嵌入服务请求失败：{str(e)}")
        except Exception as e:
            raise RuntimeError(f"嵌入结果解析失败：{str(e)}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._call_api(texts)

    def embed_query(self, text: str) -> List[float]:
        if not text.strip():
            return []
        # 服务端要求传列表，所以把单个文本包装成列表
        return self._call_api([text])[0]