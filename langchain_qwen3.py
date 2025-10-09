from langchain.llms.base import LLM
from pydantic import Field
from typing import Any, List, Mapping, Optional
import requests


class Qwen3LLM(LLM):
    # API 地址（需与你的 FastAPI 服务地址一致）
    api_url: str = Field(default="http://localhost:8000/qwen3/local/generate")

    # 模型生成参数（默认值可根据需求调整）
    enable_thinking: bool = False  # 是否启用思考模式
    max_new_tokens: int = 1024  # 最大生成 token 数
    temperature: float = 0.1  # 随机性参数

    @property
    def _llm_type(self) -> str:
        return "qwen3-0.6b-local"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> str:

        # 构建 API 请求参数（符合 FastAPI 接口的格式）
        payload = {
            "messages": [{"role": "user", "content": prompt}],  # 单轮对话
            "enable_thinking": self.enable_thinking,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature
        }

        # 发送 POST 请求到本地 API
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=60  # 超时时间（秒），根据生成长度调整
            )
            response.raise_for_status()  # 检查请求是否成功（非 200 状态码会报错）
        except requests.exceptions.RequestException as e:
            raise ValueError(f"调用 Qwen3 API 失败：{str(e)}")

        # 解析 API 响应
        result = response.json()
        generated_text = result.get("response", "")  # 取模型的最终回复

        # 处理停止词（如果提供）
        if stop:
            for s in stop:
                if s in generated_text:
                    generated_text = generated_text.split(s)[0]

        return generated_text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "api_url": self.api_url,
            "enable_thinking": self.enable_thinking,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature
        }