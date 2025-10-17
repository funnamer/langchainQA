from langchain.llms.base import LLM
from pydantic import Field
from typing import Any, List, Mapping, Optional
import requests


class Qwen3LLM(LLM):
    # API 地址（需与你的 FastAPI 服务地址一致）
    api_url: str = Field(default="http://localhost:8000/qwen3/local/generate")

    enable_thinking: bool = False
    max_new_tokens: int = 1024  
    temperature: float = 0.1  

    @property
    def _llm_type(self) -> str:
        return "qwen3-0.6b-local"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> str:

        payload = {
            "messages": [{"role": "user", "content": prompt}],  # 单轮对话
            "enable_thinking": self.enable_thinking,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=60  
            )
            response.raise_for_status()  
        except requests.exceptions.RequestException as e:
            raise ValueError(f"调用 Qwen3 API 失败：{str(e)}")

        # 解析 API 响应
        result = response.json()
        generated_text = result.get("response", "") 

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
