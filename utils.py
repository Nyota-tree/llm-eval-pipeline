"""
工具函数和类
包含 LLM 客户端等通用组件
"""

import os
import time
from typing import Optional

from dotenv import load_dotenv

# 导入配置
import config

# 加载环境变量
load_dotenv()

# 导入 API 客户端
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


class LLMClient:
    """统一的 LLM 客户端接口，支持 OpenAI、Anthropic 和 DeepSeek"""
    
    def __init__(self, provider=None, model=None, temperature=None, max_tokens=None, top_p=None):
        """
        初始化 LLM 客户端
        
        Args:
            provider: API 提供商，如果为 None 则使用 config 中的设置
            model: 模型名称，如果为 None 则使用 config 中的设置
            temperature: 温度参数，如果为 None 则使用 config 中的设置
            max_tokens: 最大 tokens，如果为 None 则使用 config 中的设置
            top_p: Top_P 参数，如果为 None 则使用 config 中的设置（仅支持 OpenAI/DeepSeek）
        """
        # 确定使用的 provider
        if provider is not None:
            self.provider = provider.lower()
        else:
            # 优先使用评估器的 provider，否则使用默认 provider
            self.provider = (getattr(config, 'EVALUATOR_API_PROVIDER', None) or config.API_PROVIDER).lower()
        
        if self.provider == "openai":
            if OpenAI is None:
                raise ImportError("openai 包未安装。请运行: pip install openai")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("未找到 OPENAI_API_KEY 环境变量。请在 .env 文件中设置。")
            self.client = OpenAI(api_key=api_key)
            self.model = model or config.OPENAI_MODEL
            self.temperature = temperature if temperature is not None else config.OPENAI_TEMPERATURE
            self.max_tokens = max_tokens or config.OPENAI_MAX_TOKENS
            self.top_p = top_p if top_p is not None else None
            
        elif self.provider == "anthropic":
            if Anthropic is None:
                raise ImportError("anthropic 包未安装。请运行: pip install anthropic")
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("未找到 ANTHROPIC_API_KEY 环境变量。请在 .env 文件中设置。")
            self.client = Anthropic(api_key=api_key)
            self.model = model or config.ANTHROPIC_MODEL
            self.temperature = temperature if temperature is not None else config.ANTHROPIC_TEMPERATURE
            self.max_tokens = max_tokens or config.ANTHROPIC_MAX_TOKENS
            self.top_p = None  # Anthropic 不支持 top_p
            
        elif self.provider == "deepseek":
            if OpenAI is None:
                raise ImportError("openai 包未安装。请运行: pip install openai")
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("未找到 DEEPSEEK_API_KEY 环境变量。请在 .env 文件中设置。")
            self.client = OpenAI(
                api_key=api_key,
                base_url=config.DEEPSEEK_BASE_URL
            )
            # 如果提供了 model，使用它；否则根据是否有评估器配置来决定
            if model is not None:
                self.model = model
            else:
                # 检查是否有评估器专用的模型配置
                self.model = getattr(config, 'DEEPSEEK_REASONER_MODEL', config.DEEPSEEK_MODEL)
            
            # 温度参数
            if temperature is not None:
                self.temperature = temperature
            else:
                self.temperature = getattr(config, 'DEEPSEEK_REASONER_TEMPERATURE', config.DEEPSEEK_TEMPERATURE)
            
            # 最大 tokens
            if max_tokens is not None:
                self.max_tokens = max_tokens
            else:
                self.max_tokens = getattr(config, 'DEEPSEEK_REASONER_MAX_TOKENS', config.DEEPSEEK_MAX_TOKENS)
            
            # Top_P 参数
            if top_p is not None:
                self.top_p = top_p
            else:
                self.top_p = getattr(config, 'DEEPSEEK_REASONER_TOP_P', None)
        else:
            raise ValueError(f"不支持的 API 提供商: {self.provider}。请在 config.py 中设置为 'openai'、'anthropic' 或 'deepseek'")
    
    def _call_api_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        """带重试的 API 调用"""
        last_error = None
        for attempt in range(config.MAX_RETRIES):
            try:
                return self._call_api(system_prompt, user_prompt)
            except Exception as e:
                last_error = e
                if attempt < config.MAX_RETRIES - 1:
                    wait_time = config.RETRY_DELAY * (2 ** attempt)
                    time.sleep(min(wait_time, 10))
                else:
                    raise Exception(f"API 调用失败（已重试 {config.MAX_RETRIES} 次）: {str(last_error)}")
        raise Exception(f"API 调用失败: {str(last_error)}")
    
    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """内部 API 调用方法"""
        if self.provider == "openai" or self.provider == "deepseek":
            # 构建 API 调用参数
            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            # 如果设置了 top_p，添加到参数中
            if hasattr(self, 'top_p') and self.top_p is not None:
                api_params["top_p"] = self.top_p
            
            response = self.client.chat.completions.create(**api_params)
            return response.choices[0].message.content.strip()
        
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.content[0].text.strip()
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """生成响应（带重试机制）"""
        return self._call_api_with_retry(system_prompt, user_prompt)
