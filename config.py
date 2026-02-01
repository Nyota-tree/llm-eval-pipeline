"""
配置文件：管理提示模板、模型设置和其他配置参数

用户可以通过修改此文件来自定义：
- API 提供商和模型选择
- 提示词模板（系统提示词、用户提示词、评估提示词）
- 批处理参数（并发数、重试次数等）
- CSV 列名配置
"""

# ==================== API 配置 ====================
# 选择使用的 API 提供商: "openai"、"anthropic" 或 "deepseek"
API_PROVIDER = "deepseek"

# OpenAI 配置
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.7
OPENAI_MAX_TOKENS = 2000

# Anthropic 配置
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
ANTHROPIC_TEMPERATURE = 0.7
ANTHROPIC_MAX_TOKENS = 2000

# DeepSeek 配置（使用 OpenAI 兼容的 API）
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_TEMPERATURE = 0.7
DEEPSEEK_MAX_TOKENS = 4000  # 增加以支持更长的输出（包含思考过程和完整评论）
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# DeepSeek 推理模型配置（用于评估器）
DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"  # DeepSeek 推理模型
DEEPSEEK_REASONER_TEMPERATURE = 0.0  # 评估需要温度为 0 以保证最大一致性
DEEPSEEK_REASONER_TOP_P = 1.0  # Top_P 设为 1.0
DEEPSEEK_REASONER_MAX_TOKENS = 2000

# ==================== 批处理配置 ====================
# 并发线程数（建议根据 API 速率限制调整，通常 3-10）
MAX_WORKERS = 5

# API 重试配置
MAX_RETRIES = 3  # 失败时的最大重试次数
RETRY_DELAY = 1  # 重试延迟（秒），使用指数退避

# ==================== Script 1: batch_generator.py 配置 ====================
# 输入 CSV 列名（包含原始数据的列）
INPUT_COLUMN = "input_text"

# 输出 CSV 列名（模型响应将保存在此列）
OUTPUT_COLUMN = "model_response"

# 系统提示词（用于 batch_generator.py）
# 用户可以完全自定义此提示词，支持 {input_text} 占位符
SYSTEM_PROMPT = """You are a helpful AI assistant.

# Task
Your task is to process the given input and generate a response according to the requirements.

# Instructions
1. Analyze the input carefully
2. Generate a response that meets the requirements
3. Follow the output format specified below

# Output Format
Please output your response in the following XML structure:

<thinking>
...your thinking process...
</thinking>

<final_result>
    <priority>P{x}</priority>
    <content>
    ...your generated content...
    </content>
</final_result>

Note: You can customize this prompt template in config.py to match your specific use case."""

# 用户提示词模板（可以使用 {input_text} 作为占位符）
# 示例: "Translate this to English: {input_text}"
USER_PROMPT_TEMPLATE = "{input_text}"

# ==================== Script 2: batch_evaluator.py 配置 ====================
# 输入 CSV 列名
MODEL_RESPONSE_COLUMN = "model_response"  # 完整输出（包含思考过程）
FINAL_CONTENT_COLUMN = "final_content"  # 提取的最终内容
INPUT_TEXT_COLUMN = "input_text"  # 原始输入

# 评估器使用的 API 提供商（可以与生成器不同）
EVALUATOR_API_PROVIDER = "deepseek"  # 使用 DeepSeek 推理模型
EVALUATOR_MODEL = "deepseek-reasoner"  # DeepSeek 推理模型

# 评估提示词模板
# 用户可以完全自定义此提示词，支持 {original_text} 和 {model_output} 占位符
# 重要：评估提示词必须要求返回 JSON 格式
EVALUATION_PROMPT = """You are an expert evaluator. Your task is to evaluate the quality of AI-generated content.

# Task
You will receive two pieces of text:
1. **Original Input**: The original input text.
2. **Model Output**: The AI-generated output content.

Please evaluate the model output based on your criteria and return a JSON object.

# Inputs
<original_input>
{original_text}
</original_input>

<model_output>
{model_output}
</model_output>

# Evaluation Criteria
Please evaluate from the following dimensions (0-10 points each):

1. **Factuality** (权重 30%): Accuracy of facts, no hallucinations
2. **Completeness** (权重 20%): Coverage of key information
3. **Adherence** (权重 25%): Following instructions and format requirements
4. **Quality** (权重 25%): Overall quality and readability

# Output Format
Return ONLY a JSON object (no Markdown code blocks):

{{
  "determined_priority": "P0/P1/P2/P3",
  "scores": {{
    "factuality_score": <0-10>,
    "completeness_score": <0-10>,
    "adherence_score": <0-10>,
    "attractiveness_score": <0-10>
  }},
  "weighted_total_score": <calculated_total_0_to_100>,
  "reasoning": "Brief explanation of your evaluation",
  "pass": <true/false>
}}

Note: 
- Each dimension score is 0-10 points
- weighted_total_score = (factuality_score * 3) + (completeness_score * 2) + (adherence_score * 2.5) + (attractiveness_score * 2.5)
- weighted_total_score ranges from 0 to 100 (since weights sum to 10)
- If factuality_score < 5, pass must be false
- Decision threshold: weighted_total_score >= 75 for PUBLISH

You can customize this evaluation prompt in config.py to match your specific evaluation criteria."""

# 评估输出列名前缀（JSON 键将被展平为这些列）
# 例如: {"clarity": 5, "accuracy": 4} -> score_clarity, score_accuracy
EVAL_COLUMN_PREFIX = "score_"
REASONING_COLUMN_NAME = "eval_reasoning"
