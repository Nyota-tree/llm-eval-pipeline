# LLM Evals Pipeline for PMs

> **拒绝"玄学"调优，让 Prompt 迭代可量化。**
> 一个专为 AI 产品经理和 Prompt 工程师设计的轻量级评估框架。支持从批量生成到多维度评分的全闭环流程。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Author](https://img.shields.io/badge/Made%20by-nyota%E4%BD%B3%E6%A0%91-purple)

## 💡 为什么需要这个工具？

在实际业务落地中，我们经常面临这样的痛点：
* **难以量化**：改了 Prompt，感觉变好了，但不知道具体好在哪，好多少。
* **流程繁琐**：不想用 LangSmith 那么重的工具，但 Excel 手搓又太慢。
* **维度单一**：需要同时评估"事实准确性"、"指令遵循度"和"用户吸引力"，普通脚本难以兼顾。
* **技术门槛高**：作为产品经理，不想学 Python，但需要快速验证 Prompt 效果。

**LLM Evals Pipeline** 帮你解决这些问题。它支持自定义**北极星指标 (North Star Metric)**，自动生成评估标准，并输出结构化的分析报告。

**更重要的是**：你不需要写代码！直接用 Cursor 等 AI 编程助手，用自然语言告诉它你的需求，就能完成配置和运行。

## ⚙️ 工作流架构 (Workflow)

```mermaid
graph TD
    Start[开始: input.csv] -->|1. 批量生成| Gen[Generator: 批量生成回复]
    Gen -->|并发请求| API[LLM API (DeepSeek/OpenAI/Claude)]
    API -->|返回结果| IntCSV[中间结果.csv]
    
    IntCSV -->|2. 自动评估| Eval[Evaluator: 裁判打分]
    Eval -->|加载 Prompt| Prompt[自动生成评估标准<br>基于北极星指标]
    
    Eval -->|3. 结构化提取| Final[最终产出.csv]
    Final -->|包含| Scores[多维度评分]
    Final -->|包含| Decision[P0-P4 优先级]
    Final -->|包含| Reason[AI 评价理由]
```

## ✨ 核心功能

### 🎯 北极星指标驱动
输入你关注的业务指标（如"幽默感"、"极其严格的合规性"），自动生成适配的评估 Prompt。

### 📊 结构化评分
自动产出 JSON 格式评分，包含 `decision` (决策优先级)、`scores` (细分维度打分) 和 `reasoning` (评分理由)。

### 🚀 高并发处理
内置线程池，支持 tqdm 进度条，从容处理千级数据。

### 🛡️ 生产级容错
自动重试机制、错误标记、优雅的错误处理（失败行不会中断整个流程）。

### 🔌 多模型支持
无缝切换 DeepSeek、OpenAI (GPT-4o)、Anthropic (Claude 3.5)。

### 🤖 AI 助手友好（产品经理零代码）
**专为产品经理设计**：无需写代码，直接用 Cursor 等 AI 编程助手帮你配置和运行。用自然语言描述需求，AI 自动生成命令并执行。

## 🚀 快速开始

### 💡 产品经理友好：使用 Cursor 零代码配置

**本项目的核心优势**：你不需要写任何代码！直接使用 Cursor（或任何 AI 编程助手）帮你配置即可。

#### 📋 准备工作清单（5 分钟搞定）

在开始之前，你需要准备以下材料：

1. **API Key（必需）**
   - **DeepSeek**（推荐，性价比高）：https://platform.deepseek.com/
   - **OpenAI**：https://platform.openai.com/api-keys
   - **Anthropic**：https://console.anthropic.com/
   - 任选一个即可，建议用 DeepSeek（便宜且效果好）

2. **输入数据 CSV 文件**
   - 准备一个 Excel 或 CSV 文件
   - 第一列必须是 `input_text`（列名必须完全一致）
   - 每行一条需要评估的内容
   - 示例格式：
     ```csv
     input_text
     "请介绍一下人工智能的基本概念"
     "解释什么是机器学习"
     "描述深度学习与传统机器学习的区别"
     ```
   - 或者直接使用项目自带的示例文件：`examples/input_example.csv`

3. **安装 Python 环境**（如果还没有）
   - 下载 Python 3.10+：https://www.python.org/downloads/
   - 安装时勾选 "Add Python to PATH"

#### 方式 1: 让 Cursor 帮你配置（推荐给产品经理）

**Step 1: 环境初始化（只需做一次）**

在 Cursor 中打开项目，告诉 Cursor：
```
帮我安装这个项目的依赖，运行 pip install -r requirements.txt
```

**Step 2: 配置 API Key**

1. 复制 `.env.example` 文件，重命名为 `.env`
2. 打开 `.env` 文件，告诉 Cursor：
   ```
   帮我在 .env 文件中填入我的 DeepSeek API Key，key 是 sk-xxxxxxxxxxxxx
   ```
   （替换为你的实际 API Key）

**Step 3: 配置项目参数**

打开 `config.py`，告诉 Cursor 你的需求，例如：
```
帮我配置这个项目：
- 使用 DeepSeek API
- 生成器温度设为 0.7
- 评估器使用 deepseek-reasoner，温度设为 0
- 我的应用场景是：小红书文案生成
- 北极星指标是：幽默感和传播力
```

**Step 4: 生成评估标准（可选但推荐）**

告诉 Cursor：
```
帮我运行 generate_evaluator_prompt.py，场景是"小红书文案生成"，指标是"幽默感和传播力"
```

**Step 5: 准备输入数据**

- 如果使用示例文件：直接告诉 Cursor 使用 `examples/input_example.csv`
- 如果有自己的数据：告诉 Cursor：
  ```
  帮我检查一下我的 input.csv 文件格式是否正确，第一列是不是 input_text
  ```

**Step 6: 运行批量生成**

告诉 Cursor：
```
帮我运行批量生成，输入文件是 examples/input_example.csv，输出到 output.csv
```

**Step 7: 运行批量评估**

告诉 Cursor：
```
帮我运行批量评估，输入文件是 output.csv，输出到 evaluated.csv
```

**优势**：
- ✅ 无需学习 Python 语法
- ✅ 无需理解命令行参数
- ✅ 直接用自然语言描述需求
- ✅ Cursor 自动帮你生成命令并执行
- ✅ 遇到错误直接问 Cursor，它会帮你解决

#### 方式 2: 传统命令行方式

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/Nyota-tree/llm-eval-pipeline.git
cd llm-eval-pipeline

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件填入你的 API Key
```

### 2. 配置参数

在 `config.py` 中调整模型和参数（已默认配置好 DeepSeek，性价比首选）。

### 3. 运行流程

**Step 1: 准备数据**  
你可以使用项目中的示例文件 `examples/input_example.csv`，或创建自己的 `input.csv` 文件。确保 CSV 文件包含 `input_text` 列。

**Step 2: 批量生成回复**

```bash
# 使用示例文件
python batch_generator.py examples/input_example.csv batch_output.csv

# 或使用你自己的文件
python batch_generator.py input.csv batch_output.csv
```
产出：`batch_output.csv`

**Step 3: 批量评估**

```bash
python batch_evaluator.py batch_output.csv batch_evaluation_results.csv
```
产出：`batch_evaluation_results.csv`

## 📖 最佳实践 (Scenarios)

作为 AI PM，你可以这样使用本工具：

### 场景 A：小红书文案风格调优

**Input**: 50 个原始标题。

**北极星指标**: "吸引力"权重 40%，"口语化"权重 30%。

**执行**: 运行 pipeline，筛选出 `attractiveness_score > 8` 的 Prompt 版本。

### 场景 B：客服话术合规性测试

**Input**: 100 个刁钻的用户投诉 Case。

**北极星指标**: "极度共情"且"绝不产生幻觉"。

**执行**: 重点查看 `factuality_score < 6` 的低分 Case，进行人工复盘。

## 🛠️ 目录结构

```
/
├── batch_generator.py   # 生成器核心
├── batch_evaluator.py   # 评估器核心
├── generate_evaluator_prompt.py  # Prompt 生成器
├── utils.py             # 通用 LLM 客户端
├── config.py            # 配置文件 (Prompt/Model)
├── requirements.txt     # Python 依赖
├── LICENSE              # MIT 许可证
├── .env.example         # 环境变量模板
├── .gitignore          # Git 忽略文件
├── README.md           # 项目说明
└── examples/           # 示例文件目录
    ├── input_example.csv    # 示例输入文件
    └── README.md            # 示例说明
```

## 🔧 详细使用指南

### Script 1: 批量生成器 (batch_generator.py)

从原始 CSV 文件生成模型响应。

**用法:**
```bash
python batch_generator.py <输入CSV文件> <输出CSV文件>
```

**输入 CSV 要求:**
- 必须包含 `config.INPUT_COLUMN` 指定的列（默认为 `input_text`）
- 每行包含一条需要处理的原始数据

**输出 CSV:**
- 包含原始数据的所有列
- 新增 `model_response` 列：LLM 的完整响应（包含思考过程）
- 新增 `eval_priority` 列：提取的优先级（如果输出包含 `<priority>` 标签）
- 新增 `final_content` 列：提取的最终内容（如果输出包含 `<content>` 标签）

### Script 2: 批量评估器 (batch_evaluator.py)

对模型响应进行评估，并将 JSON 结果展平为单独的列。

**用法:**
```bash
python batch_evaluator.py <输入CSV文件> <输出CSV文件>
```

**输入 CSV 要求:**
- 必须包含 `config.INPUT_TEXT_COLUMN` 指定的列（默认为 `input_text`）
- 必须包含 `config.FINAL_CONTENT_COLUMN` 指定的列（默认为 `final_content`）

**输出 CSV:**
- 包含原始数据的所有列
- 新增评估列：
  - `eval_priority`: 评估器判定的优先级
  - `factuality_score`: 事实性评分 (0-10)
  - `completeness_score`: 全面性评分 (0-10)
  - `adherence_score`: 指令遵循度评分 (0-10)
  - `attractiveness_score`: 吸引力评分 (0-10)
  - `weighted_total_score`: 加权总分 (0-100)
  - `decision`: 决策 (PUBLISH/REJECT/REVIEW/ERROR)
  - `reason`: 决策原因
  - `reasoning`: 评估理由
  - `pass`: 是否通过

### Script 3: 评估员 Prompt 生成器 (generate_evaluator_prompt.py)

根据应用场景和北极星指标，自动生成专业的评估员 Prompt。

**用法:**
```bash
python generate_evaluator_prompt.py <应用场景> <北极星指标> [输出文件]
```

**参数说明:**
- `<应用场景>`: 被评估内容的场景描述（例如：小红书文案、Python代码、客服回复）
- `<北极星指标>`: 用户最看重的核心价值（例如：幽默感、代码安全性、极度共情）
- `[输出文件]`: 可选，生成的 Prompt 保存路径（默认: `generated_evaluator_prompt.txt`）

**示例:**
```bash
# 生成小红书文案评估 Prompt
python generate_evaluator_prompt.py "小红书文案生成" "幽默感和传播力"

# 生成代码审查评估 Prompt
python generate_evaluator_prompt.py "Python代码审查" "代码安全性和可维护性" code_review_prompt.txt
```

**功能特点:**
- 自动生成符合规范的评估员 Prompt
- 包含完整的评分标准（Factuality/Safety、North Star Metric、Completeness）
- 自动计算权重分配（总和为 100%）
- 生成标准的 JSON 输出格式
- 可选择自动更新到 `config.py`

## ⚙️ 配置说明

所有配置都在 `config.py` 文件中，包括：

### API 配置

```python
# 选择 API 提供商
API_PROVIDER = "deepseek"  # 或 "openai"、"anthropic"

# 模型配置
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_TEMPERATURE = 0.7
DEEPSEEK_MAX_TOKENS = 4000
```

### 批处理配置

```python
MAX_WORKERS = 5        # 并发线程数，建议 3-10，根据 API 限制调整
MAX_RETRIES = 3        # API 调用失败时的重试次数
RETRY_DELAY = 1        # 重试延迟（秒）
```

### 提示词配置

#### 生成器提示词

```python
SYSTEM_PROMPT = """You are a helpful AI assistant.
...your custom instructions...
"""

USER_PROMPT_TEMPLATE = "{input_text}"  # 支持 {input_text} 占位符
```

#### 评估器提示词

```python
EVALUATION_PROMPT = """Evaluate the following content...
Original Input: {original_text}
Model Output: {model_output}
...your evaluation criteria...
"""
```

**重要**：评估提示词必须要求返回 JSON 格式，且 JSON 结构要明确！

### 列名配置

```python
INPUT_COLUMN = "input_text"              # 输入 CSV 的列名
OUTPUT_COLUMN = "model_response"         # 输出列名（生成器）
FINAL_CONTENT_COLUMN = "final_content"   # 最终内容列名
INPUT_TEXT_COLUMN = "input_text"         # 评估器读取的原始输入列名
```

## 🎨 自定义提示词

### 修改生成提示词

编辑 `config.py`:

```python
# 系统提示词
SYSTEM_PROMPT = """You are a professional translator.
Your task is to translate the given text accurately and naturally.
"""

# 用户提示词模板
USER_PROMPT_TEMPLATE = "Translate this to English: {input_text}"
```

### 修改评估提示词

编辑 `config.py` 中的 `EVALUATION_PROMPT`:

```python
EVALUATION_PROMPT = """Evaluate the text based on:
- Clarity (1-10): How clear and understandable is the text?
- Accuracy (1-10): How accurate and correct is the information?
- Fluency (1-10): How natural and fluent is the language?

Original Input: {original_text}
Model Output: {model_output}

Return ONLY JSON:
{{
    "determined_priority": "P0/P1/P2/P3",
    "scores": {{
        "factuality_score": <0-10>,
        "completeness_score": <0-10>,
        "adherence_score": <0-10>,
        "attractiveness_score": <0-10>
    }},
    "weighted_total_score": <calculated_total>,
    "reasoning": "<explanation>",
    "pass": <true/false>
}}"""
```

## 🔍 结果提取功能

生成器支持从 LLM 输出中自动提取结构化结果：

- **优先级提取**: 自动提取 `<priority>P0</priority>` 等标签
- **内容提取**: 自动提取 `<content>...</content>` 标签中的内容
- **清理功能**: 自动清理内容中的杂质标签（如 `<Title>`, `<Brief>` 等）

如果输出不包含这些标签，会使用兜底逻辑：移除 `<thinking>` 标签后返回剩余内容。

## ⚡ 性能优化

- **并发处理**: 默认使用 5 个线程，可在 `config.py` 中调整 `MAX_WORKERS`
- **错误处理**: API 调用失败会自动标记为错误，不会中断整个流程
- **进度显示**: 实时显示处理进度和预计剩余时间
- **编码自动检测**: 自动尝试多种编码格式读取 CSV 文件

## 🌐 支持的 API 提供商

### OpenAI
- 模型: gpt-4, gpt-4-turbo, gpt-4o-mini, gpt-3.5-turbo 等
- 环境变量: `OPENAI_API_KEY`
- 获取密钥: https://platform.openai.com/api-keys

### Anthropic
- 模型: claude-3-5-sonnet, claude-3-opus, claude-3-haiku 等
- 环境变量: `ANTHROPIC_API_KEY`
- 获取密钥: https://console.anthropic.com/

### DeepSeek
- 模型: deepseek-chat, deepseek-reasoner 等
- 环境变量: `DEEPSEEK_API_KEY`
- 获取密钥: https://platform.deepseek.com/
- 使用 OpenAI 兼容的 API

## ⚠️ 注意事项

1. **API 密钥安全**: 不要将 `.env` 文件提交到版本控制系统
2. **API 限制**: 注意 API 的速率限制，适当调整 `MAX_WORKERS`
3. **CSV 编码**: 输出文件使用 UTF-8-BOM 编码，确保中文正确显示
4. **JSON 解析**: 评估器会自动从响应中提取 JSON，支持多种格式（纯 JSON、代码块中的 JSON 等）
5. **提示词设计**: 确保评估提示词明确要求返回 JSON 格式，并指定 JSON 结构

## 🔧 故障排除

### 常见错误

1. **找不到 API 密钥**
   - 检查 `.env` 文件是否存在且包含正确的密钥
   - 确认密钥名称与 `config.py` 中的 API_PROVIDER 匹配

2. **CSV 列不存在**
   - 检查输入 CSV 是否包含必需的列
   - 确认 `config.py` 中的列名配置正确

3. **JSON 解析失败**
   - 检查评估提示词是否明确要求返回 JSON
   - 查看错误详情中的原始响应
   - 确保 JSON 格式正确

4. **API 调用失败**
   - 检查 API 密钥是否正确
   - 检查 API 账户余额
   - 降低 `MAX_WORKERS` 值（可能触发速率限制）
   - 检查网络连接

## 📄 许可证

MIT License

## About

Developed by **nyota佳树** (AI Product Manager / INTJ). 致力于探索 AI Native 产品的落地与评估方法论。

- 公众号：Nyota佳树
- 小红书：Nyota佳树
- X：Nyota佳树

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**祝你使用愉快！** 🎉
