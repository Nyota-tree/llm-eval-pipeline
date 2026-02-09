"""
LLM 评测流水线 - Streamlit 应用
六阶段流程：配置 → 业务 Prompt → 评估 Prompt → 生成回答 → 评测 → 结果展示
"""

import io
import os
import time
import re
import json
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
import plotly.express as px

# 项目模块（需在运行时可用）
import config
from utils import LLMClient
from generate_evaluator_prompt import PROMPT_GENERATOR_SYSTEM_PROMPT
from batch_evaluator import extract_json_from_text, extract_evaluation


def generate_evaluator_prompt_in_app(scenario: str, north_star_metric: str, api_key: str) -> str:
    """在应用内生成评测方案（不调用 sys.exit，便于 Streamlit 展示错误）。"""
    user_prompt = f"""场景：{scenario}
北极星指标：{north_star_metric}

请根据以上信息生成完整的评估员 Prompt。"""
    prev = os.environ.get("DEEPSEEK_API_KEY")
    try:
        os.environ["DEEPSEEK_API_KEY"] = api_key
        llm_client = LLMClient(provider="deepseek", model=st.session_state.get("model", DEFAULT_MODEL))
        return llm_client.generate(system_prompt=PROMPT_GENERATOR_SYSTEM_PROMPT, user_prompt=user_prompt)
    finally:
        if prev is not None:
            os.environ["DEEPSEEK_API_KEY"] = prev
        else:
            os.environ.pop("DEEPSEEK_API_KEY", None)


def generate_generation_prompt_in_app(scenario: str, north_star: str, api_key: str) -> str:
    """根据业务场景与北极星指标，调用大模型生成高质量的「生成 Prompt」。"""
    user_prompt = f"""业务场景：
{scenario}

北极星指标：
{north_star}

请根据以上信息生成一份可直接用于生产环境的业务提示词（仅输出提示词正文，无需额外说明）。"""
    prev = os.environ.get("DEEPSEEK_API_KEY")
    try:
        os.environ["DEEPSEEK_API_KEY"] = api_key
        llm_client = LLMClient(provider="deepseek", model=st.session_state.get("model", DEFAULT_MODEL))
        return llm_client.generate(
            system_prompt=GENERATION_PROMPT_GENERATOR_SYSTEM,
            user_prompt=user_prompt,
        )
    finally:
        if prev is not None:
            os.environ["DEEPSEEK_API_KEY"] = prev
        else:
            os.environ.pop("DEEPSEEK_API_KEY", None)


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="LLM 评测流水线",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 常量 ====================
REQUIRED_CSV_COLUMNS = ["question"]
OPTIONAL_ANSWER_COLUMN = "expected_answer"
GENERATED_ANSWER_COLUMN = "generated_answer"
DEFAULT_MODEL = "deepseek-chat"
MODEL_OPTIONS = ["deepseek-chat", "deepseek-reasoner"]
PHASES = ["CONFIG", "GENERATION_PROMPT_EDIT", "GENERATING", "PROMPT_EDIT", "EVALUATING", "RESULT"]

# 调用大模型生成「生成 Prompt」时使用的系统提示词
GENERATION_PROMPT_GENERATOR_SYSTEM = """# Role
你是一位世界级的提示词工程师，擅长将业务需求转化为极具执行力的 LLM 生产环境提示词。你深谙提示词工程的最佳实践，能够针对特定业务指标进行精细化调优。

# Task
根据用户提供的【业务场景】和【北极星指标】，为用户撰写一个高质量的、可直接投入生产使用的"业务提示词"。

# Input
1. 业务场景：描述该 AI 产品的具体用途和用户画像。
2. 北极星指标：衡量该 AI 表现好坏的核心业务标准。

# Output Framework (你生成的业务提示词必须包含)
1. **角色设定 (Role)**：定义一个符合场景的、专业的、具有特定人格特质的 AI 角色。
2. **任务描述 (Task)**：清晰、无歧义地描述 AI 需要完成的具体工作。
3. **约束条件 (Constraints)**：
    - 根据【北极星指标】反推的硬性要求（例如：严禁胡说八道、回复长度限制、必须包含某些关键词）。
    - 针对【业务场景】的合规性要求或语气要求。
4. **思维链要求 (Chain of Thought)**：引导 AI 在输出结果前进行内部推理，以确保逻辑严密（如果场景复杂）。
5. **输出格式 (Output Format)**：定义回复的结构（如 Markdown、JSON 或特定语气的文本）。

# Design Principles
- **指标对齐**：如果北极星指标是"专业度"，提示词应侧重引用知识库和术语；如果是"亲和力"，则侧重语气词和共情表达。
- **模块化**：结构清晰，用户拿到后可以一眼看懂每一部分的作用。
- **鲁棒性**：考虑到大模型的边界情况，增加预防误操作或越权的防御性描述。

# Language
始终使用中文输出生成的业务提示词。"""


def init_session_state():
    """初始化 session_state"""
    if "phase" not in st.session_state:
        st.session_state.phase = "CONFIG"
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL
    if "scenario" not in st.session_state:
        st.session_state.scenario = ""
    if "north_star" not in st.session_state:
        st.session_state.north_star = ""
    if "uploaded_df" not in st.session_state:
        st.session_state.uploaded_df = None
    if "generation_prompt" not in st.session_state:
        st.session_state.generation_prompt = ""
    if "generated_prompt" not in st.session_state:
        st.session_state.generated_prompt = ""
    if "evaluation_prompt" not in st.session_state:
        st.session_state.evaluation_prompt = ""
    if "results_df" not in st.session_state:
        st.session_state.results_df = None
    if "eval_elapsed" not in st.session_state:
        st.session_state.eval_elapsed = None


def get_csv_template_bytes() -> bytes:
    """生成示例 CSV 模板（仅 question，用于「生成回答」流程）"""
    template_df = pd.DataFrame({
        "question": [
            "示例问题 1：请简述合规要点",
            "示例问题 2：该场景下应如何回复客户？",
        ],
    })
    buf = io.BytesIO()
    template_df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue()


def validate_csv(df: pd.DataFrame) -> tuple[bool, str]:
    """验证 CSV 是否包含必需列。返回 (是否通过, 错误信息)。"""
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        return False, f"CSV 缺少必需列: {', '.join(missing)}。需要: {', '.join(REQUIRED_CSV_COLUMNS)}"
    if df.empty:
        return False, "CSV 为空，请上传至少包含一行的数据。"
    return True, ""


def run_single_generation(
    question: str,
    system_prompt: str,
    api_key: str,
    model: str,
) -> tuple[Optional[str], Optional[str]]:
    """使用给定的生成 Prompt 对单条题目生成回答。返回 (回答文本, 错误信息)。"""
    if not (question or "").strip():
        return None, "题目为空"
    if not (system_prompt or "").strip():
        return None, "生成 Prompt 为空"
    prev = os.environ.get("DEEPSEEK_API_KEY")
    try:
        os.environ["DEEPSEEK_API_KEY"] = api_key
        client = LLMClient(
            provider="deepseek",
            model=model,
            temperature=getattr(config, "DEEPSEEK_TEMPERATURE", 0.7),
            max_tokens=getattr(config, "DEEPSEEK_MAX_TOKENS", 4000),
        )
        answer = client.generate(system_prompt=system_prompt.strip(), user_prompt=(question or "").strip())
        return (answer or "").strip(), None
    except Exception as e:
        return None, str(e)
    finally:
        if prev is not None:
            os.environ["DEEPSEEK_API_KEY"] = prev
        else:
            os.environ.pop("DEEPSEEK_API_KEY", None)


def _fill_evaluation_prompt(prompt: str, original_text: str, model_output: str) -> str:
    """仅替换 {original_text} 与 {model_output}，避免 JSON 等花括号被 format 误解析。"""
    return prompt.replace("{original_text}", original_text).replace("{model_output}", model_output)


def run_single_evaluation(
    row: pd.Series,
    evaluation_prompt: str,
    api_key: str,
    model: str,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """对单行执行评测。model_output 优先用 generated_answer，否则 expected_answer。"""
    original_text = str(row.get("question", ""))
    model_output = str(
        row.get(GENERATED_ANSWER_COLUMN) or row.get(OPTIONAL_ANSWER_COLUMN) or ""
    ).strip()
    if not original_text.strip():
        return None, "问题为空，已跳过"
    if not model_output or model_output.lower() in ("nan", ""):
        return None, "该行无回答内容（需先「生成回答」或上传带 expected_answer 的 CSV），已跳过"

    prompt_filled = _fill_evaluation_prompt(evaluation_prompt, original_text, model_output)
    prev_key = os.environ.get("DEEPSEEK_API_KEY")
    try:
        os.environ["DEEPSEEK_API_KEY"] = api_key
        client = LLMClient(
            provider="deepseek",
            model=model,
            temperature=getattr(config, "DEEPSEEK_REASONER_TEMPERATURE", 0.0),
            max_tokens=getattr(config, "DEEPSEEK_REASONER_MAX_TOKENS", 2000),
            top_p=getattr(config, "DEEPSEEK_REASONER_TOP_P", None),
        )
        response = client.generate(system_prompt="", user_prompt=prompt_filled)
    except Exception as e:
        return None, str(e)
    finally:
        if prev_key is not None:
            os.environ["DEEPSEEK_API_KEY"] = prev_key
        else:
            os.environ.pop("DEEPSEEK_API_KEY", None)

    json_obj = extract_json_from_text(response)
    if json_obj is None:
        return None, f"无法从响应中提取 JSON: {response[:200]}…"

    try:
        evaluation_result = extract_evaluation(json_obj)
        return evaluation_result, None
    except Exception as e:
        return None, str(e)


# ==================== 侧边栏 ====================
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ 配置")
        st.divider()

        api_key = st.text_input(
            "API Key",
            value=st.session_state.get("api_key", ""),
            type="password",
            placeholder="sk-…",
            help="DeepSeek API Key",
        )
        st.session_state.api_key = api_key

        model = st.selectbox(
            "Model",
            options=MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(st.session_state.get("model", DEFAULT_MODEL)),
            help="评测使用的模型",
        )
        st.session_state.model = model

        st.divider()
        st.caption("数据模板")
        template_bytes = get_csv_template_bytes()
        st.download_button(
            label="下载 CSV 模板",
            data=template_bytes,
            file_name="eval_template.csv",
            mime="text/csv",
        )
        st.divider()

        if st.button("🔄 重新开始", width="stretch"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.rerun()


# ==================== Phase 1: 配置与上传 ====================
def render_phase_config():
    st.subheader("阶段一：场景定义与上传")
    st.divider()

    st.caption("业务场景（请尽量详细描述，越详细生成的 Prompt 越精准）")
    scenario = st.text_area(
        "业务场景",
        value=st.session_state.scenario,
        height=220,
        placeholder="请详细描述：\n· 该 AI 产品的具体用途、目标用户画像\n· 典型使用场景与边界情况\n· 希望的语气、风格或合规要求\n\n例如：面向 6–10 岁儿童的故事生成助手，需在 2 分钟内产出 300 字以内、无暴力情节、语言浅白有趣的故事片段……",
        help="描述越详细，大模型生成的业务提示词越贴合你的需求。",
    )
    st.session_state.scenario = scenario

    st.divider()
    north_star = st.text_input(
        "北极星指标",
        value=st.session_state.north_star,
        placeholder="例如：趣味性、符合儿童心智、专业度、安全性",
        help="衡量该 AI 表现好坏的核心业务标准，可写多条。",
    )
    st.session_state.north_star = north_star

    st.divider()
    uploaded = st.file_uploader("上传评测数据（仅限 CSV）", type=["csv"], help="需包含 question 列；可选 expected_answer（有则可不生成直接评测）")

    if uploaded is not None:
        df = None
        last_err = None
        for enc in ("utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"):
            try:
                uploaded.seek(0)
                df = pd.read_csv(uploaded, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                last_err = e
                break
        if df is None:
            st.error(f"文件解析失败：{last_err or '无法识别的编码'}. 请使用 UTF-8 或 GBK 编码的 CSV。")
            return
        ok, err = validate_csv(df)
        if not ok:
            st.error(err)
            return
        st.session_state.uploaded_df = df
        st.caption("预览（前 3 行）")
        st.dataframe(df.head(3), width="stretch", hide_index=True)

    st.divider()
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("下一步：生成 Prompt", type="primary", width="stretch"):
            if not st.session_state.api_key.strip():
                st.error("请在侧边栏填写 API Key。")
            elif not st.session_state.scenario.strip() or not st.session_state.north_star.strip():
                st.error("请填写业务场景和北极星指标。")
            elif st.session_state.uploaded_df is None or st.session_state.uploaded_df.empty:
                st.error("请先上传包含 question 的 CSV 文件。")
            else:
                with st.spinner("正在调用大模型生成业务提示词…"):
                    try:
                        st.session_state.generation_prompt = generate_generation_prompt_in_app(
                            st.session_state.scenario,
                            st.session_state.north_star,
                            st.session_state.api_key,
                        )
                        st.session_state.phase = "GENERATION_PROMPT_EDIT"
                        st.rerun()
                    except Exception as e:
                        st.error(f"生成 Prompt 失败（请检查 API Key 与网络）：{e}")
    with col_btn2:
        has_answer = (
            st.session_state.uploaded_df is not None
            and not st.session_state.uploaded_df.empty
            and (
                OPTIONAL_ANSWER_COLUMN in st.session_state.uploaded_df.columns
                or GENERATED_ANSWER_COLUMN in st.session_state.uploaded_df.columns
            )
        )
        if st.button("已有回答，直接生成评测方案", width="stretch", disabled=not has_answer):
            if not st.session_state.api_key.strip():
                st.error("请在侧边栏填写 API Key。")
            elif not st.session_state.scenario.strip() or not st.session_state.north_star.strip():
                st.error("请填写业务场景和北极星指标。")
            else:
                with st.spinner("正在生成评测方案…"):
                    try:
                        prompt = generate_evaluator_prompt_in_app(
                            st.session_state.scenario,
                            st.session_state.north_star,
                            st.session_state.api_key,
                        )
                        st.session_state.generated_prompt = prompt
                        st.session_state.evaluation_prompt = prompt
                        st.session_state.phase = "PROMPT_EDIT"
                        st.success("评测方案已生成，请确认并编辑下方提示词。")
                        st.rerun()
                    except Exception as e:
                        st.error(f"生成评测方案失败（请检查 API Key 与网络）：{e}")


# ==================== Phase 2: 业务 Prompt 确认（可编辑） ====================
def render_phase_generation_prompt_edit():
    st.subheader("阶段二：业务 Prompt 确认")
    st.divider()
    st.caption("根据业务场景与北极星指标已生成下方业务提示词，可编辑。确认后进入下一步生成「评估 Prompt」。")

    generation_prompt = st.text_area(
        "业务 Prompt（可编辑）",
        value=st.session_state.generation_prompt,
        height=280,
        help="用于调用模型生成回答的系统提示词，每条题目将作为用户输入传入。",
    )
    st.session_state.generation_prompt = generation_prompt

    st.divider()
    if st.button("下一步：生成评估 Prompt", type="primary", width="content"):
        if not (st.session_state.generation_prompt or "").strip():
            st.error("请填写或保留业务 Prompt。")
            return
        with st.spinner("正在根据场景与北极星指标生成评估 Prompt…"):
            try:
                prompt = generate_evaluator_prompt_in_app(
                    st.session_state.scenario,
                    st.session_state.north_star,
                    st.session_state.api_key,
                )
                st.session_state.generated_prompt = prompt
                st.session_state.evaluation_prompt = prompt
                st.session_state.phase = "PROMPT_EDIT"
                st.success("评估 Prompt 已生成，请确认并编辑下方提示词，再开始处理数据。")
                st.rerun()
            except Exception as e:
                st.error(f"生成评估 Prompt 失败（请检查 API Key 与网络）：{e}")

    if st.button("返回配置", width="content"):
        st.session_state.phase = "CONFIG"
        st.rerun()


# ==================== Phase 4: 生成回答 ====================
def render_phase_generating():
    st.subheader("阶段四：生成回答")
    st.divider()

    df = st.session_state.uploaded_df
    n = len(df) if df is not None else 0
    api_key = st.session_state.api_key
    model = st.session_state.model
    generation_prompt = st.session_state.generation_prompt

    if not api_key or df is None or n == 0 or not (generation_prompt or "").strip():
        st.error("配置或数据不完整，请返回上一步。")
        if st.button("返回评估 Prompt"):
            st.session_state.phase = "PROMPT_EDIT"
            st.rerun()
        return

    if GENERATED_ANSWER_COLUMN not in df.columns:
        df[GENERATED_ANSWER_COLUMN] = None

    # 若所有有题目的行已有生成回答，则不再重新生成，直接展示结果与「下一步」（避免点击「生成评测方案」后 rerun 时又跑一遍）
    rows_with_question = (df["question"].notna() & df["question"].astype(str).str.strip() != "").sum()
    if rows_with_question > 0 and GENERATED_ANSWER_COLUMN in df.columns:
        filled = (df[GENERATED_ANSWER_COLUMN].notna() & (df[GENERATED_ANSWER_COLUMN].astype(str).str.strip() != "")).sum()
        if filled >= rows_with_question:
            st.caption("生成已完成，可直接点击下方「下一步：开始评测」。")
            st.session_state.uploaded_df = df
            # 跳转到下方的「生成结果 + 下一步」展示，不执行下面的 for 循环
            _render_generation_result_and_next(df, api_key)
            return

    progress_bar = st.progress(0.0, text="准备中…")
    status = st.status("生成回答中…", expanded=True)

    with status:
        for i, (idx, row) in enumerate(df.iterrows()):
            progress_bar.progress((i + 1) / n, text=f"正在生成第 {i+1}/{n} 条…")
            q = str(row.get("question", "") or "").strip()
            st.write(f"[{i+1}/{n}] {q[:60]}…" if len(q) > 60 else f"[{i+1}/{n}] {q}")
            if not q:
                df.at[idx, GENERATED_ANSWER_COLUMN] = ""
                st.write("  ⏭ 题目为空，已跳过")
                continue
            answer, err = run_single_generation(q, generation_prompt, api_key, model)
            if err:
                df.at[idx, GENERATED_ANSWER_COLUMN] = ""
                st.write(f"  ❌ {err}")
            else:
                df.at[idx, GENERATED_ANSWER_COLUMN] = answer or ""
                st.write("  ✅ 已生成")

    progress_bar.progress(1.0, text="生成完成")
    status.update(label="生成完成", state="complete")
    st.session_state.uploaded_df = df

    st.divider()
    _render_generation_result_and_next(df, api_key)


def _render_generation_result_and_next(df: pd.DataFrame, api_key: str):
    """展示生成结果（只读）与「下一步：开始评测」按钮（评估 Prompt 已在前面步骤生成并确认）。"""
    st.subheader("生成结果（只读）")
    st.caption("以下为根据当前业务 Prompt 得到的回答，仅供查看不可修改。确认后点击「下一步：开始评测」。")
    if GENERATED_ANSWER_COLUMN in df.columns:
        display_df = df[["question", GENERATED_ANSWER_COLUMN]].copy()
        display_df.columns = ["题目", "生成回答"]
        st.dataframe(display_df, width="stretch", hide_index=True)
    st.divider()
    if st.button("下一步：开始评测", type="primary", width="content"):
        st.session_state.phase = "EVALUATING"
        st.rerun()


# ==================== Phase 3: 评估 Prompt 确认（可编辑） ====================
def render_phase_prompt_edit():
    st.subheader("阶段三：评估 Prompt 确认")
    st.divider()
    st.caption("上一步已生成评估用提示词，可编辑。确认后进入「处理数据」：先生成回答，再执行评测。")

    evaluation_prompt = st.text_area(
        "评估 Prompt（可编辑）",
        value=st.session_state.evaluation_prompt,
        height=320,
        help="可根据需要修改生成的评测标准；须包含占位符 {original_text} 与 {model_output}。",
    )
    st.session_state.evaluation_prompt = evaluation_prompt

    if "{original_text}" not in evaluation_prompt or "{model_output}" not in evaluation_prompt:
        st.warning("提示词中建议包含占位符 `{original_text}` 与 `{model_output}`，以便对每条题目进行评测。")

    st.divider()
    if st.button("确认并开始处理数据", type="primary", width="content"):
        if not st.session_state.evaluation_prompt.strip():
            st.error("请填写或保留评估提示词。")
            return
        st.session_state.phase = "GENERATING"
        st.rerun()

    if st.button("返回业务 Prompt", width="content"):
        st.session_state.phase = "GENERATION_PROMPT_EDIT"
        st.rerun()


# ==================== Phase 5: 执行评测 ====================
def render_phase_evaluating():
    st.subheader("阶段五：执行评测")
    st.divider()

    df = st.session_state.uploaded_df
    n = len(df) if df is not None else 0
    api_key = st.session_state.api_key
    model = st.session_state.model
    evaluation_prompt = st.session_state.evaluation_prompt

    if not api_key:
        st.error("请先在侧边栏填写 API Key。")
        st.session_state.phase = "PROMPT_EDIT"
        return
    if df is None or n == 0:
        st.error("无有效数据，请返回上传 CSV。")
        st.session_state.phase = "CONFIG"
        return

    progress_bar = st.progress(0.0, text="准备中…")
    status = st.status("评测进行中…", expanded=True)

    eval_columns = [
        "eval_priority", "factuality_score", "north_star_score", "completeness_score",
        "weighted_total_score", "decision", "reason", "reasoning", "pass",
    ]
    for col in eval_columns:
        if col not in df.columns:
            df[col] = None

    start_time = time.time()
    with status:
        for i, (idx, row) in enumerate(df.iterrows()):
            progress_bar.progress((i + 1) / n, text=f"正在评测第 {i+1}/{n} 条…")
            st.write(f"[{i+1}/{n}] 题目: {str(row.get('question', ''))[:50]}…")

            result, err = run_single_evaluation(row, evaluation_prompt, api_key, model)
            if err:
                df.at[idx, "decision"] = "ERROR"
                df.at[idx, "reason"] = f"error: {err}"
                st.write(f"  ❌ {err}")
            else:
                df.at[idx, "eval_priority"] = result.get("priority")
                df.at[idx, "factuality_score"] = result.get("factuality_score")
                df.at[idx, "north_star_score"] = result.get("north_star_score")
                df.at[idx, "completeness_score"] = result.get("completeness_score")
                df.at[idx, "weighted_total_score"] = result.get("weighted_total_score")
                df.at[idx, "decision"] = result.get("decision")
                df.at[idx, "reason"] = result.get("reason")
                df.at[idx, "reasoning"] = result.get("reasoning")
                df.at[idx, "pass"] = result.get("pass")
                st.write(f"  ✅ 得分: {result.get('weighted_total_score', 0):.1f} | {result.get('decision', '')}")

    elapsed = time.time() - start_time
    progress_bar.progress(1.0, text="评测完成")
    status.update(label="评测完成", state="complete")

    st.session_state.results_df = df
    st.session_state.eval_elapsed = elapsed
    st.session_state.phase = "RESULT"
    st.success(f"共评测 {n} 条，耗时 {elapsed:.1f} 秒。")
    st.rerun()


# ==================== Phase 6: 结果展示 ====================
def render_phase_result():
    st.subheader("阶段六：结果展示")
    st.divider()

    df = st.session_state.results_df
    if df is None:
        st.warning("暂无结果，请先完成评测。")
        return

    score_numeric = pd.to_numeric(df["weighted_total_score"], errors="coerce") if "weighted_total_score" in df.columns else pd.Series(dtype=float)
    valid = df[score_numeric.notna()]
    n_valid = len(valid)
    n_total = len(df)

    st.caption("核心指标")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        avg_score = score_numeric.mean() if score_numeric.notna().any() else 0
        st.metric("平均分", f"{avg_score:.1f}" if score_numeric.notna().any() else "—")
    with col2:
        pass_count = valid["pass"].sum() if "pass" in valid.columns else (valid["decision"] == "PUBLISH").sum()
        pass_rate = (pass_count / n_valid * 100) if n_valid else 0
        st.metric("通过率", f"{pass_rate:.1f}%" if n_valid else "—")
    with col3:
        err_count = (df["decision"] == "ERROR").sum()
        st.metric("错误条数", int(err_count))
    with col4:
        st.metric("总条数", n_total)
    with col5:
        elapsed = st.session_state.get("eval_elapsed")
        st.metric("总耗时", f"{elapsed:.1f} 秒" if elapsed is not None else "—")

    st.divider()

    if n_valid > 0 and "weighted_total_score" in df.columns:
        st.caption("得分分布")
        score_series = pd.to_numeric(valid["weighted_total_score"], errors="coerce").dropna()
        if len(score_series) > 0:
            score_counts = score_series.round(0).value_counts().sort_index()
            fig = px.bar(
                x=score_counts.index.astype(int),
                y=score_counts.values,
                labels={"x": "加权总分", "y": "条数"},
                title="加权总分分布",
            )
            fig.update_layout(showlegend=False, margin=dict(t=40))
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("无有效数值得分，跳过得分分布图。")
    st.divider()

    # 本次使用的 Prompt（供核查）
    st.caption("本次使用的 Prompt（供核查）")
    with st.expander("业务 Prompt（生成回答时使用）", expanded=False):
        gen_prompt = st.session_state.get("generation_prompt") or ""
        st.text_area("业务 Prompt", value=gen_prompt, height=200, disabled=True, label_visibility="collapsed")
    with st.expander("评估 Prompt（评测时使用）", expanded=False):
        eval_prompt = st.session_state.get("evaluation_prompt") or ""
        st.text_area("评估 Prompt", value=eval_prompt, height=280, disabled=True, label_visibility="collapsed")
    st.divider()

    # 各维度小分一览（仅展示常用维度，不展示始终为 0 的遵循度/吸引力）
    score_col_order = ["factuality_score", "north_star_score", "completeness_score", "weighted_total_score"]
    score_label_map = {
        "factuality_score": "事实性/安全性",
        "north_star_score": "北极星指标",
        "completeness_score": "完整性与连贯性",
        "weighted_total_score": "加权总分",
    }
    existing_score_cols = [c for c in score_col_order if c in df.columns]
    if existing_score_cols:
        st.caption("各维度小分")
        labels = [score_label_map[c] for c in existing_score_cols]
        score_df = df[existing_score_cols].copy()
        score_df.columns = labels
        st.dataframe(score_df, width="stretch", hide_index=True, column_config={lb: st.column_config.NumberColumn(lb, format="%.1f") for lb in labels})
        st.divider()

    st.caption("完整结果（含原题、回答、各维度小分、总分、决策与理由）")
    st.caption("说明：REJECT 表示「事实性/安全性」分数低于阈值（0–10 分制低于 5 分，或 0–100 分制低于 50 分），与总分无关。")
    display_cols = [
        "question", GENERATED_ANSWER_COLUMN, OPTIONAL_ANSWER_COLUMN,
        "factuality_score", "north_star_score", "completeness_score",
        "weighted_total_score", "decision", "reason", "reasoning",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    col_config = {}
    for col, label in [
        ("factuality_score", "事实性/安全性"),
        ("north_star_score", "北极星指标"),
        ("completeness_score", "完整性与连贯性"),
        ("weighted_total_score", "加权总分"),
    ]:
        if col in display_cols:
            col_config[col] = st.column_config.NumberColumn(label, format="%.1f")
    st.dataframe(
        df[display_cols] if display_cols else df,
        width="stretch",
        hide_index=True,
        column_config=col_config if col_config else None,
    )

    st.divider()
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button(
        label="下载完整结果 CSV",
        data=buf.getvalue(),
        file_name="eval_results.csv",
        mime="text/csv",
    )


# ==================== Main ====================
def main():
    init_session_state()
    render_sidebar()

    st.title("📊 LLM 评测流水线")
    st.caption("配置 → 业务 Prompt → 评估 Prompt → 生成回答 → 评测 → 结果展示")
    st.divider()

    phase = st.session_state.phase
    if phase == "CONFIG":
        render_phase_config()
    elif phase == "GENERATION_PROMPT_EDIT":
        render_phase_generation_prompt_edit()
    elif phase == "GENERATING":
        render_phase_generating()
    elif phase == "PROMPT_EDIT":
        render_phase_prompt_edit()
    elif phase == "EVALUATING":
        render_phase_evaluating()
    elif phase == "RESULT":
        render_phase_result()
    else:
        st.session_state.phase = "CONFIG"
        st.rerun()


if __name__ == "__main__":
    main()
