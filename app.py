"""
LLM 评测流水线 - Streamlit 应用
五阶段流程：配置 → 生成回答 → 提示词确认 → 评测中 → 结果展示
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
PHASES = ["CONFIG", "GENERATING", "PROMPT_EDIT", "EVALUATING", "RESULT"]
GENERATOR_SYSTEM_TEMPLATE = """你是「{scenario}」场景下的专业助手。请严格围绕以下北极星指标来回答：{north_star}。

要求：直接给出专业、完整的回答，不要额外元说明或重复题目。"""


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
    scenario: str,
    north_star: str,
    api_key: str,
    model: str,
) -> tuple[Optional[str], Optional[str]]:
    """根据业务描述与北极星指标对单条题目生成回答。返回 (回答文本, 错误信息)。"""
    if not (question or "").strip():
        return None, "题目为空"
    prev = os.environ.get("DEEPSEEK_API_KEY")
    try:
        os.environ["DEEPSEEK_API_KEY"] = api_key
        client = LLMClient(
            provider="deepseek",
            model=model,
            temperature=getattr(config, "DEEPSEEK_TEMPERATURE", 0.7),
            max_tokens=getattr(config, "DEEPSEEK_MAX_TOKENS", 4000),
        )
        system_prompt = GENERATOR_SYSTEM_TEMPLATE.format(scenario=scenario, north_star=north_star)
        answer = client.generate(system_prompt=system_prompt, user_prompt=(question or "").strip())
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

        if st.button("🔄 重新开始", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.rerun()


# ==================== Phase 1: 配置与上传 ====================
def render_phase_config():
    st.subheader("阶段一：场景定义与上传")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        scenario = st.text_input(
            "测试场景",
            value=st.session_state.scenario,
            placeholder="例如：金融合规助手",
        )
        st.session_state.scenario = scenario
    with c2:
        north_star = st.text_input(
            "北极星指标",
            value=st.session_state.north_star,
            placeholder="例如：专业度、安全性",
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
        st.dataframe(df.head(3), use_container_width=True, hide_index=True)

    st.divider()
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("生成回答", type="primary", use_container_width=True):
            if not st.session_state.api_key.strip():
                st.error("请在侧边栏填写 API Key。")
            elif not st.session_state.scenario.strip() or not st.session_state.north_star.strip():
                st.error("请填写测试场景和北极星指标。")
            elif st.session_state.uploaded_df is None or st.session_state.uploaded_df.empty:
                st.error("请先上传包含 question 的 CSV 文件。")
            else:
                st.session_state.phase = "GENERATING"
                st.rerun()
    with col_btn2:
        has_answer = (
            st.session_state.uploaded_df is not None
            and not st.session_state.uploaded_df.empty
            and (
                OPTIONAL_ANSWER_COLUMN in st.session_state.uploaded_df.columns
                or GENERATED_ANSWER_COLUMN in st.session_state.uploaded_df.columns
            )
        )
        if st.button("已有回答，直接生成评测方案", use_container_width=True, disabled=not has_answer):
            if not st.session_state.api_key.strip():
                st.error("请在侧边栏填写 API Key。")
            elif not st.session_state.scenario.strip() or not st.session_state.north_star.strip():
                st.error("请填写测试场景和北极星指标。")
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


# ==================== Phase 2: 生成回答 ====================
def render_phase_generating():
    st.subheader("阶段二：生成回答")
    st.divider()

    df = st.session_state.uploaded_df
    n = len(df) if df is not None else 0
    api_key = st.session_state.api_key
    model = st.session_state.model
    scenario = st.session_state.scenario
    north_star = st.session_state.north_star

    if not api_key or df is None or n == 0:
        st.error("配置或数据不完整，请返回上一步。")
        if st.button("返回配置"):
            st.session_state.phase = "CONFIG"
            st.rerun()
        return

    if GENERATED_ANSWER_COLUMN not in df.columns:
        df[GENERATED_ANSWER_COLUMN] = None

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
            answer, err = run_single_generation(q, scenario, north_star, api_key, model)
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
    if st.button("下一步：生成评测方案", type="primary", use_container_width=False):
        with st.spinner("正在根据场景与北极星指标生成评测方案…"):
            try:
                prompt = generate_evaluator_prompt_in_app(scenario, north_star, api_key)
                st.session_state.generated_prompt = prompt
                st.session_state.evaluation_prompt = prompt
                st.session_state.phase = "PROMPT_EDIT"
                st.success("评测方案已生成，请确认并编辑下方提示词。")
                st.rerun()
            except Exception as e:
                st.error(f"生成评测方案失败：{e}")


# ==================== Phase 3: 提示词确认 ====================
def render_phase_prompt_edit():
    st.subheader("阶段三：提示词确认")
    st.divider()

    evaluation_prompt = st.text_area(
        "评测 System Prompt（可编辑）",
        value=st.session_state.evaluation_prompt,
        height=320,
        help="可根据需要修改生成的评测标准",
    )
    st.session_state.evaluation_prompt = evaluation_prompt

    if "{original_text}" not in evaluation_prompt or "{model_output}" not in evaluation_prompt:
        st.warning("提示词中建议包含占位符 `{original_text}` 与 `{model_output}`，以便对每条题目进行评测。")

    st.divider()
    if st.button("确认并开始评测", type="primary", use_container_width=False):
        if not st.session_state.evaluation_prompt.strip():
            st.error("请填写或保留评测提示词。")
            return
        st.session_state.phase = "EVALUATING"
        st.rerun()


# ==================== Phase 4: 执行评测 ====================
def render_phase_evaluating():
    st.subheader("阶段四：执行评测")
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
        "eval_priority", "factuality_score", "completeness_score",
        "adherence_score", "attractiveness_score", "weighted_total_score",
        "decision", "reason", "reasoning", "pass",
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
                df.at[idx, "completeness_score"] = result.get("completeness_score")
                df.at[idx, "adherence_score"] = result.get("adherence_score")
                df.at[idx, "attractiveness_score"] = result.get("attractiveness_score")
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


# ==================== Phase 5: 结果展示 ====================
def render_phase_result():
    st.subheader("阶段五：结果展示")
    st.divider()

    df = st.session_state.results_df
    if df is None:
        st.warning("暂无结果，请先完成评测。")
        return

    valid = df[df["weighted_total_score"].notna()]
    n_valid = len(valid)
    n_total = len(df)

    st.caption("核心指标")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        avg_score = valid["weighted_total_score"].mean() if n_valid else 0
        st.metric("平均分", f"{avg_score:.1f}" if n_valid else "—")
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
        score_counts = valid["weighted_total_score"].round(0).value_counts().sort_index()
        fig = px.bar(
            x=score_counts.index.astype(int),
            y=score_counts.values,
            labels={"x": "加权总分", "y": "条数"},
            title="加权总分分布",
        )
        fig.update_layout(showlegend=False, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
    st.divider()

    st.caption("完整结果（含原题、回答、评分理由）")
    display_cols = ["question", GENERATED_ANSWER_COLUMN, OPTIONAL_ANSWER_COLUMN, "weighted_total_score", "decision", "reasoning"]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[display_cols] if display_cols else df, use_container_width=True, hide_index=True)

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
    st.caption("配置 → 生成回答 → 提示词确认 → 评测 → 结果展示")
    st.divider()

    phase = st.session_state.phase
    if phase == "CONFIG":
        render_phase_config()
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
