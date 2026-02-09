"""
LLM 评测流水线 - Streamlit 应用
四阶段流程：配置 → 提示词确认 → 评测中 → 结果展示
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
REQUIRED_CSV_COLUMNS = ["question", "expected_answer"]
DEFAULT_MODEL = "deepseek-chat"
MODEL_OPTIONS = ["deepseek-chat", "deepseek-reasoner"]
PHASES = ["CONFIG", "PROMPT_EDIT", "EVALUATING", "RESULT"]


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
    """生成示例 CSV 模板（question, expected_answer）"""
    template_df = pd.DataFrame({
        "question": [
            "示例问题 1：请简述合规要点",
            "示例问题 2：该场景下应如何回复客户？",
        ],
        "expected_answer": [
            "示例期望回答 1：合规要点包括…",
            "示例期望回答 2：应首先确认身份…",
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


def run_single_evaluation(
    row: pd.Series,
    evaluation_prompt: str,
    api_key: str,
    model: str,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    对单行执行评测。返回 (评估结果字典, 错误信息)。
    使用 question -> original_text, expected_answer -> model_output。
    """
    original_text = str(row.get("question", ""))
    model_output = str(row.get("expected_answer", ""))
    if not original_text.strip() or not model_output.strip() or model_output.lower() in ("nan", ""):
        return None, "问题或期望回答为空，已跳过"

    try:
        prompt_filled = evaluation_prompt.format(
            original_text=original_text,
            model_output=model_output,
        )
    except KeyError as e:
        return None, f"评测提示词缺少占位符: {e}。请确保包含 {original_text} 和 {model_output} 的占位符。"

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
    uploaded = st.file_uploader("上传评测数据（仅限 CSV）", type=["csv"], help="需包含 question、expected_answer 列")

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
    if st.button("生成评测方案", type="primary", use_container_width=False):
        if not st.session_state.api_key.strip():
            st.error("请在侧边栏填写 API Key。")
            return
        if not st.session_state.scenario.strip() or not st.session_state.north_star.strip():
            st.error("请填写测试场景和北极星指标。")
            return
        if st.session_state.uploaded_df is None or st.session_state.uploaded_df.empty:
            st.error("请先上传包含 question、expected_answer 的 CSV 文件。")
            return

        with st.spinner("正在根据场景与北极星指标生成评测方案…"):
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


# ==================== Phase 2: 提示词确认 ====================
def render_phase_prompt_edit():
    st.subheader("阶段二：提示词确认")
    st.divider()

    evaluation_prompt = st.text_area(
        "评测 System Prompt（可编辑）",
        value=st.session_state.evaluation_prompt,
        height=320,
        help="可根据需要修改生成的评测标准",
    )
    st.session_state.evaluation_prompt = evaluation_prompt

    # 占位符检查
    if "{original_text}" not in evaluation_prompt or "{model_output}" not in evaluation_prompt:
        st.warning("提示词中建议包含占位符 `{original_text}` 与 `{model_output}`，以便对每条题目进行评测。")

    st.divider()
    if st.button("确认并开始评测", type="primary", use_container_width=False):
        if not st.session_state.evaluation_prompt.strip():
            st.error("请填写或保留评测提示词。")
            return
        st.session_state.phase = "EVALUATING"
        st.rerun()


# ==================== Phase 3: 执行评测 ====================
def render_phase_evaluating():
    st.subheader("阶段三：执行评测")
    st.divider()

    df = st.session_state.uploaded_df
    n = len(df)
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
    log_lines = []

    with status:
        for i, (idx, row) in enumerate(df.iterrows()):
            progress_bar.progress((i + 1) / n, text=f"正在评测第 {i+1}/{n} 条…")
            log_lines.append(f"[{i+1}/{n}] 题目: {str(row.get('question', ''))[:50]}…")
            st.write(log_lines[-1])

            result, err = run_single_evaluation(row, evaluation_prompt, api_key, model)
            if err:
                df.at[idx, "decision"] = "ERROR"
                df.at[idx, "reason"] = f"error: {err}"
                log_lines.append(f"  ❌ {err}")
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
                log_lines.append(f"  ✅ 得分: {result.get('weighted_total_score', 0):.1f} | {result.get('decision', '')}")
                st.write(f"  ✅ 得分: {result.get('weighted_total_score', 0):.1f} | {result.get('decision', '')}")

    elapsed = time.time() - start_time
    progress_bar.progress(1.0, text="评测完成")
    status.update(label="评测完成", state="complete")

    st.session_state.results_df = df
    st.session_state.eval_elapsed = elapsed
    st.session_state.phase = "RESULT"
    st.success(f"共评测 {n} 条，耗时 {elapsed:.1f} 秒。")
    st.rerun()


# ==================== Phase 4: 结果展示 ====================
def render_phase_result():
    st.subheader("阶段四：结果展示")
    st.divider()

    df = st.session_state.results_df
    if df is None:
        st.warning("暂无结果，请先完成评测。")
        return

    # 只统计有效评分行
    valid = df[df["weighted_total_score"].notna()]
    n_valid = len(valid)
    n_total = len(df)

    # 核心指标卡片
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

    # 得分分布柱状图
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

    # 完整结果表格
    st.caption("完整结果（含原题、回答、评分理由）")
    display_cols = ["question", "expected_answer", "weighted_total_score", "decision", "reasoning"]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[display_cols] if display_cols else df, use_container_width=True, hide_index=True)

    st.divider()
    # 导出 CSV
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
    st.caption("配置 → 提示词确认 → 评测 → 结果展示")
    st.divider()

    phase = st.session_state.phase
    if phase == "CONFIG":
        render_phase_config()
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
