"""
Script 2: 批量评估器
从 CSV 文件中读取模型响应，使用 LLM 进行评估，并将 JSON 结果展平为单独的列。
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional
import json
import re
import time

# 导入配置
import config

# 导入工具类
from utils import LLMClient

# 加载环境变量
load_dotenv()


def extract_evaluation(response_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    从评估 JSON 中提取并计算评分
    
    Args:
        response_json: 模型返回的 JSON 对象
    
    Returns:
        包含优先级、评分、决策等信息的字典
    """
    priority = response_json.get("determined_priority", "P3")
    scores = response_json.get("scores", {})
    
    # 获取各项分数
    factuality_score = scores.get('factuality_score', 0)
    completeness_score = scores.get('completeness_score', 0)
    adherence_score = scores.get('adherence_score', 0)
    attractiveness_score = scores.get('attractiveness_score', 0)
    
    # 计算加权总分 (代码二次校验，防止模型算错)
    weighted_score = (
        factuality_score * 3 +
        completeness_score * 2 +
        adherence_score * 2.5 +
        attractiveness_score * 2.5
    )
    
    # 逻辑判定
    # 1. 幻觉熔断
    if factuality_score < 5:
        decision = "REJECT"  # 直接丢弃
        reason = "Hallucination Detected"
    # 2. 质量门槛 (75 分才发布)
    elif weighted_score >= 75:
        decision = "PUBLISH"
        reason = "High Quality Score"
    else:
        decision = "REVIEW"  # 人工复核
        reason = "Low Quality Score"
    
    return {
        "priority": priority,
        "factuality_score": factuality_score,
        "completeness_score": completeness_score,
        "adherence_score": adherence_score,
        "attractiveness_score": attractiveness_score,
        "weighted_total_score": weighted_score,
        "decision": decision,
        "reason": reason,
        "reasoning": response_json.get("reasoning", ""),
        "pass": response_json.get("pass", False)
    }


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    从文本中提取 JSON 对象
    处理可能包含在代码块或其他文本中的 JSON
    """
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 尝试提取 JSON 代码块
    json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_block_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 尝试查找第一个 { ... } 块
    brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(brace_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None


def flatten_json(json_obj: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    将嵌套的 JSON 对象展平为一维字典
    例如: {"clarity": 5, "accuracy": 4, "reasoning": "good"} 
    -> {"score_clarity": 5, "score_accuracy": 4, "eval_reasoning": "good"}
    """
    flattened = {}
    
    for key, value in json_obj.items():
        # 处理 reasoning 字段
        if key.lower() == "reasoning":
            new_key = config.REASONING_COLUMN_NAME
        else:
            # 使用前缀
            new_key = f"{config.EVAL_COLUMN_PREFIX}{key}"
        
        # 如果值是字典，递归展平
        if isinstance(value, dict):
            nested = flatten_json(value, prefix=new_key + "_")
            flattened.update(nested)
        else:
            flattened[new_key] = value
    
    return flattened


def process_single_row(args: tuple) -> tuple:
    """处理单行数据"""
    idx, row, llm_client = args
    
    try:
        # 获取原始新闻和模型输出
        original_text = str(row[config.INPUT_TEXT_COLUMN])
        model_output = str(row[config.FINAL_CONTENT_COLUMN])
        
        # 跳过错误行
        if model_output.startswith("error:") or not model_output or model_output == "nan":
            return (idx, None, f"跳过错误行或空内容")
        
        # 格式化评估提示词
        eval_prompt = config.EVALUATION_PROMPT.format(
            original_text=original_text,
            model_output=model_output
        )
        
        # 调用 LLM 进行评估（使用空系统提示词，因为提示词已经在 eval_prompt 中）
        response = llm_client.generate(
            system_prompt="",
            user_prompt=eval_prompt
        )
        
        # 提取 JSON
        json_obj = extract_json_from_text(response)
        
        if json_obj is None:
            return (idx, None, f"无法从响应中提取 JSON: {response[:100]}")
        
        # 提取并计算评估结果
        evaluation_result = extract_evaluation(json_obj)
        
        return (idx, evaluation_result, None)
    
    except Exception as e:
        return (idx, None, str(e))


def validate_csv_columns(df: pd.DataFrame, required_columns: list) -> None:
    """验证 CSV 是否包含必需的列"""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV 文件缺少必需的列: {', '.join(missing_columns)}")


def batch_evaluate(input_csv: str, output_csv: str) -> None:
    """
    批量评估响应
    
    Args:
        input_csv: 输入 CSV 文件路径（应包含 input_text 和 final_content 列）
        output_csv: 输出 CSV 文件路径
    """
    # 加载 CSV（尝试多种编码）
    print(f"正在加载 CSV 文件: {input_csv}")
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(input_csv, encoding=enc)
            print(f"成功使用编码: {enc}")
            break
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        except Exception as e:
            if "codec can't decode" not in str(e):
                raise
    
    if df is None:
        print(f"错误: 无法读取 CSV 文件，尝试了多种编码都失败")
        sys.exit(1)
    
    # 验证必需的列
    required_columns = [config.INPUT_TEXT_COLUMN, config.FINAL_CONTENT_COLUMN]
    validate_csv_columns(df, required_columns)
    
    # 初始化 LLM 客户端（使用评估器配置）
    evaluator_provider = getattr(config, 'EVALUATOR_API_PROVIDER', config.API_PROVIDER)
    print(f"正在初始化 {evaluator_provider.upper()} 评估客户端...")
    try:
        # 使用评估器专用的模型配置
        if evaluator_provider.lower() == "deepseek":
            model = getattr(config, 'DEEPSEEK_REASONER_MODEL', config.DEEPSEEK_MODEL)
            temperature = getattr(config, 'DEEPSEEK_REASONER_TEMPERATURE', config.DEEPSEEK_TEMPERATURE)
            max_tokens = getattr(config, 'DEEPSEEK_REASONER_MAX_TOKENS', config.DEEPSEEK_MAX_TOKENS)
            top_p = getattr(config, 'DEEPSEEK_REASONER_TOP_P', None)
            llm_client = LLMClient(
                provider=evaluator_provider, 
                model=model, 
                temperature=temperature, 
                max_tokens=max_tokens,
                top_p=top_p
            )
        else:
            llm_client = LLMClient(provider=evaluator_provider)
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)
    
    # 准备数据
    total_rows = len(df)
    print(f"总共需要处理 {total_rows} 行数据")
    
    # 使用线程池并发处理
    results = {}
    errors = {}
    
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_single_row, (idx, row, llm_client)): idx
            for idx, row in df.iterrows()
        }
        
        # 使用 tqdm 显示进度
        with tqdm(total=total_rows, desc="评估响应") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    row_idx, flattened_dict, error = future.result()
                    if error:
                        errors[row_idx] = error
                        results[row_idx] = None
                    else:
                        results[row_idx] = flattened_dict
                except Exception as e:
                    errors[idx] = str(e)
                    results[idx] = None
                finally:
                    pbar.update(1)
    
    # 初始化评估结果列
    eval_columns = [
        "eval_priority", "factuality_score", "completeness_score", 
        "adherence_score", "attractiveness_score", "weighted_total_score",
        "decision", "reason", "reasoning", "pass"
    ]
    for col in eval_columns:
        if col not in df.columns:
            df[col] = None
    
    # 更新 DataFrame
    for idx, evaluation_result in results.items():
        if evaluation_result:
            # 保存评估结果
            df.at[idx, "eval_priority"] = evaluation_result.get("priority")
            df.at[idx, "factuality_score"] = evaluation_result.get("factuality_score")
            df.at[idx, "completeness_score"] = evaluation_result.get("completeness_score")
            df.at[idx, "adherence_score"] = evaluation_result.get("adherence_score")
            df.at[idx, "attractiveness_score"] = evaluation_result.get("attractiveness_score")
            df.at[idx, "weighted_total_score"] = evaluation_result.get("weighted_total_score")
            df.at[idx, "decision"] = evaluation_result.get("decision")
            df.at[idx, "reason"] = evaluation_result.get("reason")
            df.at[idx, "reasoning"] = evaluation_result.get("reasoning")
            df.at[idx, "pass"] = evaluation_result.get("pass")
        else:
            # 标记错误
            error_msg = errors.get(idx, "未知错误")
            df.at[idx, "decision"] = "ERROR"
            df.at[idx, "reason"] = f"error: {error_msg}"
    
    # 保存结果
    print(f"正在保存结果到: {output_csv}")
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    # 统计信息
    success_count = sum(1 for v in results.values() if v is not None)
    error_count = total_rows - success_count
    print(f"\n处理完成!")
    print(f"成功: {success_count} 行")
    print(f"错误: {error_count} 行")
    
    if errors:
        print("\n错误详情:")
        for idx, error_msg in errors.items():
            print(f"  行 {idx}: {error_msg}")
    
    # 显示统计信息
    if "decision" in df.columns:
        decision_counts = df["decision"].value_counts()
        print(f"\n决策分布:")
        for decision, count in decision_counts.items():
            print(f"  {decision}: {count} 条")
    
    if "weighted_total_score" in df.columns:
        valid_scores = df[df["weighted_total_score"].notna()]["weighted_total_score"]
        if len(valid_scores) > 0:
            print(f"\n评分统计:")
            print(f"  平均分: {valid_scores.mean():.2f}")
            print(f"  最高分: {valid_scores.max():.2f}")
            print(f"  最低分: {valid_scores.min():.2f}")
    
    print(f"\n创建的评估列: {', '.join(eval_columns)}")


def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("用法: python batch_evaluator.py <输入CSV文件> <输出CSV文件>")
        print("示例: python batch_evaluator.py output.csv evaluated.csv")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    batch_evaluate(input_csv, output_csv)


if __name__ == "__main__":
    main()
