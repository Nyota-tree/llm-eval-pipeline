"""
Script 1: 批量生成器
从 CSV 文件中读取原始数据，调用 LLM API 生成响应，并保存到新的 CSV 文件。
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, Tuple
import json
import re
import time

# 导入配置
import config

# 导入工具类
from utils import LLMClient

# 加载环境变量
load_dotenv()


def extract_rewrite_result(llm_output: str) -> Tuple[str, str]:
    """
    从 LLM 输出中提取改写结果
    
    Args:
        llm_output: LLM 的完整输出（包含 thinking 和 final_result）
    
    Returns:
        Tuple[priority, final_content]: 提取的优先级和最终内容
    """
    if not llm_output or not isinstance(llm_output, str):
        return "P4", ""
    
    # 提取优先级 (P0-P4)
    priority_regex = r'<priority>(.*?)</priority>'
    priority_match = re.search(priority_regex, llm_output, re.DOTALL)
    
    priority = "P4"  # 默认兜底
    if priority_match and priority_match.group(1):
        priority = priority_match.group(1).strip()
    
    # 提取正文内容（包含换行符）
    content_regex = r'<content>([\s\S]*?)</content>'
    content_match = re.search(content_regex, llm_output, re.DOTALL)
    
    final_content = ""
    if content_match and content_match.group(1):
        final_content = content_match.group(1).strip()
        # 清理可能残留的内部 XML 杂质（如误生成的 <Title>, <Brief> 等）
        final_content = re.sub(r'<Title>|</Title>|<Brief>|</Brief>', "", final_content)
    else:
        # 兜底逻辑：如果没匹配到 content 标签，尝试移除 thinking 后返回
        final_content = re.sub(r'<thinking>[\s\S]*?</thinking>', "", llm_output)
        final_content = re.sub(r'<final_result>|</final_result>', "", final_content)
        final_content = final_content.strip()
    
    return priority, final_content


def process_single_row(args: tuple) -> tuple:
    """处理单行数据"""
    idx, row, llm_client = args
    
    try:
        # 获取输入文本
        input_text = str(row[config.INPUT_COLUMN])
        
        # 格式化用户提示词
        user_prompt = config.USER_PROMPT_TEMPLATE.format(input_text=input_text)
        
        # 调用 LLM
        full_response = llm_client.generate(
            system_prompt=config.SYSTEM_PROMPT,
            user_prompt=user_prompt
        )
        
        # 提取改写结果
        priority, final_content = extract_rewrite_result(full_response)
        
        # 返回：索引, 完整输出, 优先级, 最终内容, 错误
        return (idx, full_response, priority, final_content, None)
    
    except Exception as e:
        return (idx, None, "P4", "", str(e))


def validate_csv_columns(df: pd.DataFrame, required_columns: list) -> None:
    """验证 CSV 是否包含必需的列"""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV 文件缺少必需的列: {', '.join(missing_columns)}\n"
                        f"当前 CSV 文件的列: {', '.join(df.columns.tolist())}\n"
                        f"请确保 CSV 文件的第一列是 '{required_columns[0]}'")
    
    # 检查是否有空行
    empty_rows = df[df[required_columns[0]].isna() | (df[required_columns[0]].astype(str).str.strip() == "")]
    if len(empty_rows) > 0:
        print(f"警告: 发现 {len(empty_rows)} 行空数据，这些行将被跳过")


def batch_generate(input_csv: str, output_csv: str) -> None:
    """
    批量生成响应
    
    Args:
        input_csv: 输入 CSV 文件路径
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
    
    # 验证列
    validate_csv_columns(df, [config.INPUT_COLUMN])
    
    # 初始化 LLM 客户端
    print(f"正在初始化 {config.API_PROVIDER.upper()} 客户端...")
    try:
        llm_client = LLMClient()
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)
    
    # 初始化输出列
    if config.OUTPUT_COLUMN not in df.columns:
        df[config.OUTPUT_COLUMN] = None
    
    # 初始化提取结果列
    if "eval_priority" not in df.columns:
        df["eval_priority"] = None
    if "final_content" not in df.columns:
        df["final_content"] = None
    
    # 准备数据
    total_rows = len(df)
    print(f"总共需要处理 {total_rows} 行数据")
    
    # 使用线程池并发处理
    results = {}  # {idx: (full_response, priority, final_content)}
    errors = {}
    
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_single_row, (idx, row, llm_client)): idx
            for idx, row in df.iterrows()
        }
        
        # 使用 tqdm 显示进度
        with tqdm(total=total_rows, desc="生成响应") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    row_idx, full_response, priority, final_content, error = future.result()
                    if error:
                        errors[row_idx] = error
                        results[row_idx] = ("error", "P4", "")
                    else:
                        results[row_idx] = (full_response, priority, final_content)
                except Exception as e:
                    errors[idx] = str(e)
                    results[idx] = ("error", "P4", "")
                finally:
                    pbar.update(1)
    
    # 更新 DataFrame
    for idx, result in results.items():
        full_response, priority, final_content = result
        if full_response == "error":
            df.at[idx, config.OUTPUT_COLUMN] = f"error: {errors.get(idx, '未知错误')}"
            df.at[idx, "eval_priority"] = "P4"
            df.at[idx, "final_content"] = ""
        else:
            # 保存完整输出（包含思考过程）
            df.at[idx, config.OUTPUT_COLUMN] = full_response
            # 保存提取的优先级
            df.at[idx, "eval_priority"] = priority
            # 保存提取的最终内容
            df.at[idx, "final_content"] = final_content
    
    # 保存结果
    print(f"正在保存结果到: {output_csv}")
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    # 统计信息
    success_count = sum(1 for v in results.values() if v[0] != "error")
    error_count = total_rows - success_count
    print(f"\n处理完成!")
    print(f"成功: {success_count} 行")
    print(f"错误: {error_count} 行")
    
    if errors:
        print("\n错误详情:")
        for idx, error_msg in errors.items():
            print(f"  行 {idx}: {error_msg}")


def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("用法: python batch_generator.py <输入CSV文件> <输出CSV文件>")
        print("示例: python batch_generator.py input.csv output.csv")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    batch_generate(input_csv, output_csv)


if __name__ == "__main__":
    main()
