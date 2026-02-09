"""
Script 3: 评估员 Prompt 生成器
根据用户提供的应用场景和北极星指标，调用 LLM 生成评估员的 Prompt。
"""

import os
import sys
import json
from dotenv import load_dotenv

# 导入配置
import config

# 导入工具类
from utils import LLMClient

# 加载环境变量
load_dotenv()


# 生成评估员 Prompt 的系统提示词
PROMPT_GENERATOR_SYSTEM_PROMPT = """# Role
你是一位精通 LLM Evals (大模型评估) 的架构师。你的专长是将抽象的业务目标（North Star Metrics）转化为可执行、量化、严谨的**评估员 Prompt (Evaluator Prompt)**。

# Task
用户将提供两个信息：
1.  **应用场景 (Scenario)**：被评估的内容是什么（例如：小红书文案、Python 代码、客服回复）。
2.  **北极星指标 (North Star Metric)**：用户最看重的核心价值（例如：幽默感、代码安全性、极度共情）。

你需要生成一个完整的、结构化的 System Prompt，该 Prompt 将被用于配置一个 AI 评委（Evaluator Model）。

# Constraint & Requirements (生成规范)
你生成的 Evaluator Prompt 必须严格包含以下模块：

1.  **Role Definition**: 根据应用场景，定义评委的角色（例如：资深主编、安全审计员）。
2.  **Inputs**: 定义输入变量，必须包含 `<original_input>` 和 `<model_output>`。
3.  **Scoring Criteria (评分标准)**:
    * **必须包含 [Factuality/Safety] (权重 30%-40%)**: 这是护栏指标，检查幻觉、有害信息。对应 JSON 键名建议 `factuality_safety_score`。
    * **必须包含 [North Star Metric] (权重 30%-40%)**: **北极星指标的描述必须完全按照用户输入的北极星指标来写**（不擅自替换为其他表述）。对应 JSON 中 **必须使用键名 `north_star_score`**，不得使用其他键名（如 attractiveness_score、quality_score 等）。
    * **必须包含 [Completeness/Coherence] (权重 20%)**: 基础质量指标。对应 JSON 键名建议 `completeness_coherence_score`。
    * *注意：总权重必须等于 100%。*
4.  **Scoring Scale (评分档次与差异度)**:
    * 定义 0-100 分制，且**必须明确写出各分数段对应的表现**（例如：90-100 卓越 / 80-89 良好 / 70-79 合格 / 60-69 有待改进 / 0-59 不合格），并给出每档的典型特征。
    * **必须要求评委严格区分档次**：鼓励使用 90+ / 80-89 / 70-79 等区间内的**细致分数**（如 87、82、76），**避免大量样本打出同一分数**；在 Prompt 中明确写出「请根据实际表现拉开分差，不要扎堆打高分」或类似约束。
5.  **Output Format (JSON Only)**:
    * 强制评委输出 JSON 格式。
    * JSON 结构必须包含：`determined_priority` (字符串, P0/P1/P2/P3), `scores` (对象), `weighted_total_score` (数值, 0-100), `reasoning` (字符串), `pass` (布尔值)。
    * `scores` 对象**必须采用扁平格式**，且**北极星维度键名必须为 `north_star_score`**。示例：`{"factuality_safety_score": 90, "north_star_score": 85, "completeness_coherence_score": 88}` （0-100 分制）。
    * `weighted_total_score` 计算公式：各维度分数（0-100）× 对应权重（0-1），总和为 0-100 分制。
    * 决策阈值：`weighted_total_score >= 75` 发布，`< 75` 人工复核，事实性相关分数 < 50 直接拒绝。

# Output Format
请直接输出完整的 Evaluator Prompt，不要包含额外的说明文字。Prompt 应该可以直接用于配置评估器。"""


def generate_evaluator_prompt(scenario: str, north_star_metric: str) -> str:
    """
    根据场景和北极星指标生成评估员 Prompt
    
    Args:
        scenario: 应用场景描述
        north_star_metric: 北极星指标描述
    
    Returns:
        生成的评估员 Prompt
    """
    # 初始化 LLM 客户端
    print(f"正在初始化 {config.API_PROVIDER.upper()} 客户端...")
    try:
        llm_client = LLMClient()
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)
    
    # 构建用户提示词
    user_prompt = f"""场景：{scenario}
北极星指标：{north_star_metric}

请根据以上信息生成完整的评估员 Prompt。"""
    
    print("正在生成评估员 Prompt...")
    print(f"场景: {scenario}")
    print(f"北极星指标: {north_star_metric}")
    print("-" * 60)
    
    # 调用 LLM 生成 Prompt
    generated_prompt = llm_client.generate(
        system_prompt=PROMPT_GENERATOR_SYSTEM_PROMPT,
        user_prompt=user_prompt
    )
    
    return generated_prompt


def save_prompt_to_file(prompt: str, output_file: str = "generated_evaluator_prompt.txt"):
    """将生成的 Prompt 保存到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(prompt)
    print(f"\n生成的 Prompt 已保存到: {output_file}")


def update_config_file(prompt: str, config_file: str = "config.py"):
    """
    将生成的 Prompt 更新到 config.py 文件
    
    Args:
        prompt: 生成的评估员 Prompt
        config_file: 配置文件路径
    """
    try:
        # 读取现有配置文件
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找 EVALUATION_PROMPT 的位置
        start_marker = 'EVALUATION_PROMPT = """'
        end_marker = '"""'
        
        start_idx = content.find(start_marker)
        if start_idx == -1:
            print("警告: 未找到 EVALUATION_PROMPT 配置项，将保存到单独文件")
            return False
        
        # 找到结束位置（查找下一个 """）
        start_idx += len(start_marker)
        end_idx = content.find(end_marker, start_idx)
        if end_idx == -1:
            print("警告: 未找到 EVALUATION_PROMPT 的结束标记，将保存到单独文件")
            return False
        
        # 替换 Prompt
        new_content = (
            content[:start_idx] + 
            prompt + 
            content[end_idx:]
        )
        
        # 保存更新后的配置文件
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"\n已更新 {config_file} 中的 EVALUATION_PROMPT")
        return True
        
    except Exception as e:
        print(f"更新配置文件时出错: {str(e)}")
        return False


def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("用法: python generate_evaluator_prompt.py <应用场景> <北极星指标> [输出文件]")
        print("\n示例:")
        print('  python generate_evaluator_prompt.py "小红书文案生成" "幽默感和传播力"')
        print('  python generate_evaluator_prompt.py "医疗咨询回复" "极度的同理心和安抚能力" evaluator_prompt.txt')
        print("\n参数说明:")
        print("  <应用场景>: 被评估内容的场景描述（例如：小红书文案、Python代码、客服回复）")
        print("  <北极星指标>: 用户最看重的核心价值（例如：幽默感、代码安全性、极度共情）")
        print("  [输出文件]: 可选，生成的 Prompt 保存路径（默认: generated_evaluator_prompt.txt）")
        sys.exit(1)
    
    scenario = sys.argv[1]
    north_star_metric = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "generated_evaluator_prompt.txt"
    
    # 生成评估员 Prompt
    generated_prompt = generate_evaluator_prompt(scenario, north_star_metric)
    
    # 显示生成的 Prompt
    print("\n" + "=" * 60)
    print("生成的评估员 Prompt:")
    print("=" * 60)
    print(generated_prompt)
    print("=" * 60)
    
    # 保存到文件
    save_prompt_to_file(generated_prompt, output_file)
    
    # 询问是否更新 config.py
    print("\n是否要将生成的 Prompt 更新到 config.py? (y/n): ", end='')
    try:
        choice = input().strip().lower()
        if choice == 'y' or choice == 'yes':
            if update_config_file(generated_prompt):
                print("✅ 配置文件已更新！")
            else:
                print("⚠️  配置文件更新失败，Prompt 已保存到单独文件")
        else:
            print("已跳过配置文件更新")
    except:
        print("已跳过配置文件更新（非交互模式）")


if __name__ == "__main__":
    main()
