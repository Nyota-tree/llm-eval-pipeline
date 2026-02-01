# 示例文件说明

## input_example.csv

这是一个简单的示例输入文件，展示了 CSV 文件的格式要求。

### 格式要求

1. **必需列名**：第一列必须命名为 `input_text`（或在 `config.py` 中配置的 `INPUT_COLUMN` 值）
2. **编码格式**：支持 UTF-8、GBK 等常见编码
3. **文本格式**：每行一个输入，文本用双引号包裹（CSV 标准格式）

### 使用示例

```bash
# 使用示例文件进行测试
python batch_generator.py examples/input_example.csv output.csv
```

### 自定义输入

你可以创建自己的输入文件，只需确保：
- 第一列名称与 `config.py` 中的 `INPUT_COLUMN` 一致
- 每行包含一条需要处理的数据
- 文件编码为 UTF-8 或 GBK
