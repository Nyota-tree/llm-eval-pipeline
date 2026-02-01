# 上传到 GitHub 的完整步骤

## 前提条件
- 已安装 Git（下载：https://git-scm.com/download/win）
- 或使用 GitHub Desktop（下载：https://desktop.github.com/）

## 方法 1: 使用命令行（需要 Git）

在项目目录 `F:\newsEvals\llm-eval-pipeline` 中执行：

```bash
# 1. 初始化 Git 仓库
git init

# 2. 添加所有文件
git add .

# 3. 创建初始提交
git commit -m "Initial commit: LLM Evals Pipeline for PMs"

# 4. 重命名分支为 main
git branch -M main

# 5. 连接到远程仓库
git remote add origin https://github.com/Nyota-tree/llm-eval-pipeline.git

# 6. 推送到 GitHub
git push -u origin main
```

## 方法 2: 使用 GitHub Desktop（推荐，图形界面）

1. 打开 GitHub Desktop
2. File → Add Local Repository
3. 选择 `F:\newsEvals\llm-eval-pipeline`
4. 点击 "Publish repository"
5. 仓库名：`llm-eval-pipeline`
6. 点击 "Publish Repository"

## 方法 3: 使用 Cursor 内置功能

1. 在 Cursor 中打开项目
2. 点击左侧源代码管理图标（Ctrl+Shift+G）
3. 点击 "Initialize Repository"（如果还没有初始化）
4. 暂存所有文件（点击 "+"）
5. 输入提交信息："Initial commit: LLM Evals Pipeline for PMs"
6. 点击 "Commit"
7. 点击 "..." → "Publish Branch"
8. 选择仓库并发布

## 验证上传

上传成功后，访问：https://github.com/Nyota-tree/llm-eval-pipeline

应该能看到所有项目文件。
