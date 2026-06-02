# GitHub 上传说明

当前目录已经整理为可上传仓库内容，但本机没有 `gh` 命令，且我不能代你使用账户密码登录 GitHub。

你可以在本机安装并登录 `gh` 后执行：

```bash
git init
git add .
git commit -m "Initial SMGF experiment framework"
gh repo create SMGF --public --source . --remote origin --push
```

如果你希望我继续自动完成这一步，请先在本机完成以下任一条件：

1. 安装 `gh` 并执行 `gh auth login`
2. 或提供一个已登录的 GitHub CLI 环境

然后我就可以继续直接创建 `SMGF` 仓库并推送当前代码与文档。
