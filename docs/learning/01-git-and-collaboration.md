# Git 与团队协作

## 学习目标

完成本教程后，你应该能够：

- 理解 Git、本地仓库和 GitHub/GitLab 的区别；
- 安全地创建分支、提交、推送和同步代码；
- 通过 Pull Request 与组员协作；
- 看懂 `git status` 和 `git diff`；
- 处理简单冲突和常见误操作；
- 知道哪些文件不能提交到仓库。

## 1. Git 到底解决什么问题

如果三个人通过微信发送“最终版.py”“最终版2.py”“真最终版.py”，很快就会出现：

- 不知道哪个文件最新；
- 不知道某行是谁、为什么修改；
- 两个人同时修改后互相覆盖；
- 代码出错后无法恢复；
- 论文中的结果无法对应到当时的代码。

Git 为每次有意义的修改保存一个带说明的版本。GitHub/GitLab 则负责把仓库放到远程服务器，提供成员权限、Issue 和 Pull Request。

## 2. 四个位置

```text
工作区：你正在编辑的文件
   ↓ git add
暂存区：准备进入下一次提交的修改
   ↓ git commit
本地仓库：保存在本机的版本历史
   ↓ git push
远程仓库：GitHub/GitLab 上的共享版本
```

几个关键结论：

- 保存文件不等于 `commit`；
- `commit` 不等于上传；
- `push` 才会把本地提交上传；
- `pull` 会获取并合并远程修改；
- 分支不是复制整份项目，而是从某个版本开始的一条开发线。

## 3. 第一次配置

安装 Git 后打开 PowerShell 或 Git Bash：

```bash
git --version
git config --global user.name "你的姓名"
git config --global user.email "你的邮箱"
git config --global init.defaultBranch main
git config --global pull.ff only
```

检查配置：

```bash
git config --global --list
```

姓名和邮箱会写入提交记录。建议使用真实姓名和你用于 GitHub/GitLab 的邮箱。

## 4. 获取仓库

负责人在 GitHub/GitLab 创建私有仓库并邀请成员。其他成员执行：

```bash
git clone <仓库地址>
cd <仓库目录>
git status
git remote -v
```

`origin` 通常是远程仓库的默认名称。

## 5. 日常工作流

### 开始任务

先同步 `main`，再创建任务分支：

```bash
git switch main
git pull --ff-only
git switch -c feature/nearest-baseline
```

分支名建议使用：

```text
feature/citygml-parser
feature/data-schema
feature/nearest-baseline
feature/visibility-filter
fix/heading-angle
docs/learning-notes
```

### 查看修改

```bash
git status
git diff
```

养成习惯：每次 `add` 前先看 `status` 和 `diff`，防止把数据、临时文件或密钥一起提交。

### 暂存和提交

优先明确指定文件，而不是盲目使用 `git add .`：

```bash
git add src/baseline.py
git add tests/test_baseline.py
git commit -m "feat: add nearest-building baseline"
```

好的提交应当：

- 只做一件事；
- 修改规模较小；
- 提交信息说明完成了什么；
- 提交后代码仍能运行。

常用提交前缀：

```text
feat: 新功能
fix: 修复错误
test: 添加或修改测试
docs: 文档修改
refactor: 不改变功能的代码整理
```

### 推送分支

第一次推送：

```bash
git push -u origin feature/nearest-baseline
```

后续在同一分支继续提交时，只需：

```bash
git push
```

### 创建 Pull Request

在 GitHub/GitLab 网页中创建 Pull Request/Merge Request，说明：

```text
完成了什么：
如何运行：
如何检查结果：
仍有什么限制：
```

至少让另一名成员检查：

- 是否符合任务要求；
- 是否能在另一台机器运行；
- 是否误提交数据或密钥；
- 结果是否有最小验证。

### 合并以后

```bash
git switch main
git pull --ff-only
git branch -d feature/nearest-baseline
```

## 6. 如何同步组员修改

查看远程更新但暂不合并：

```bash
git fetch origin
```

查看分支图：

```bash
git log --oneline --graph --all
```

自己的功能分支落后于 `main` 时，可以将最新 `main` 合并进来：

```bash
git fetch origin
git merge origin/main
```

对于 Git 初学团队，优先使用普通 merge。暂时不要把复杂 rebase 当作必备技能。

## 7. 合并冲突是什么

当两个人修改了同一文件的同一位置，Git 无法自动判断保留哪一版，会出现：

```text
<<<<<<< HEAD
你的内容
=======
另一边的内容
>>>>>>> origin/main
```

处理步骤：

1. 执行 `git status` 查看冲突文件；
2. 与相关成员确认正确内容；
3. 删除冲突标记，保留最终内容；
4. 运行最小测试；
5. 暂存并提交。

```bash
git add 冲突文件
git commit
```

不要在不理解内容时直接选择“全部接受当前”或“全部接受传入”。

## 8. 常见误操作

取消暂存，但保留工作区修改：

```bash
git restore --staged 文件名
```

丢弃未提交修改：

```bash
git restore 文件名
```

第二条命令会丢失该文件尚未提交的修改，执行前必须确认。

临时保存未完成工作：

```bash
git stash push -m "unfinished work"
git stash list
git stash pop
```

已经推送的错误提交，优先创建一个反向提交：

```bash
git revert <提交编号>
```

初学阶段不要随意使用：

```bash
git reset --hard
git push --force
git clean -fd
```

出现问题时先保存 `git status`、`git log --oneline --graph --all` 的输出，再寻求帮助。

## 9. 哪些文件不能提交

不提交：

- 原始街景照片；
- 大型 CityGML 数据；
- `.pth`、`.ckpt` 等模型权重；
- 实验产生的大量图片；
- `.env`；
- SSH 私钥；
- 服务器密码和云密钥；
- Python 虚拟环境；
- 缓存文件。

项目初期的 `.gitignore` 可以包含：

```gitignore
.venv/
__pycache__/
*.pyc
.env
data/*
outputs/*
checkpoints/*
*.pth
*.pt
*.ckpt
.ipynb_checkpoints/
```

如果需要保留 `data/README.md`，可以增加：

```gitignore
!data/README.md
```

## 10. 项目练习

三个人分别完成：

1. clone 仓库；
2. 创建 `docs/git-practice-姓名.md`；
3. 写入自己的姓名和一句项目理解；
4. 创建个人分支；
5. 提交并推送；
6. 创建 Pull Request；
7. 由另一名成员审查和合并；
8. 本地切回 `main` 并拉取合并结果。

## 验收清单

- [ ] 能解释 `add`、`commit`、`push` 的区别；
- [ ] 能从最新 `main` 创建分支；
- [ ] 提交前会查看 `status` 和 `diff`；
- [ ] 能创建 Pull Request；
- [ ] 能安全取消暂存；
- [ ] 知道哪些命令有数据丢失风险；
- [ ] 知道原始数据和密钥不能提交。

## 官方资料

- [Pro Git 官方中文版](https://git-scm.com/book/zh/v2)
- [GitHub：设置 Git](https://docs.github.com/en/get-started/git-basics/set-up-git)
- [GitHub：Pull Request](https://docs.github.com/en/pull-requests/get-started/about-pull-requests)

