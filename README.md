# LLM 初学者基础知识网站

一个面向初学者的大语言模型（LLM）基础知识学习网站，使用 Hugo 静态网站生成器构建。

## 项目简介

本网站旨在为初学者提供大语言模型相关的基础知识，包括线性代数、概率论、神经网络等核心概念，帮助读者理解LLM的工作原理和数学基础。

## 网站内容

当前网站包含以下主要内容：

- **矩阵基础知识**：介绍矩阵的定义、运算、性质及其在LLM中的应用
- **Softmax函数**：详细讲解Softmax函数的定义、性质、实现和应用
- **更多内容**：将持续更新，涵盖LLM相关的更多基础知识

## 如何本地运行

要在本地运行此网站，您需要安装Hugo静态网站生成器。

### 安装Hugo

请根据您的操作系统安装Hugo：

- **macOS**：`brew install hugo`
- **Linux**：`sudo apt install hugo`
- **Windows**：`choco install hugo-extended`

或者从[Hugo官方网站](https://gohugo.io/getting-started/installing/)下载预编译版本。

### 克隆项目

```bash
git clone https://github.com/akzj/llm-basics-for-beginners.git
cd llm-basics-for-beginners
```

### 安装主题

本项目使用PaperMod主题，已通过git submodule引入：

```bash
git submodule update --init --recursive
```

### 运行本地服务器

```bash
hugo server -D
```

然后在浏览器中访问 `http://localhost:1313` 查看网站。

## 如何部署

网站使用GitHub Pages进行部署，配置了GitHub Actions自动构建和部署。

### 手动部署

如果需要手动部署，可以执行以下命令：

```bash
hugo --gc --minify
git add public
git commit -m "Update site"
git push
```

## 项目结构

```
llm-basics-for-beginners/
├── archetypes/       # 内容模板
├── assets/           # 静态资源
├── content/          # 网站内容
│   ├── _index.md     # 首页
│   ├── about.md      # 关于页面
│   ├── matrix_basics.md  # 矩阵基础知识
│   └── softmax.md    # Softmax函数
├── layouts/          # 页面布局
├── static/           # 静态文件
├── themes/           # 网站主题
├── .gitignore        # Git忽略文件
├── .gitmodules       # Git子模块配置
├── hugo.toml         # Hugo配置
└── README.md         # 项目说明
```

## 如何贡献

欢迎贡献内容或改进网站！

1. Fork本仓库
2. 创建新分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m "Add some feature"`
4. 推送分支：`git push origin feature/your-feature`
5. 创建Pull Request

### 内容贡献指南

- 请使用Markdown格式编写内容
- 数学公式使用LaTeX格式，用`$`包围行内公式，用`$$`包围块级公式
- 确保内容准确、清晰，适合初学者阅读

## 许可证

本项目采用MIT许可证，详情请查看[LICENSE](LICENSE)文件。

## 联系方式

如有问题或建议，欢迎提交Issue或Pull Request。

---

**网站地址**：[https://akzj.github.io/llm-basics-for-beginners/](https://akzj.github.io/llm-basics-for-beginners/)