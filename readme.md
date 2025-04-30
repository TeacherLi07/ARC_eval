# ARC 评估框架

## 简介

本仓库提供了一个用于在抽象与推理语料库 (Abstraction and Reasoning Corpus, ARC) 上评估人工智能 (AI) 模型的框架。ARC 是一个旨在测试系统执行抽象推理和解决问题能力的基准，类似于人类的流体智力。该框架有助于运行评估实验、管理结果以及与 ARC 任务进行交互。它包含了来自 ARC-AGI-1 和 ARC-AGI-2 的数据。

参考：
*   [ARC-AGI-1](https://github.com/fchollet/arc-agi)
*   [ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2)
*   [On the Measure of Intelligence](https://arxiv.org/abs/1911.01547)

## 特性

*   使用各种大语言模型 (LLM) 通过 API（例如 SiliconFlow, OpenAI）运行评估的脚本。
*   用于处理 API 调用、处理响应和评估结果的核心模块。
*   数据库集成 (SQLite) 用于存储和管理评估运行及结果。
*   包含 ARC-AGI-1 和 ARC-AGI-2 数据集。
*   一个基于 Web 的测试界面（来自 ARC-AGI-1），用于可视化任务和手动测试解决方案。

## 目录结构

*   `code/`: 包含评估框架的核心 Python 代码。
    *   `core/`: 评估逻辑、API 处理、数据库工具等核心模块。
    *   `experiments/`: 用于运行特定评估实验的示例脚本。
*   `data/`: 包含 ARC 数据集。
    *   `ARC-AGI-1/`: ARC-AGI v1 的数据和测试界面。
    *   `ARC-AGI-2/`: ARC-AGI v2 的数据。
*   `results/`: 存储评估结果（SQLite 数据库）的目录。
*   `readme.md`: 本文件。

## 使用方法

1.  **设置:**
    *   克隆本仓库。
    *   安装必要的 Python 依赖项: `pip install -r requirements.txt`
    *   设置所需的环境变量，例如 API 密钥 (`SF_API_KEY` 等)。
2.  **运行评估:**
    *   导航到 `code/experiments/` 目录。
    *   修改或创建一个实验脚本 (例如 `simple_exp_glm.py`, `exp_openai.py`) 来配置模型、API 端点、提示和其他参数。
    *   执行脚本: `python code/experiments/your_experiment_script.py`
    *   结果将保存到 `results/` 目录中指定的 SQLite 数据库。
3.  **浏览数据/界面:**
    *   浏览 `data/` 目录以查找 ARC 任务文件 (.json)。
    *   在 Web 浏览器中打开 `data/ARC-AGI-1/apps/testing_interface.html` 以手动与 ARC 任务交互。

## 依赖项

*   Python 3.x
*   `requests`: 用于进行 HTTP API 调用。
*   `openai`: 用于与 OpenAI SDK 兼容的 API 进行交互。

