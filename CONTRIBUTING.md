# 贡献指南 / Contributing to AsrServe

感谢您对 AsrServe 的贡献！本指南将帮助您设置开发环境并了解我们的贡献流程。

Thank you for your interest in contributing to AsrServe! This guide will help you set up your development environment and understand our contribution process.

## 目录 / Table of Contents
- [开发环境 / Development Environment](#开发环境--development-environment)
- [代码风格 / Code Style](#代码风格--code-style)
- [测试 / Testing](#测试--testing)
- [分支策略 / Branching Strategy](#分支策略--branching-strategy)
- [Pull Request 流程 / Pull Request Process](#pull-request-流程--pull-request-process)
- [Issue 指南 / Issue Guidelines](#issue-指南--issue-guidelines)

## 开发环境 / Development Environment

### 先决条件 / Prerequisites
- Python 3.11 或 3.12 / Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/)（包管理器）/ [uv](https://docs.astral.sh/uv/) (package manager)
- Rust 工具链（CPU 后端）/ Rust toolchain (for CPU backend)
- Docker（容器构建）/ Docker (for container builds)
- FFmpeg、libsndfile、OpenBLAS（系统依赖）/ FFmpeg, libsndfile, OpenBLAS (system dependencies)

### 设置 / Setup

```bash
# 克隆仓库 / Clone the repository
git clone https://github.com/your-username/AsrServe.git
cd AsrServe

# 安装依赖 / Install dependencies
uv sync --frozen

# 构建 Rust CPU 后端（如需要）/ Build the Rust CPU backend (if needed)
./scripts/build-rust.sh

# 运行应用 / Run the application
uv run python start.py
```

### 开发依赖 / Development Dependencies

```bash
# 安装开发工具 / Install development tools
uv pip install ruff ty pytest pytest-cov httpx
```

## 代码风格 / Code Style

我们使用以下工具来维护代码质量：

We use the following tools to maintain code quality:

- **ruff** - 快速 Python 代码检查和格式化工具 / Fast Python linter and formatter
- **ty** - 快速类型检查器 / Fast type checker

### 运行检查 / Running Checks

```bash
# 代码检查 / Lint
ruff check app/ tests/

# 格式化 / Format
ruff format app/ tests/

# 类型检查 / Type check
ty app/
```

### 风格指南 / Style Guidelines
- 遵循 PEP 8 规范 / Follow PEP 8 conventions
- 所有函数签名使用类型提示 / Use type hints for all function signatures
- 公共函数和类编写文档字符串 / Write docstrings for public functions and classes
- 行长度不超过 100 字符 / Keep lines under 100 characters
- 使用 f-string 进行字符串格式化 / Use f-strings for string formatting

## 测试 / Testing

### 运行测试 / Running Tests

```bash
# 运行所有测试 / Run all tests
pytest tests/

# 带覆盖率运行 / Run with coverage
pytest tests/ --cov=app --cov-report=term-missing

# 运行特定测试文件 / Run specific test file
pytest tests/test_api_contract.py

# 运行特定测试 / Run specific test
pytest tests/test_api_contract.py::APIContractTest::test_offline_all_response_formats
```

### 编写测试 / Writing Tests
- 使用 `pytest` 进行测试发现和运行 / Use `pytest` for test discovery and execution
- 使用 `unittest.mock` 进行模拟 / Use `unittest.mock` for mocking
- 测试正常路径和错误条件 / Test both happy paths and error conditions
- 业务逻辑追求高覆盖率 / Aim for high coverage of business logic
- 单元测试中模拟外部依赖（模型、网络）/ Mock external dependencies (models, network) in unit tests

### 覆盖率要求 / Coverage Requirements
- 最低 70% 行覆盖率 / Minimum 70% line coverage
- 新代码不应降低整体覆盖率 / New code should not decrease overall coverage

## 分支策略 / Branching Strategy

所有开发都使用功能分支：

We use feature branches for all development:

```
main ← feat/issue-N-short-description
  ↑        ↑
  |        └── 你的功能分支 / Your feature branch
  └── 上游 main（Quantatirsk/AsrServe）/ Upstream main (Quantatirsk/AsrServe)
```

### 分支命名 / Branch Naming
- `feat/issue-N-description` - 新功能 / New features
- `fix/issue-N-description` - 错误修复 / Bug fixes
- `docs/issue-N-description` - 文档变更 / Documentation changes
- `chore/issue-N-description` - 维护任务 / Maintenance tasks

### 示例 / Example
```bash
git checkout -b feat/issue-1-localization-en
```

## Pull Request 流程 / Pull Request Process

1. **创建功能分支** 从 `main` / **Create a feature branch** from `main`
2. **做出更改** 并添加适当的测试 / **Make your changes** with appropriate tests
3. **运行所有检查**：lint、format、type check、tests / **Run all checks**: lint, format, type check, tests
4. **推送分支** 并创建 Pull Request / **Push your branch** and create a Pull Request
5. **处理审查意见** 并更新 PR / **Address review comments** and update the PR
6. **Squash 并合并** 一旦批准 / **Squash and merge** once approved

### PR 检查清单 / PR Checklist
- [ ] 代码遵循风格指南（ruff check 通过）/ Code follows style guidelines (ruff check passes)
- [ ] 类型检查通过（ty）/ Type checking passes (ty)
- [ ] 所有测试通过（pytest）/ All tests pass (pytest)
- [ ] 覆盖率保持或提高 / Coverage is maintained or improved
- [ ] 文档已更新（如需要）/ Documentation is updated (if needed)
- [ ] PR 描述解释了更改和动机 / PR description explains the changes and motivation

### PR 描述模板 / PR Description Template
```markdown
## 摘要 / Summary
更改的简要描述。/ Brief description of the changes.

## 动机 / Motivation
为什么需要这些更改？如适用，链接到 issue。/ Why are these changes needed? Link to issue if applicable.

## 测试 / Testing
- 测试了什么 / What was tested
- 如何测试的 / How it was tested
- 测试结果 / Test results

## 截图/演示 / Screenshots/Demos
（UI 变更时）/ (For UI changes)

## 检查清单 / Checklist
- [ ] Lint 通过 / Lint passes
- [ ] 类型检查通过 / Type check passes
- [ ] 测试通过 / Tests pass
- [ ] 覆盖率保持 / Coverage maintained
- [ ] 文档已更新 / Documentation updated
```

## Issue 指南 / Issue Guidelines

创建 issue 时，请：

When creating an issue, please:
- 使用清晰、描述性的标题 / Use a clear, descriptive title
- 提供背景和动机 / Provide context and motivation
- 包含复现步骤（对于 bug）/ Include steps to reproduce (for bugs)
- 指定预期行为与实际行为 / Specify expected vs actual behavior
- 添加相关标签 / Add relevant labels

### Bug 报告模板 / Bug Report Template
```markdown
## 描述 / Description
问题是什么？/ What is the issue?

## 复现步骤 / Steps to Reproduce
1. 步骤 1 / Step 1
2. 步骤 2 / Step 2
3. 步骤 3 / Step 3

## 预期行为 / Expected Behavior
应该发生什么？/ What should happen?

## 实际行为 / Actual Behavior
实际发生了什么？/ What actually happened?

## 环境 / Environment
- 操作系统 / OS:
- Python 版本 / Python version:
- AsrServe 版本 / AsrServe version:
- GPU（如适用）/ GPU (if applicable):
```

### 功能请求模板 / Feature Request Template
```markdown
## 摘要 / Summary
功能的简要描述。/ Brief description of the feature.

## 动机 / Motivation
为什么需要这个功能？/ Why is this feature needed?

## 建议实现 / Proposed Implementation
应该如何实现？/ How should this be implemented?

## 考虑过的替代方案 / Alternatives Considered
考虑过哪些其他方法？/ What other approaches were considered?
```

## 发布流程 / Release Process

发布由维护者管理。要请求发布：

Releases are managed by the maintainers. To request a release:
1. 确保所有更改已合并到 `main` / Ensure all changes are merged to `main`
2. 创建包含建议发布版本的 issue / Create an issue with the proposed release version
3. 维护者将创建发布标签并更新 Docker 镜像 / Maintainers will create a release tag and update Docker images

## 沟通 / Communication

- 使用 GitHub Issues 进行 bug 报告和功能请求 / Use GitHub Issues for bug reports and feature requests
- 使用 GitHub Discussions 进行一般性问题 / Use GitHub Discussions for general questions
- 在所有沟通中保持尊重和建设性 / Be respectful and constructive in all communications

## 许可证 / License

通过贡献，您同意您的贡献将按照项目的许可证进行许可。

By contributing, you agree that your contributions will be licensed under the project's license.
