# 贡献指南

## 欢迎贡献！

感谢您对 Uber 票价预测系统项目的关注。我们欢迎各种形式的贡献，包括但不限于：

- 🐛 报告问题和 bug
- 💡 提出新功能建议
- 📝 改进文档
- 🔧 提交代码改进
- ✨ 添加新的示例和教程

---

## 如何贡献

### 报告问题

如果您发现了 bug 或有功能建议，请：

1. 首先检查现有的 issues，确保问题尚未被报告
2. 创建新的 issue，使用清晰的标题和描述
3. 包含以下信息：
   - 问题的详细描述
   - 重现步骤
   - 预期行为
   - 实际行为
   - R 版本和操作系统信息
   - 相关的代码片段或错误消息

**示例**:
```
标题: 加载大型数据集时出现内存错误

描述:
当尝试加载超过 100MB 的 Excel 文件时，程序抛出内存错误。

环境:
- R 版本: 4.2.0
- 操作系统: Windows 10
- 内存: 8GB

重现步骤:
1. 准备一个 150MB 的 Excel 文件
2. 运行 data <- read_excel("large_file.xlsx")
3. 观察到错误: "Error: cannot allocate vector of size XX MB"

预期: 应该能够加载文件或提供有用的错误提示
实际: 抛出内存错误并终止
```

---

### 改进文档

文档贡献非常重要！如果您想改进文档：

#### 小改动（拼写、语法、格式）

1. 直接编辑相关的 Markdown 文件
2. 确保格式正确
3. 提交更改并说明原因

#### 大改动（新章节、重构）

1. 首先创建 issue 讨论您的想法
2. 获得反馈后再开始工作
3. 保持一致的格式和风格
4. 更新相关的交叉引用

#### 文档风格指南

- **语言**: 使用简体中文
- **语气**: 友好、专业、清晰
- **代码示例**: 
  - 必须可运行
  - 包含注释
  - 显示预期输出
- **标题**: 使用层次化的标题（H1-H4）
- **列表**: 使用项目符号或编号
- **强调**: 使用粗体或代码块

**文档模板**:
```markdown
# 功能名称

## 概述
简短描述（1-2 句话）

## 语法
\`\`\`r
function_name(param1, param2)
\`\`\`

## 参数
- `param1` (类型): 描述
- `param2` (类型): 描述

## 返回值
描述返回值

## 示例
\`\`\`r
# 示例代码
result <- function_name(value1, value2)
print(result)
\`\`\`

## 注意事项
任何重要的注意事项或限制
```

---

### 代码贡献

#### 开始之前

1. Fork 项目仓库
2. 创建新的分支（`git checkout -b feature/amazing-feature`）
3. 确保您的 R 环境配置正确

#### 代码风格指南

**命名约定**:
```r
# 变量和函数: snake_case
my_variable <- 10
calculate_fare_amount <- function(distance) { }

# 常量: UPPER_SNAKE_CASE
MAX_FARE <- 500
MIN_DISTANCE <- 0

# 类和对象: PascalCase
FareModel <- R6Class("FareModel", ...)
```

**注释**:
```r
# 单行注释使用 #

# 函数注释应该包括：
# 功能描述
# @param 参数名 参数描述
# @return 返回值描述
# @examples 使用示例
calculate_fare <- function(distance, base_rate) {
  # 实现逻辑
  return(fare)
}
```

**代码组织**:
```r
# 1. 加载库
library(dplyr)
library(ggplot2)

# 2. 定义常量
THRESHOLD <- 0.05

# 3. 定义函数
helper_function <- function() { }
main_function <- function() { }

# 4. 主要逻辑
# ... 代码 ...
```

#### 编写测试

虽然当前项目没有正式的测试框架，但建议：

1. 手动测试您的代码
2. 使用不同的输入测试边界情况
3. 验证输出的正确性
4. 在注释中记录测试场景

**测试示例**:
```r
# 测试场景 1: 正常输入
data <- data.frame(distance = c(1, 2, 3), fare = c(5, 10, 15))
result <- my_function(data)
# 预期: 成功返回处理后的数据

# 测试场景 2: 空数据
empty_data <- data.frame()
result <- my_function(empty_data)
# 预期: 返回空数据框或适当的错误

# 测试场景 3: 缺失值
data_with_na <- data.frame(distance = c(1, NA, 3), fare = c(5, 10, 15))
result <- my_function(data_with_na)
# 预期: 正确处理缺失值
```

#### 性能考虑

- 使用向量化操作而不是循环
- 避免不必要的数据复制
- 对大型数据集进行采样测试
- 使用 `system.time()` 测量性能

```r
# 好的实践
result <- data %>% filter(distance > 0) %>% summarise(mean_fare = mean(fare))

# 避免
result <- data.frame()
for (i in 1:nrow(data)) {
  if (data[i, "distance"] > 0) {
    result <- rbind(result, data[i, ])
  }
}
```

---

### 添加新功能

#### 新功能检查清单

在添加新功能之前，请确保：

- [ ] 功能符合项目目标
- [ ] 没有重复现有功能
- [ ] 考虑了性能影响
- [ ] 遵循代码风格指南
- [ ] 添加了充分的注释
- [ ] 创建了使用示例
- [ ] 更新了相关文档

#### 新功能模板

```r
#' 功能名称
#'
#' 详细描述功能做什么，为什么需要它
#'
#' @param param1 参数1的描述
#' @param param2 参数2的描述，默认值: value
#' @return 返回值的描述
#' @examples
#' # 示例1
#' result <- new_function(data, param = "value")
#' 
#' # 示例2
#' result <- new_function(data)
#' @export
new_function <- function(param1, param2 = "default") {
  # 输入验证
  if (missing(param1)) {
    stop("param1 is required")
  }
  
  # 主要逻辑
  result <- process_data(param1, param2)
  
  # 返回结果
  return(result)
}
```

---

### 文档贡献工作流

1. **确定需要改进的文档**
   - 查看现有文档
   - 识别不清楚或缺失的部分

2. **进行更改**
   - 编辑相关的 Markdown 文件
   - 保持格式一致性
   - 添加必要的示例

3. **更新索引**
   - 如果添加了新章节，更新 [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)
   - 更新交叉引用

4. **验证更改**
   - 检查所有链接是否正常工作
   - 确保代码示例可运行
   - 检查拼写和语法

5. **提交更改**
   - 使用清晰的提交消息
   - 说明更改的原因和内容

---

## 项目结构

了解项目结构有助于贡献：

```
/workspace
├── DTS206TC_CW_2144212.R      # 主要的 R 脚本
├── sample_uber.xlsx           # 示例数据
├── README.md                  # 项目概览
├── DOCUMENTATION_INDEX.md     # 文档索引
├── API_DOCUMENTATION.md       # API 参考
├── FUNCTION_LIBRARY.md        # 函数库
├── USER_GUIDE.md              # 用户指南
└── CONTRIBUTING.md            # 本文件
```

### 文档组织

- **README.md**: 项目介绍，第一印象
- **DOCUMENTATION_INDEX.md**: 所有文档的导航中心
- **API_DOCUMENTATION.md**: 技术参考，适合开发者
- **FUNCTION_LIBRARY.md**: 可重用函数，适合高级用户
- **USER_GUIDE.md**: 教程和示例，适合所有用户

---

## 提交指南

### Git 提交消息格式

使用清晰、描述性的提交消息：

```
<类型>: <简短描述>

<详细描述（可选）>

<相关 issue（可选）>
```

**类型**:
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 格式调整（不影响代码功能）
- `refactor`: 代码重构
- `test`: 添加测试
- `chore`: 构建或辅助工具的变动

**示例**:
```
feat: 添加地理聚类可视化函数

实现了新的函数 plot_clusters()，可以可视化 K-means 聚类结果。
包括上车和下车位置的聚类展示。

相关 issue #123
```

### Pull Request 流程

1. **创建 PR**
   - 提供清晰的标题和描述
   - 说明更改的动机和内容
   - 列出相关的 issues

2. **PR 描述模板**
   ```markdown
   ## 更改类型
   - [ ] Bug 修复
   - [ ] 新功能
   - [ ] 文档更新
   - [ ] 性能改进
   - [ ] 代码重构
   
   ## 描述
   简要描述此 PR 的目的和内容
   
   ## 相关 Issue
   关闭 #issue_number
   
   ## 测试
   描述如何测试这些更改
   
   ## 检查清单
   - [ ] 代码遵循项目风格指南
   - [ ] 添加了必要的注释
   - [ ] 更新了相关文档
   - [ ] 所有测试通过
   - [ ] 没有引入新的警告
   ```

3. **代码审查**
   - 耐心等待审查
   - 积极响应反馈
   - 根据建议进行修改

4. **合并后**
   - 删除您的功能分支
   - 更新本地仓库

---

## 开发环境设置

### 必需软件

- R (>= 4.0.0)
- RStudio (推荐)
- Git

### 安装依赖

```r
# 安装所有必需的包
required_packages <- c(
  "readxl", "ggplot2", "dplyr", "lubridate",
  "reshape2", "car", "lmtest", "nortest"
)

install.packages(required_packages)
```

### 推荐的 RStudio 设置

1. **代码**: Tools → Global Options → Code
   - 启用自动缩进
   - 使用 2 个空格缩进
   - 自动保存工作空间: Never

2. **外观**: Tools → Global Options → Appearance
   - 选择您喜欢的主题
   - 合适的字体大小

3. **包**: Tools → Global Options → Packages
   - 选择 CRAN 镜像

---

## 社区准则

### 行为准则

- 尊重所有贡献者
- 欢迎新手提问
- 建设性地提供反馈
- 专注于问题，而非个人
- 保持专业和友好

### 沟通渠道

- **Issues**: 用于 bug 报告和功能请求
- **Pull Requests**: 用于代码审查和讨论
- **文档**: 用于使用指南和参考

---

## 常见贡献场景

### 场景 1: 修复文档中的拼写错误

```bash
# 1. 编辑文件
# 2. 提交更改
git add DOCUMENTATION.md
git commit -m "docs: 修复拼写错误"
git push origin main
```

### 场景 2: 添加新的示例

1. 在相关的文档中添加示例
2. 确保代码可运行
3. 添加注释和说明
4. 更新目录（如果需要）

### 场景 3: 报告 Bug

1. 创建新的 issue
2. 使用 "Bug" 标签
3. 提供详细信息和重现步骤
4. 附上代码片段或截图

### 场景 4: 提出新功能

1. 创建新的 issue
2. 使用 "Enhancement" 标签
3. 清楚地描述功能
4. 解释为什么需要这个功能
5. 讨论可能的实现方法

---

## 资源

### 学习资源

- **R 语言**: https://www.r-project.org/
- **dplyr**: https://dplyr.tidyverse.org/
- **ggplot2**: https://ggplot2.tidyverse.org/
- **Markdown**: https://www.markdownguide.org/

### 项目资源

- [README](README.md) - 项目概述
- [文档索引](DOCUMENTATION_INDEX.md) - 导航
- [API 文档](API_DOCUMENTATION.md) - 技术参考
- [用户指南](USER_GUIDE.md) - 使用教程

---

## 问题和帮助

如果您在贡献过程中遇到问题：

1. 查看现有的文档和指南
2. 搜索相关的 issues
3. 创建新的 issue 寻求帮助
4. 在 issue 中提供详细的上下文

---

## 致谢

感谢所有为这个项目做出贡献的人！您的时间和努力使这个项目变得更好。

特别感谢：
- 报告 bug 的用户
- 改进文档的贡献者
- 提出建议的社区成员
- 审查代码的开发者

---

## 许可证

通过贡献本项目，您同意您的贡献将根据项目的许可证进行授权。

---

**最后更新**: 2025-11-21  
**版本**: 1.0

如有任何疑问或建议，请随时创建 issue 或联系项目维护者。

再次感谢您的贡献！🎉
