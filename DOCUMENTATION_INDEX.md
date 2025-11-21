# Uber 票价预测系统 - 文档索引

## 欢迎使用 Uber 票价预测系统文档

本项目提供了一套完整的统计分析工具，用于 Uber 票价预测和数据分析。以下是完整的文档索引，帮助您快速找到所需信息。

---

## 📚 文档导航

### 1. 项目概览
- **[README.md](README.md)** - 项目简介和主要特性
  - 项目背景和目标
  - 关键功能和技术亮点
  - 项目结果和应用价值
  - 技术栈说明

### 2. API 和函数文档
- **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** - 完整的 API 参考文档
  - 环境配置和依赖安装
  - 数据处理 API
  - 探索性数据分析函数
  - 特征工程方法
  - 回归模型 API
  - 模型诊断工具
  - 完整使用示例

### 3. 函数库参考
- **[FUNCTION_LIBRARY.md](FUNCTION_LIBRARY.md)** - 可重用函数库
  - 数据处理函数
  - 可视化函数集合
  - 特征工程函数
  - 建模和预测函数
  - 诊断检验函数
  - 实用工具函数
  - 端到端工作流程示例

### 4. 用户使用指南
- **[USER_GUIDE.md](USER_GUIDE.md)** - 详细的使用教程
  - 快速开始（5分钟入门）
  - 安装和配置指南
  - 数据准备和质量检查
  - 基础使用教程（4个教程）
  - 高级使用技巧（5个技巧）
  - 常见场景示例（5个场景）
  - 故障排除指南
  - 性能优化建议
  - 常见问题解答

### 5. 源代码
- **[DTS206TC_CW_2144212.R](DTS206TC_CW_2144212.R)** - 完整的 R 脚本
  - 数据加载和预处理
  - 探索性数据分析代码
  - 特征工程实现
  - 模型构建和优化
  - 诊断测试代码
  - 修正措施实现

### 6. 数据文件
- **sample_uber.xlsx** - 示例数据集
  - 包含真实的 Uber 行程数据
  - 用于训练和测试模型

---

## 🚀 快速导航

### 我想要...

#### 快速开始使用系统
→ 阅读 [USER_GUIDE.md - 快速开始](USER_GUIDE.md#快速开始)

#### 了解所有可用的函数和 API
→ 查看 [API_DOCUMENTATION.md](API_DOCUMENTATION.md)

#### 学习如何使用每个函数
→ 参考 [FUNCTION_LIBRARY.md](FUNCTION_LIBRARY.md)

#### 看实际应用示例
→ 浏览 [USER_GUIDE.md - 常见场景示例](USER_GUIDE.md#常见场景示例)

#### 解决遇到的问题
→ 查阅 [USER_GUIDE.md - 故障排除](USER_GUIDE.md#故障排除)

#### 了解项目背景和研究成果
→ 阅读 [README.md](README.md)

#### 查看完整的源代码
→ 打开 [DTS206TC_CW_2144212.R](DTS206TC_CW_2144212.R)

---

## 📖 按任务类型查找

### 数据准备
| 任务 | 文档位置 |
|------|---------|
| 加载 Excel 数据 | [API_DOCUMENTATION.md - 数据加载函数](API_DOCUMENTATION.md#1-数据加载函数) |
| 数据清理 | [FUNCTION_LIBRARY.md - clean_data](FUNCTION_LIBRARY.md#clean_datadata-remove_na--true-remove_outliers--false) |
| 提取时间特征 | [FUNCTION_LIBRARY.md - add_time_features](FUNCTION_LIBRARY.md#add_time_featuresdata) |
| 数据质量检查 | [USER_GUIDE.md - 数据质量检查](USER_GUIDE.md#数据质量检查) |

### 数据探索
| 任务 | 文档位置 |
|------|---------|
| 相关性分析 | [API_DOCUMENTATION.md - 相关性热图](API_DOCUMENTATION.md#1-相关性热图) |
| 分布分析 | [FUNCTION_LIBRARY.md - plot_distribution](FUNCTION_LIBRARY.md#plot_distributiondata-variable-title--null-fill_color--orange) |
| 时间序列分析 | [USER_GUIDE.md - 时间分析](USER_GUIDE.md#可视化-3-时间分析) |
| 地理分析 | [USER_GUIDE.md - 地理分布](USER_GUIDE.md#可视化-4-地理分布) |

### 特征工程
| 任务 | 文档位置 |
|------|---------|
| 创建地理聚类 | [API_DOCUMENTATION.md - K-means 地理聚类](API_DOCUMENTATION.md#1-k-means-地理聚类) |
| 创建交互项 | [FUNCTION_LIBRARY.md - add_interaction_terms](FUNCTION_LIBRARY.md#add_interaction_termsdata-var1-var2-interaction_name--null) |
| 变量转换 | [FUNCTION_LIBRARY.md - apply_transformations](FUNCTION_LIBRARY.md#apply_transformationsdata-log_vars--null-sqrt_vars--null) |
| 白天/夜间特征 | [API_DOCUMENTATION.md - 白天/夜间特征](API_DOCUMENTATION.md#2-白天夜间特征) |

### 模型构建
| 任务 | 文档位置 |
|------|---------|
| 建立线性模型 | [FUNCTION_LIBRARY.md - build_linear_model](FUNCTION_LIBRARY.md#build_linear_modeldata-formula-summary_output--true) |
| 逐步回归 | [API_DOCUMENTATION.md - 逐步回归优化](API_DOCUMENTATION.md#2-逐步回归优化) |
| 模型评估 | [FUNCTION_LIBRARY.md - evaluate_model](FUNCTION_LIBRARY.md#evaluate_modelmodel-test_data--null) |
| 进行预测 | [USER_GUIDE.md - 进行预测](USER_GUIDE.md#步骤-4-进行预测) |

### 模型诊断
| 任务 | 文档位置 |
|------|---------|
| 综合诊断 | [FUNCTION_LIBRARY.md - perform_comprehensive_diagnostics](FUNCTION_LIBRARY.md#perform_comprehensive_diagnosticsmodel-data) |
| 同方差性检验 | [API_DOCUMENTATION.md - 同方差性检验](API_DOCUMENTATION.md#3-同方差性检验) |
| 正态性检验 | [FUNCTION_LIBRARY.md - test_normality](FUNCTION_LIBRARY.md#test_normalitymodel) |
| 多重共线性检验 | [API_DOCUMENTATION.md - 多重共线性检验](API_DOCUMENTATION.md#6-多重共线性检验) |
| 识别影响点 | [FUNCTION_LIBRARY.md - identify_influential_points](FUNCTION_LIBRARY.md#identify_influential_pointsmodel-data-threshold_multiplier--20) |

### 问题解决
| 问题类型 | 文档位置 |
|---------|---------|
| 安装问题 | [USER_GUIDE.md - 问题2: 包加载失败](USER_GUIDE.md#问题-2-包加载失败) |
| 数据加载问题 | [USER_GUIDE.md - 问题1: 无法加载Excel文件](USER_GUIDE.md#问题-1-无法加载-excel-文件) |
| 内存问题 | [USER_GUIDE.md - 问题3: 内存不足](USER_GUIDE.md#问题-3-内存不足) |
| 模型警告 | [USER_GUIDE.md - 问题4: 模型拟合警告](USER_GUIDE.md#问题-4-模型拟合警告) |
| 性能优化 | [USER_GUIDE.md - 性能优化建议](USER_GUIDE.md#性能优化建议) |

---

## 🎯 按用户类型推荐

### 初学者路径

1. **第一步**: 阅读 [README.md](README.md) 了解项目背景
2. **第二步**: 按照 [USER_GUIDE.md - 快速开始](USER_GUIDE.md#快速开始) 运行第一个示例
3. **第三步**: 学习 [USER_GUIDE.md - 基础使用教程](USER_GUIDE.md#基础使用教程) 的所有教程
4. **第四步**: 查看 [USER_GUIDE.md - 常见场景示例](USER_GUIDE.md#常见场景示例) 了解实际应用

### 中级用户路径

1. **探索**: [USER_GUIDE.md - 高级使用技巧](USER_GUIDE.md#高级使用技巧)
2. **深入**: [FUNCTION_LIBRARY.md](FUNCTION_LIBRARY.md) 学习所有可用函数
3. **实践**: [API_DOCUMENTATION.md - 完整使用示例](API_DOCUMENTATION.md#完整使用示例)
4. **优化**: [USER_GUIDE.md - 性能优化建议](USER_GUIDE.md#性能优化建议)

### 高级用户路径

1. **API 参考**: [API_DOCUMENTATION.md](API_DOCUMENTATION.md) 完整 API 文档
2. **源代码**: [DTS206TC_CW_2144212.R](DTS206TC_CW_2144212.R) 研究实现细节
3. **自定义**: 基于 [FUNCTION_LIBRARY.md](FUNCTION_LIBRARY.md) 创建自己的函数
4. **扩展**: 根据 [API_DOCUMENTATION.md - 修正措施](API_DOCUMENTATION.md#修正措施) 优化模型

### 数据科学家路径

1. **统计方法**: [API_DOCUMENTATION.md - 模型诊断](API_DOCUMENTATION.md#模型诊断)
2. **特征工程**: [API_DOCUMENTATION.md - 特征工程](API_DOCUMENTATION.md#特征工程)
3. **模型优化**: [API_DOCUMENTATION.md - 修正措施](API_DOCUMENTATION.md#修正措施)
4. **性能评估**: [FUNCTION_LIBRARY.md - 建模函数](FUNCTION_LIBRARY.md#建模函数)

---

## 📊 功能覆盖矩阵

| 功能类别 | API 文档 | 函数库 | 用户指南 | 源代码 |
|---------|---------|--------|---------|--------|
| **数据加载** | ✅ | ✅ | ✅ | ✅ |
| **数据清理** | ✅ | ✅ | ✅ | ✅ |
| **EDA** | ✅ | ✅ | ✅ | ✅ |
| **可视化** | ✅ | ✅ | ✅ | ✅ |
| **特征工程** | ✅ | ✅ | ✅ | ✅ |
| **模型构建** | ✅ | ✅ | ✅ | ✅ |
| **模型评估** | ✅ | ✅ | ✅ | ✅ |
| **模型诊断** | ✅ | ✅ | ✅ | ✅ |
| **预测** | ✅ | ✅ | ✅ | ✅ |
| **故障排除** | ❌ | ❌ | ✅ | ❌ |
| **示例代码** | ✅ | ✅ | ✅ | ✅ |

---

## 🔧 技术规格

### 支持的 R 版本
- R 4.0.0 或更高

### 必需的 R 包
```r
readxl      # 1.3.1 或更高
ggplot2     # 3.3.0 或更高
dplyr       # 1.0.0 或更高
lubridate   # 1.7.0 或更高
reshape2    # 1.4.4 或更高
car         # 3.0.0 或更高
lmtest      # 0.9.38 或更高
nortest     # 1.0.4 或更高
```

### 数据要求
- 格式: Excel (.xlsx)
- 最小行数: 100
- 必需列: 8 个（见用户指南）
- 推荐行数: 10,000+

---

## 📈 文档统计

| 文档 | 页数估计 | 字数估计 | 代码示例 | 难度 |
|------|---------|---------|---------|------|
| README.md | 3 | 800 | 0 | ⭐ |
| API_DOCUMENTATION.md | 40+ | 12,000+ | 100+ | ⭐⭐⭐ |
| FUNCTION_LIBRARY.md | 35+ | 10,000+ | 80+ | ⭐⭐⭐⭐ |
| USER_GUIDE.md | 45+ | 15,000+ | 120+ | ⭐⭐ |
| 总计 | 120+ | 37,000+ | 300+ | - |

---

## 🎓 学习路径建议

### 路径 1: 快速上手（1-2 小时）
```
README.md 
  → USER_GUIDE.md (快速开始) 
  → USER_GUIDE.md (教程 1-2)
  → 运行第一个模型
```

### 路径 2: 全面学习（1-2 天）
```
README.md 
  → USER_GUIDE.md (完整阅读) 
  → API_DOCUMENTATION.md (浏览主要部分)
  → FUNCTION_LIBRARY.md (选择性阅读)
  → 实践所有教程
```

### 路径 3: 深入掌握（3-5 天）
```
README.md 
  → 所有文档完整阅读
  → 源代码分析
  → 实践所有示例
  → 自定义扩展开发
```

---

## 📝 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| 1.0 | 2025-11-21 | 初始版本发布 |
|     |            | - 完整的 API 文档 |
|     |            | - 函数库参考 |
|     |            | - 用户使用指南 |
|     |            | - 文档索引 |

---

## 🤝 贡献指南

如果您想改进文档或添加新功能：

1. 确保理解现有文档结构
2. 遵循现有的格式和风格
3. 添加充分的示例和说明
4. 更新相关的索引和交叉引用

---

## 📧 获取帮助

如果您在使用过程中遇到问题：

1. **首先**: 检查 [USER_GUIDE.md - 故障排除](USER_GUIDE.md#故障排除)
2. **其次**: 查阅 [USER_GUIDE.md - 常见问题解答](USER_GUIDE.md#常见问题解答)
3. **然后**: 搜索相关文档中的关键词
4. **最后**: 查看源代码中的注释

---

## ⚖️ 许可证

本项目用于学术研究目的。请参考项目根目录中的许可证文件。

---

## 🌟 致谢

感谢所有为这个项目做出贡献的人员和使用的开源 R 包的开发者。

---

**文档版本**: 1.0  
**最后更新**: 2025-11-21  
**维护者**: DTS206TC_CW_2144212

---

## 快速链接

- [返回主 README](README.md)
- [查看 API 文档](API_DOCUMENTATION.md)
- [查看函数库](FUNCTION_LIBRARY.md)
- [查看用户指南](USER_GUIDE.md)
- [查看源代码](DTS206TC_CW_2144212.R)
