# Uber 票价预测系统 - API 和函数文档

## 目录
1. [概述](#概述)
2. [环境配置](#环境配置)
3. [数据处理 API](#数据处理-api)
4. [探索性数据分析 (EDA) 函数](#探索性数据分析-eda-函数)
5. [特征工程](#特征工程)
6. [回归模型 API](#回归模型-api)
7. [模型诊断](#模型诊断)
8. [完整使用示例](#完整使用示例)

---

## 概述

本项目提供了一套完整的 R 语言统计分析工具，用于 Uber 票价预测和分析。通过多元线性回归模型，结合特征工程和严格的诊断测试，实现对票价的准确预测（R² = 0.7646）。

### 核心功能
- 数据加载和预处理
- 多维度探索性数据分析
- 智能特征工程（聚类、交互项、时间特征）
- 多元线性回归建模
- 逐步回归优化
- 全面的模型诊断和修正

---

## 环境配置

### 依赖包安装

```r
# 安装所需包
install.packages("readxl")     # Excel 文件读取
install.packages("ggplot2")    # 数据可视化
install.packages("dplyr")      # 数据处理
install.packages("lubridate")  # 日期时间处理
install.packages("reshape2")   # 数据重塑
install.packages("car")        # 回归诊断
install.packages("lmtest")     # 线性模型测试
install.packages("nortest")    # 正态性测试
```

### 加载库

```r
library(readxl)
library(ggplot2)
library(dplyr)
library(lubridate)
library(reshape2)
library(car)
library(lmtest)
library(nortest)
```

---

## 数据处理 API

### 1. 数据加载函数

#### `read_excel(file_path)`

**功能**: 从 Excel 文件加载 Uber 行程数据

**参数**:
- `file_path` (字符串): Excel 文件的路径

**返回值**: 数据框 (data.frame) 包含所有行程记录

**使用示例**:
```r
# 设置随机种子以确保可重复性
set.seed(123)

# 加载数据集
file_path <- "sample_uber.xlsx"
data <- read_excel(file_path)

# 查看数据结构
head(data)
summary(data)
```

**输出字段**:
- `pickup_datetime`: 上车时间
- `pickup_longitude`: 上车经度
- `pickup_latitude`: 上车纬度
- `dropoff_longitude`: 下车经度
- `dropoff_latitude`: 下车纬度
- `passenger_count`: 乘客数量
- `distance`: 行程距离
- `fare_amount`: 票价金额
- `Code`: 代码（将被移除）
- `key`: 键值（将被移除）

---

### 2. 数据预处理管道

#### 时间特征提取

**功能**: 从 `pickup_datetime` 提取年、月、日、小时信息

**使用示例**:
```r
data <- data %>%
  mutate(
    pickup_datetime = ymd_hms(pickup_datetime),  # 转换为日期时间格式
    year = year(pickup_datetime),                # 提取年份
    month = month(pickup_datetime),              # 提取月份
    day = day(pickup_datetime),                  # 提取日期
    hour = hour(pickup_datetime)                 # 提取小时
  )
```

**生成的新列**:
- `year`: 年份 (整数)
- `month`: 月份 (1-12)
- `day`: 日期 (1-31)
- `hour`: 小时 (0-23)

#### 移除冗余列

```r
# 移除不需要的列
data <- subset(data, select = -c(Code, key))
```

---

## 探索性数据分析 (EDA) 函数

### 1. 相关性热图

#### `generate_correlation_heatmap(data)`

**功能**: 生成数值变量之间的相关性热图

**实现**:
```r
# 计算相关矩阵
cor_matrix <- cor(data %>% select_if(is.numeric), use = "complete.obs")
melted_cor_matrix <- melt(cor_matrix)

# 绘制热图
ggplot(data = melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       midpoint = 0, limit = c(-1, 1), space = "Lab",
                       name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1)) +
  coord_fixed() +
  labs(title = "Correlation Heatmap", x = "", y = "")
```

**输出**: 相关性热图，显示变量间的线性关系强度

---

### 2. 票价分布分析

#### `plot_fare_distribution(data)`

**功能**: 绘制票价金额的密度分布图

**实现**:
```r
ggplot(data, aes(x = fare_amount)) +
  geom_density(fill = "orange", alpha = 0.5) +  
  labs(title = "Density Plot of Trip Fare", x = "Fares", y = "Density") +
  theme_minimal()
```

**用途**: 
- 识别票价分布的偏斜度
- 检测异常值
- 了解票价的集中趋势

---

### 3. 距离分布分析

#### `plot_distance_distribution(data)`

**功能**: 绘制行程距离的密度分布图

**实现**:
```r
ggplot(data, aes(x = distance)) +
  geom_density(fill = "orange", alpha = 0.5) +  
  labs(title = "Distance Distribution", x = "Distance", y = "Frequency") +
  theme_minimal()
```

---

### 4. 票价与距离关系分析

#### `plot_fare_vs_distance(data)`

**功能**: 绘制票价与距离的散点图

**实现**:
```r
ggplot(data, aes(x = distance, y = fare_amount)) +
  geom_point(alpha = 0.5, color = "blue") +
  labs(title = "Scatterplot of Fare Amount vs Distance", 
       x = "Distance", 
       y = "Fare Amount ($)") +
  theme_minimal()
```

**预期结果**: 应该显示正相关关系

---

### 5. 地理位置分析

#### `plot_pickup_locations(data)`

**功能**: 可视化上车位置及其票价分布

**实现**:
```r
ggplot(data, aes(x = pickup_longitude, y = pickup_latitude, color = fare_amount)) +
  geom_point(alpha = 0.6) +
  scale_color_gradient(low = "blue", high = "red") +
  labs(title = "Scatterplot of Fare Amount by Pickup Longitude and Latitude", 
       x = "Pickup Longitude", 
       y = "Pickup Latitude", 
       color = "Fare Amount ($)") +
  theme_minimal() +
  coord_cartesian(xlim = c(min(data$pickup_longitude), max(data$pickup_longitude)),
                  ylim = c(min(data$pickup_latitude), max(data$pickup_latitude)))
```

#### `plot_dropoff_locations(data)`

**功能**: 可视化下车位置及其票价分布

**实现**:
```r
ggplot(data, aes(x = dropoff_longitude, y = dropoff_latitude, color = fare_amount)) +
  geom_point(alpha = 0.6) +
  scale_color_gradient(low = "blue", high = "red") +
  labs(title = "Scatterplot of Fare Amount by Dropoff Longitude and Latitude", 
       x = "Dropoff Longitude", 
       y = "Dropoff Latitude", 
       color = "Fare Amount ($)") +
  theme_minimal()
```

---

## 特征工程

### 1. K-means 地理聚类

#### `create_location_clusters(data, n_clusters = 5)`

**功能**: 使用 K-means 算法对上车和下车位置进行聚类

**参数**:
- `data`: 数据框
- `n_clusters`: 聚类数量（默认为 5）

**实现**:
```r
# 准备地理坐标数据
pickup_data <- data.frame(data$pickup_longitude, data$pickup_latitude)
dropoff_data <- data.frame(data$dropoff_longitude, data$dropoff_latitude)

# 执行 K-means 聚类
pickup_clusters <- kmeans(pickup_data, centers = 5)$cluster
dropoff_clusters <- kmeans(dropoff_data, centers = 5)$cluster

# 将聚类结果添加到数据集
data <- data %>%
  mutate(
    dropoff_cluster = dropoff_clusters,
    pickup_cluster = pickup_clusters
  )
```

**返回值**: 包含 `pickup_cluster` 和 `dropoff_cluster` 列的数据框

**可视化聚类结果**:
```r
# 上车位置聚类可视化
ggplot(data, aes(x = pickup_longitude, y = pickup_latitude, color = factor(pickup_cluster))) +
  geom_point(alpha = 0.5) +
  labs(title = "Pickup Locations Clustering", 
       x = "Pickup Longitude", 
       y = "Pickup Latitude", 
       color = "Pickup Cluster") +
  theme_minimal()

# 下车位置聚类可视化
ggplot(data, aes(x = dropoff_longitude, y = dropoff_latitude, color = factor(dropoff_cluster))) +
  geom_point(alpha = 0.5) +
  labs(title = "Dropoff Locations Clustering", 
       x = "Dropoff Longitude", 
       y = "Dropoff Latitude", 
       color = "Dropoff Cluster") +
  theme_minimal()
```

---

### 2. 白天/夜间特征

#### `create_daytime_feature(data)`

**功能**: 创建白天/夜间二元特征（6:00-20:00 为白天）

**实现**:
```r
data <- data %>%
  mutate(
    daytime = ifelse(hour >= 6 & hour < 20, 1, 0)
  )
```

**输出**:
- `daytime = 1`: 白天（6:00 AM - 8:00 PM）
- `daytime = 0`: 夜间（8:00 PM - 6:00 AM）

**分析白天/夜间的票价差异**:
```r
ggplot(data, aes(x = factor(daytime), y = fare_amount, fill = factor(daytime))) +
  geom_bar(stat = "summary", fun = "mean", color = "black", alpha = 0.7) +
  labs(title = "Average Fare Amount: Night vs Day", 
       x = "Daytime (1: Day, 0: Night)", 
       y = "Average Fare Amount ($)", 
       fill = "Time of Day") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c("orange", "blue"), 
                    labels = c("Night", "Day"))
```

---

### 3. 交互项特征

#### `create_interaction_features(data)`

**功能**: 创建距离与白天时间的交互项

**实现**:
```r
data <- data %>%
  mutate(
    distance_daytime_interaction = distance * daytime
  )
```

**用途**: 捕捉距离对票价的影响在白天和夜间的不同效应

---

### 4. 时间分析可视化

#### 年度票价分析
```r
data$year <- as.factor(data$year)
ggplot(data, aes(x = year, y = fare_amount)) +
  geom_boxplot(alpha = 0.7, fill = "blue", color = "black") +
  labs(title = "Boxplot of Fare Amount vs Year", 
       x = "Year", 
       y = "Fare Amount ($)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

#### 月度票价分析
```r
data$month <- factor(data$month, levels = 1:12)
ggplot(data, aes(x = month, y = fare_amount)) +
  geom_boxplot(alpha = 0.7, fill = "blue", color = "black") +
  labs(title = "Boxplot of Fare Amount vs Month", 
       x = "Month", 
       y = "Fare Amount ($)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

---

### 5. 乘客数量分析

#### `analyze_passenger_count(data)`

**功能**: 分析乘客数量对平均票价的影响

**实现**:
```r
ggplot(data, aes(x = factor(passenger_count), y = fare_amount, fill = factor(passenger_count))) +
  geom_bar(stat = "summary", fun = "mean", color = "black", alpha = 0.7) +
  labs(title = "Average Fare Amount vs Passenger Count", 
       x = "Passenger Count", 
       y = "Average Fare Amount ($)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_brewer(palette = "Blues")
```

---

## 回归模型 API

### 1. 基础线性回归模型

#### `build_linear_model(data)`

**功能**: 构建包含所有特征的多元线性回归模型

**参数**:
- `data`: 包含所有特征的数据框

**实现**:
```r
model <- lm(fare_amount ~ pickup_longitude + pickup_latitude + 
              dropoff_longitude + dropoff_latitude + distance + passenger_count +
              distance_daytime_interaction + 
              year + month + day + hour + 
              dropoff_clusters + pickup_clusters, 
            data = data)

# 查看模型摘要
summary(model)
```

**返回值**: 线性模型对象 (lm)

**模型公式**:
```
fare_amount = β0 + β1·pickup_longitude + β2·pickup_latitude + 
              β3·dropoff_longitude + β4·dropoff_latitude + 
              β5·distance + β6·passenger_count + 
              β7·distance_daytime_interaction + 
              β8·year + β9·month + β10·day + β11·hour + 
              β12·dropoff_cluster + β13·pickup_cluster + ε
```

---

### 2. 逐步回归优化

#### `optimize_model_stepwise(model, direction = "backward")`

**功能**: 使用逐步回归方法优化模型

**参数**:
- `model`: 初始线性模型对象
- `direction`: 选择方向（"backward", "forward", "both"）

**实现**:
```r
# 向后逐步回归
stepwise_model <- step(model, direction = "backward")

# 查看优化后的模型
summary(stepwise_model)
```

**返回值**: 优化后的线性模型对象

**提取系数和 p 值**:
```r
# 提取系数
coefficients <- coef(stepwise_model)
print("Coefficients:")
print(coefficients)

# 提取 p 值
p_values <- summary(stepwise_model)$coefficients[, 4]
print("P-values:")
print(p_values)
```

---

### 3. 模型性能评估

#### `evaluate_model_performance(model)`

**功能**: 评估模型的拟合优度

**实现**:
```r
# 计算 R² 和调整 R²
r_squared <- summary(stepwise_model)$r.squared
adjusted_r_squared <- summary(stepwise_model)$adj.r.squared

print(paste("R-squared:", r_squared))
print(paste("Adjusted R-squared:", adjusted_r_squared))
```

**输出指标**:
- `R-squared`: 决定系数（0-1），表示模型解释的方差比例
- `Adjusted R-squared`: 调整决定系数，考虑了变量数量的影响

---

## 模型诊断

### 1. 综合诊断图

#### `plot_model_diagnostics(model)`

**功能**: 生成四个标准诊断图

**实现**:
```r
par(mfrow = c(2, 2))
plot(stepwise_model)
```

**生成的图表**:
1. **残差 vs 拟合值**: 检查线性关系和同方差性
2. **Q-Q 图**: 检查残差正态性
3. **标准化残差**: 检查同方差性
4. **残差 vs 杠杆值**: 识别高影响点

---

### 2. 线性关系检验

#### `check_linearity(model)`

**功能**: 检查残差 vs 拟合值的线性关系

**实现**:
```r
plot(stepwise_model, 1)
```

**评估标准**:
- 残差应该随机分布在 0 线周围
- 不应该有明显的模式或曲线

---

### 3. 同方差性检验

#### `breusch_pagan_test(model)`

**功能**: 使用 Breusch-Pagan 检验检测异方差性

**实现**:
```r
bptest(stepwise_model)
```

**假设**:
- H0: 同方差性（方差恒定）
- H1: 异方差性（方差不恒定）

**判断标准**:
- p 值 > 0.05: 接受 H0，同方差性假设成立
- p 值 ≤ 0.05: 拒绝 H0，存在异方差性

**可视化检查**:
```r
plot(stepwise_model, 3)  # Scale-Location 图
```

---

### 4. 独立性检验

#### `durbin_watson_test(model)`

**功能**: 使用 Durbin-Watson 检验检测残差自相关

**实现**:
```r
durbinWatsonTest(stepwise_model)
```

**DW 统计量解释**:
- DW ≈ 2: 无自相关
- DW < 2: 正自相关
- DW > 2: 负自相关

**判断标准**:
- p 值 > 0.05: 无显著自相关
- p 值 ≤ 0.05: 存在自相关

---

### 5. 正态性检验

#### `check_normality(model)`

**功能**: 检查残差的正态性

**Q-Q 图**:
```r
plot(stepwise_model, 2)
```

**Anderson-Darling 检验**:
```r
ad.test(residuals(stepwise_model))
```

**假设**:
- H0: 残差服从正态分布
- H1: 残差不服从正态分布

**判断标准**:
- p 值 > 0.05: 接受 H0，正态性假设成立
- p 值 ≤ 0.05: 拒绝 H0，残差不服从正态分布

---

### 6. 多重共线性检验

#### `check_multicollinearity(model)`

**功能**: 使用方差膨胀因子 (VIF) 检测多重共线性

**实现**:
```r
library(car)
vif_values <- vif(stepwise_model)
print(vif_values)
```

**VIF 判断标准**:
- VIF < 5: 无显著多重共线性
- 5 ≤ VIF < 10: 中度多重共线性
- VIF ≥ 10: 严重多重共线性，需要处理

---

### 7. 影响点分析

#### `identify_influential_points(model)`

**功能**: 识别高影响点（使用 Cook's 距离）

**实现**:
```r
# 查看影响点图
plot(stepwise_model, 4)

# 计算 Cook's 距离
cooksd <- cooks.distance(model)

# 设置阈值
threshold <- 20 / nrow(data)

# 识别高影响点
high_influence_points <- which(cooksd > threshold)

print(paste("发现", length(high_influence_points), "个高影响点"))
```

**Cook's 距离判断标准**:
- D > 4/n: 潜在影响点
- D > 1: 高影响点

---

## 修正措施

### 1. 移除高影响点

#### `remove_influential_points(data, model, threshold = 20)`

**功能**: 移除 Cook's 距离超过阈值的数据点

**实现**:
```r
# 计算 Cook's 距离
cooksd <- cooks.distance(model)

# 设置阈值
threshold <- 20 / nrow(data)

# 识别高影响点
high_influence_points <- which(cooksd > threshold)

# 创建新数据集（不包含高影响点）
data_clean <- data[-high_influence_points, ]
```

**返回值**: 清理后的数据框

---

### 2. 变量转换

#### `apply_transformations(data)`

**功能**: 对响应变量和预测变量应用对数和平方根转换

**实现**:
```r
# 对票价进行对数转换（处理右偏分布）
data$log_fare_amount <- log(data$fare_amount)

# 对距离进行平方根转换（稳定方差）
data$sqrt_distance <- sqrt(data$distance)
```

**用途**:
- 对数转换: 处理右偏分布，稳定方差
- 平方根转换: 减少异常值的影响

---

### 3. 重新拟合优化模型

#### `refit_transformed_model(data)`

**功能**: 使用转换后的变量和清理后的数据重新拟合模型

**实现**:
```r
# 应用变量转换
data_without_high_influence_points$log_fare_amount <- log(data_without_high_influence_points$fare_amount)
data_without_high_influence_points$sqrt_distance <- sqrt(data_without_high_influence_points$distance)

# 重新拟合模型
model_transformed <- lm(log_fare_amount ~ pickup_longitude +  
                          dropoff_longitude + sqrt_distance + distance_daytime_interaction + 
                          year + month + pickup_cluster, 
                        data = data_without_high_influence_points)

# 查看新模型摘要
summary(model_transformed)
```

**验证改进**:
```r
# 诊断图
par(mfrow = c(2, 2))
plot(model_transformed)
plot(model_transformed, which = 2)  # Q-Q 图
plot(model_transformed, which = 5)  # Scale-Location 图
```

---

## 完整使用示例

### 示例 1: 完整的数据分析流程

```r
# ========================================
# 第一步: 环境设置
# ========================================
library(readxl)
library(ggplot2)
library(dplyr)
library(lubridate)
library(reshape2)
library(car)
library(lmtest)
library(nortest)

set.seed(123)

# ========================================
# 第二步: 数据加载和预处理
# ========================================
file_path <- "sample_uber.xlsx"
data <- read_excel(file_path)

# 提取时间特征
data <- data %>%
  mutate(
    pickup_datetime = ymd_hms(pickup_datetime),
    year = year(pickup_datetime),
    month = month(pickup_datetime),
    day = day(pickup_datetime),
    hour = hour(pickup_datetime)
  )

# 移除冗余列
data <- subset(data, select = -c(Code, key))

# 查看数据
head(data)
summary(data)

# ========================================
# 第三步: 探索性数据分析
# ========================================

# 相关性分析
cor_matrix <- cor(data %>% select_if(is.numeric), use = "complete.obs")
melted_cor_matrix <- melt(cor_matrix)

ggplot(data = melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       midpoint = 0, limit = c(-1, 1), space = "Lab",
                       name="Correlation") +
  theme_minimal() +
  labs(title = "Correlation Heatmap")

# 票价分布
ggplot(data, aes(x = fare_amount)) +
  geom_density(fill = "orange", alpha = 0.5) +
  labs(title = "Density Plot of Trip Fare")

# 距离分布
ggplot(data, aes(x = distance)) +
  geom_density(fill = "orange", alpha = 0.5) +
  labs(title = "Distance Distribution")

# ========================================
# 第四步: 特征工程
# ========================================

# 地理聚类
pickup_data <- data.frame(data$pickup_longitude, data$pickup_latitude)
dropoff_data <- data.frame(data$dropoff_longitude, data$dropoff_latitude)

pickup_clusters <- kmeans(pickup_data, centers = 5)$cluster
dropoff_clusters <- kmeans(dropoff_data, centers = 5)$cluster

data <- data %>%
  mutate(
    dropoff_cluster = dropoff_clusters,
    pickup_cluster = pickup_clusters,
    daytime = ifelse(hour >= 6 & hour < 20, 1, 0),
    distance_daytime_interaction = distance * daytime
  )

# ========================================
# 第五步: 建立回归模型
# ========================================

# 初始模型
model <- lm(fare_amount ~ pickup_longitude + pickup_latitude + 
              dropoff_longitude + dropoff_latitude + distance + passenger_count +
              distance_daytime_interaction + 
              year + month + day + hour + 
              dropoff_clusters + pickup_clusters, 
            data = data)

summary(model)

# 逐步回归优化
stepwise_model <- step(model, direction = "backward")
summary(stepwise_model)

# 评估模型性能
r_squared <- summary(stepwise_model)$r.squared
adjusted_r_squared <- summary(stepwise_model)$adj.r.squared
print(paste("R-squared:", r_squared))
print(paste("Adjusted R-squared:", adjusted_r_squared))

# ========================================
# 第六步: 模型诊断
# ========================================

# 综合诊断图
par(mfrow = c(2, 2))
plot(stepwise_model)

# 同方差性检验
bptest(stepwise_model)

# 独立性检验
durbinWatsonTest(stepwise_model)

# 正态性检验
ad.test(residuals(stepwise_model))

# 多重共线性检验
vif_values <- vif(stepwise_model)
print(vif_values)

# ========================================
# 第七步: 修正措施
# ========================================

# 识别并移除高影响点
cooksd <- cooks.distance(model)
threshold <- 20 / nrow(data)
high_influence_points <- which(cooksd > threshold)
data_clean <- data[-high_influence_points, ]

# 应用变量转换
data_clean$log_fare_amount <- log(data_clean$fare_amount)
data_clean$sqrt_distance <- sqrt(data_clean$distance)

# 重新拟合模型
model_transformed <- lm(log_fare_amount ~ pickup_longitude +  
                          dropoff_longitude + sqrt_distance + distance_daytime_interaction + 
                          year + month + pickup_cluster, 
                        data = data_clean)

summary(model_transformed)

# 验证改进
par(mfrow = c(2, 2))
plot(model_transformed)
```

---

### 示例 2: 预测新数据

```r
# 准备新的行程数据
new_trip <- data.frame(
  pickup_longitude = -73.98,
  pickup_latitude = 40.75,
  dropoff_longitude = -73.95,
  dropoff_latitude = 40.78,
  distance = 2.5,
  passenger_count = 1,
  year = 2015,
  month = 6,
  hour = 14,
  daytime = 1,
  distance_daytime_interaction = 2.5 * 1,
  pickup_cluster = 3,
  dropoff_cluster = 2
)

# 使用逐步回归模型进行预测
predicted_fare <- predict(stepwise_model, newdata = new_trip)
print(paste("预测票价:", round(predicted_fare, 2), "美元"))

# 使用转换后的模型进行预测
new_trip$sqrt_distance <- sqrt(new_trip$distance)
predicted_log_fare <- predict(model_transformed, newdata = new_trip)
predicted_fare_transformed <- exp(predicted_log_fare)  # 反向转换
print(paste("转换模型预测票价:", round(predicted_fare_transformed, 2), "美元"))
```

---

### 示例 3: 批量预测和性能评估

```r
# 划分训练集和测试集
set.seed(123)
train_indices <- sample(1:nrow(data), 0.8 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# 在训练集上建立模型
model_train <- lm(fare_amount ~ pickup_longitude + pickup_latitude + 
                    dropoff_longitude + dropoff_latitude + distance + passenger_count +
                    distance_daytime_interaction + 
                    year + month + day + hour + 
                    dropoff_clusters + pickup_clusters, 
                  data = train_data)

# 在测试集上进行预测
predictions <- predict(model_train, newdata = test_data)

# 计算预测性能指标
mae <- mean(abs(predictions - test_data$fare_amount))
rmse <- sqrt(mean((predictions - test_data$fare_amount)^2))
mape <- mean(abs((predictions - test_data$fare_amount) / test_data$fare_amount)) * 100

print(paste("平均绝对误差 (MAE):", round(mae, 2)))
print(paste("均方根误差 (RMSE):", round(rmse, 2)))
print(paste("平均绝对百分比误差 (MAPE):", round(mape, 2), "%"))

# 可视化预测结果
results <- data.frame(
  Actual = test_data$fare_amount,
  Predicted = predictions
)

ggplot(results, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "实际票价 vs 预测票价", 
       x = "实际票价 ($)", 
       y = "预测票价 ($)") +
  theme_minimal()
```

---

## 性能指标说明

### 模型拟合指标

| 指标 | 含义 | 理想值 |
|------|------|--------|
| **R²** | 模型解释的方差比例 | 接近 1 |
| **Adjusted R²** | 调整后的 R²（考虑变量数量） | 接近 1 |
| **F-statistic** | 模型整体显著性 | p-value < 0.05 |
| **Residual Std Error** | 残差标准误差 | 越小越好 |

### 预测性能指标

| 指标 | 公式 | 含义 |
|------|------|------|
| **MAE** | Σ\|y - ŷ\| / n | 平均绝对误差 |
| **RMSE** | √(Σ(y - ŷ)² / n) | 均方根误差 |
| **MAPE** | Σ\|y - ŷ\| / y × 100 / n | 平均绝对百分比误差 |

---

## 常见问题和解决方案

### 问题 1: 异方差性

**症状**: Breusch-Pagan 检验 p 值 < 0.05

**解决方案**:
1. 对响应变量进行对数转换
2. 对预测变量进行平方根或对数转换
3. 使用加权最小二乘法 (WLS)

```r
# 对数转换
data$log_fare <- log(data$fare_amount)
model_log <- lm(log_fare ~ ., data = data)
```

---

### 问题 2: 非正态残差

**症状**: Q-Q 图显示偏离正态线，Anderson-Darling 检验 p < 0.05

**解决方案**:
1. 对响应变量进行 Box-Cox 转换
2. 移除极端异常值
3. 使用稳健回归方法

```r
# Box-Cox 转换
library(MASS)
bc <- boxcox(model)
lambda <- bc$x[which.max(bc$y)]
data$transformed_fare <- (data$fare_amount^lambda - 1) / lambda
```

---

### 问题 3: 多重共线性

**症状**: VIF > 10

**解决方案**:
1. 移除高度相关的变量
2. 使用主成分分析 (PCA)
3. 使用岭回归或 Lasso 回归

```r
# 移除高 VIF 变量
vif_values <- vif(model)
high_vif_vars <- names(vif_values[vif_values > 10])
formula_new <- as.formula(paste("fare_amount ~", 
                                paste(setdiff(names(data), c("fare_amount", high_vif_vars)), 
                                      collapse = " + ")))
model_reduced <- lm(formula_new, data = data)
```

---

### 问题 4: 自相关

**症状**: Durbin-Watson 统计量显著偏离 2

**解决方案**:
1. 添加滞后变量
2. 使用时间序列模型 (ARIMA)
3. 使用广义最小二乘法 (GLS)

---

## 最佳实践

### 1. 数据预处理
- ✅ 始终设置随机种子以确保可重复性
- ✅ 检查并处理缺失值
- ✅ 识别并处理异常值
- ✅ 标准化或归一化数值变量（如果必要）

### 2. 特征工程
- ✅ 创建领域知识驱动的特征（如交互项）
- ✅ 使用聚类方法提取地理信息
- ✅ 提取时间序列特征（年、月、日、小时等）

### 3. 模型构建
- ✅ 从简单模型开始，逐步增加复杂度
- ✅ 使用逐步回归或其他特征选择方法
- ✅ 交叉验证模型性能

### 4. 模型诊断
- ✅ 检查所有回归假设
- ✅ 识别并处理影响点
- ✅ 验证模型在测试集上的表现

### 5. 结果解释
- ✅ 解释系数的实际意义
- ✅ 评估预测的不确定性
- ✅ 可视化关键发现

---

## 参考资料

### R 包文档
- **readxl**: https://readxl.tidyverse.org/
- **ggplot2**: https://ggplot2.tidyverse.org/
- **dplyr**: https://dplyr.tidyverse.org/
- **car**: https://cran.r-project.org/web/packages/car/
- **lmtest**: https://cran.r-project.org/web/packages/lmtest/

### 统计方法
- 多元线性回归理论
- 逐步回归方法
- 回归诊断技术
- K-means 聚类算法

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| 1.0 | 2025-11-21 | 初始版本，包含完整的 API 文档 |

---

## 联系和支持

如有问题或建议，请参考项目的 README.md 文件或提交 issue。

---

**文档生成日期**: 2025-11-21  
**项目**: Uber 票价预测系统  
**作者**: DTS206TC_CW_2144212
