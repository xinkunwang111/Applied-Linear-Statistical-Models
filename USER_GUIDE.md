# Uber 票价预测系统 - 用户使用指南

## 目录

1. [快速开始](#快速开始)
2. [安装和配置](#安装和配置)
3. [数据准备](#数据准备)
4. [基础使用教程](#基础使用教程)
5. [高级使用技巧](#高级使用技巧)
6. [常见场景示例](#常见场景示例)
7. [故障排除](#故障排除)
8. [性能优化建议](#性能优化建议)
9. [常见问题解答](#常见问题解答)

---

## 快速开始

### 5 分钟快速入门

如果您想快速了解系统功能，按照以下步骤操作：

```r
# 1. 加载所需库
library(readxl)
library(ggplot2)
library(dplyr)
library(lubridate)
library(reshape2)
library(car)
library(lmtest)
library(nortest)

# 2. 设置随机种子
set.seed(123)

# 3. 加载数据
file_path <- "sample_uber.xlsx"
data <- read_excel(file_path)

# 4. 基础预处理
data <- data %>%
  mutate(
    pickup_datetime = ymd_hms(pickup_datetime),
    year = year(pickup_datetime),
    month = month(pickup_datetime),
    day = day(pickup_datetime),
    hour = hour(pickup_datetime)
  )

# 5. 简单模型
model <- lm(fare_amount ~ distance + passenger_count, data = data)
summary(model)

# 6. 查看结果
cat("R²:", summary(model)$r.squared, "\n")
```

**预期输出**: 您将看到模型摘要，包括 R²、系数和 p 值。

---

## 安装和配置

### 系统要求

- **R 版本**: 4.0.0 或更高
- **操作系统**: Windows, macOS, 或 Linux
- **内存**: 至少 4GB RAM（推荐 8GB）
- **存储空间**: 至少 500MB 可用空间

### 安装依赖包

#### 方法 1: 一次性安装（推荐）

```r
# 定义所需包列表
required_packages <- c(
  "readxl",      # Excel 文件读取
  "ggplot2",     # 数据可视化
  "dplyr",       # 数据处理
  "lubridate",   # 日期时间处理
  "reshape2",    # 数据重塑
  "car",         # 回归诊断
  "lmtest",      # 线性模型测试
  "nortest"      # 正态性测试
)

# 检查并安装缺失的包
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

# 加载所有包
lapply(required_packages, library, character.only = TRUE)

cat("所有依赖包已成功加载！\n")
```

#### 方法 2: 逐个安装

```r
install.packages("readxl")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("lubridate")
install.packages("reshape2")
install.packages("car")
install.packages("lmtest")
install.packages("nortest")
```

### 验证安装

```r
# 检查包是否正确安装
check_packages <- function(packages) {
  installed <- packages %in% installed.packages()[,"Package"]
  
  if (all(installed)) {
    cat("✓ 所有包已正确安装\n")
  } else {
    missing <- packages[!installed]
    cat("✗ 缺失以下包:\n")
    print(missing)
  }
}

check_packages(c("readxl", "ggplot2", "dplyr", "lubridate", 
                "reshape2", "car", "lmtest", "nortest"))
```

---

## 数据准备

### 数据格式要求

系统期望输入的 Excel 文件包含以下列：

| 列名 | 数据类型 | 描述 | 示例 |
|------|---------|------|------|
| `pickup_datetime` | 日期时间 | 上车时间 | "2015-06-15 08:30:00" |
| `pickup_longitude` | 数值 | 上车经度 | -73.9876 |
| `pickup_latitude` | 数值 | 上车纬度 | 40.7484 |
| `dropoff_longitude` | 数值 | 下车经度 | -73.9512 |
| `dropoff_latitude` | 数值 | 下车纬度 | 40.7896 |
| `passenger_count` | 整数 | 乘客数量 | 1, 2, 3... |
| `distance` | 数值 | 行程距离（英里） | 2.5 |
| `fare_amount` | 数值 | 票价金额（美元） | 12.50 |

### 数据质量检查

```r
# 加载数据
data <- read_excel("sample_uber.xlsx")

# 1. 查看数据结构
str(data)

# 2. 查看前几行
head(data)

# 3. 统计摘要
summary(data)

# 4. 检查缺失值
cat("缺失值统计:\n")
colSums(is.na(data))

# 5. 检查数据范围
cat("\n数据范围检查:\n")
cat("票价范围:", range(data$fare_amount), "\n")
cat("距离范围:", range(data$distance), "\n")
cat("经度范围:", range(c(data$pickup_longitude, data$dropoff_longitude)), "\n")
cat("纬度范围:", range(c(data$pickup_latitude, data$dropoff_latitude)), "\n")

# 6. 检查异常值
boxplot(data$fare_amount, main="Fare Amount Boxplot")
boxplot(data$distance, main="Distance Boxplot")
```

### 数据清理示例

```r
# 创建数据清理函数
clean_uber_data <- function(data) {
  cat("原始数据行数:", nrow(data), "\n")
  
  # 1. 移除缺失值
  data <- na.omit(data)
  cat("移除缺失值后:", nrow(data), "行\n")
  
  # 2. 移除异常值
  # 票价应该为正值且合理
  data <- data[data$fare_amount > 0 & data$fare_amount < 500, ]
  
  # 距离应该为正值且合理
  data <- data[data$distance > 0 & data$distance < 100, ]
  
  # 乘客数量应该合理
  data <- data[data$passenger_count > 0 & data$passenger_count <= 6, ]
  
  # 经纬度应该在纽约市范围内（如果是纽约数据）
  data <- data[data$pickup_longitude > -75 & data$pickup_longitude < -73, ]
  data <- data[data$pickup_latitude > 40 & data$pickup_latitude < 41, ]
  data <- data[data$dropoff_longitude > -75 & data$dropoff_longitude < -73, ]
  data <- data[data$dropoff_latitude > 40 & data$dropoff_latitude < 41, ]
  
  cat("清理后:", nrow(data), "行\n")
  
  return(data)
}

# 使用清理函数
data_clean <- clean_uber_data(data)
```

---

## 基础使用教程

### 教程 1: 数据加载和探索

#### 步骤 1: 加载数据

```r
# 加载库
library(readxl)
library(dplyr)
library(lubridate)

# 读取数据
file_path <- "sample_uber.xlsx"
data <- read_excel(file_path)

# 查看数据
print(head(data))
print(dim(data))
```

#### 步骤 2: 提取时间特征

```r
# 转换日期时间并提取特征
data <- data %>%
  mutate(
    pickup_datetime = ymd_hms(pickup_datetime),
    year = year(pickup_datetime),
    month = month(pickup_datetime),
    day = day(pickup_datetime),
    hour = hour(pickup_datetime),
    weekday = wday(pickup_datetime),
    is_weekend = ifelse(weekday %in% c(1, 7), 1, 0)
  )

# 查看新特征
head(data[, c("pickup_datetime", "year", "month", "day", "hour", "weekday")])
```

#### 步骤 3: 基础统计分析

```r
# 票价统计
cat("票价统计:\n")
cat("平均票价:", mean(data$fare_amount), "\n")
cat("中位数票价:", median(data$fare_amount), "\n")
cat("标准差:", sd(data$fare_amount), "\n")

# 距离统计
cat("\n距离统计:\n")
cat("平均距离:", mean(data$distance), "\n")
cat("中位数距离:", median(data$distance), "\n")
cat("标准差:", sd(data$distance), "\n")

# 按乘客数量统计
cat("\n按乘客数量统计:\n")
passenger_summary <- data %>%
  group_by(passenger_count) %>%
  summarise(
    count = n(),
    avg_fare = mean(fare_amount),
    avg_distance = mean(distance)
  )
print(passenger_summary)
```

---

### 教程 2: 数据可视化

#### 可视化 1: 票价分布

```r
library(ggplot2)

# 直方图
ggplot(data, aes(x = fare_amount)) +
  geom_histogram(bins = 50, fill = "skyblue", color = "black", alpha = 0.7) +
  labs(title = "票价分布直方图", 
       x = "票价金额 ($)", 
       y = "频数") +
  theme_minimal()

# 密度图
ggplot(data, aes(x = fare_amount)) +
  geom_density(fill = "orange", alpha = 0.5) +
  labs(title = "票价密度分布图", 
       x = "票价金额 ($)", 
       y = "密度") +
  theme_minimal()
```

#### 可视化 2: 票价与距离关系

```r
# 散点图
ggplot(data, aes(x = distance, y = fare_amount)) +
  geom_point(alpha = 0.3, color = "blue") +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  labs(title = "票价与距离的关系", 
       x = "距离 (英里)", 
       y = "票价金额 ($)") +
  theme_minimal()

# 计算相关系数
correlation <- cor(data$distance, data$fare_amount)
cat("距离与票价的相关系数:", correlation, "\n")
```

#### 可视化 3: 时间分析

```r
# 按小时统计平均票价
hourly_stats <- data %>%
  group_by(hour) %>%
  summarise(
    avg_fare = mean(fare_amount),
    count = n()
  )

# 按小时的票价变化
ggplot(hourly_stats, aes(x = hour, y = avg_fare)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "red", size = 2) +
  scale_x_continuous(breaks = 0:23) +
  labs(title = "一天中不同时段的平均票价", 
       x = "小时", 
       y = "平均票价 ($)") +
  theme_minimal()

# 按月份的票价变化
monthly_stats <- data %>%
  group_by(month) %>%
  summarise(
    avg_fare = mean(fare_amount),
    count = n()
  )

ggplot(monthly_stats, aes(x = factor(month), y = avg_fare, fill = factor(month))) +
  geom_bar(stat = "identity", color = "black") +
  labs(title = "不同月份的平均票价", 
       x = "月份", 
       y = "平均票价 ($)") +
  theme_minimal() +
  theme(legend.position = "none")
```

#### 可视化 4: 地理分布

```r
# 上车位置的地理分布（按票价着色）
ggplot(data, aes(x = pickup_longitude, y = pickup_latitude, color = fare_amount)) +
  geom_point(alpha = 0.5, size = 1) +
  scale_color_gradient(low = "blue", high = "red", name = "票价 ($)") +
  labs(title = "上车位置的地理分布", 
       x = "经度", 
       y = "纬度") +
  theme_minimal() +
  coord_fixed(ratio = 1)
```

---

### 教程 3: 建立简单线性回归模型

#### 步骤 1: 准备数据

```r
# 移除不需要的列
data_model <- subset(data, select = -c(pickup_datetime, Code, key))

# 查看数据
str(data_model)
```

#### 步骤 2: 建立模型

```r
# 简单线性回归（仅使用距离）
model_simple <- lm(fare_amount ~ distance, data = data_model)
summary(model_simple)

# 多元线性回归（使用多个变量）
model_multiple <- lm(fare_amount ~ distance + passenger_count + hour, 
                     data = data_model)
summary(model_multiple)
```

#### 步骤 3: 解释结果

```r
# 提取关键指标
r_squared <- summary(model_multiple)$r.squared
adj_r_squared <- summary(model_multiple)$adj.r.squared

cat("模型性能:\n")
cat("R²:", r_squared, "\n")
cat("调整 R²:", adj_r_squared, "\n")
cat("\n解释: 模型解释了", round(r_squared * 100, 2), "% 的票价变异\n")

# 提取系数
coefficients <- coef(model_multiple)
cat("\n模型系数:\n")
print(coefficients)

cat("\n系数解释:\n")
cat("- 截距:", coefficients[1], "\n")
cat("- 距离每增加1英里，票价增加:", coefficients["distance"], "美元\n")
cat("- 乘客每增加1人，票价变化:", coefficients["passenger_count"], "美元\n")
```

#### 步骤 4: 进行预测

```r
# 创建新数据
new_trip <- data.frame(
  distance = 5.0,
  passenger_count = 2,
  hour = 14
)

# 预测票价
predicted_fare <- predict(model_multiple, newdata = new_trip)
cat("预测票价:", round(predicted_fare, 2), "美元\n")

# 批量预测
new_trips <- data.frame(
  distance = c(2, 5, 10, 15),
  passenger_count = c(1, 2, 1, 3),
  hour = c(8, 14, 18, 22)
)

predictions <- predict(model_multiple, newdata = new_trips)
new_trips$predicted_fare <- round(predictions, 2)
print(new_trips)
```

---

### 教程 4: 模型诊断

#### 诊断 1: 残差分析

```r
# 绘制诊断图
par(mfrow = c(2, 2))
plot(model_multiple)
par(mfrow = c(1, 1))

# 残差 vs 拟合值
plot(model_multiple, which = 1)

# 判断标准:
# ✓ 残差应该随机分布在 0 线周围
# ✓ 不应该有明显的模式
# ✗ 如果呈现漏斗形或曲线，说明违反了假设
```

#### 诊断 2: 正态性检查

```r
# Q-Q 图
plot(model_multiple, which = 2)

# 正态性检验
library(nortest)
residuals_model <- residuals(model_multiple)
ad_test <- ad.test(residuals_model)
print(ad_test)

if (ad_test$p.value > 0.05) {
  cat("✓ 残差服从正态分布\n")
} else {
  cat("✗ 残差不服从正态分布，可能需要转换\n")
}

# 直方图
hist(residuals_model, breaks = 50, main = "残差分布直方图", 
     xlab = "残差", col = "lightblue")
```

#### 诊断 3: 同方差性检验

```r
library(lmtest)

# Breusch-Pagan 检验
bp_test <- bptest(model_multiple)
print(bp_test)

if (bp_test$p.value > 0.05) {
  cat("✓ 同方差性假设成立\n")
} else {
  cat("✗ 存在异方差性\n")
}

# Scale-Location 图
plot(model_multiple, which = 3)
```

#### 诊断 4: 多重共线性检验

```r
library(car)

# 计算 VIF
vif_values <- vif(model_multiple)
print(vif_values)

# 判断标准
for (var in names(vif_values)) {
  vif <- vif_values[var]
  if (vif < 5) {
    cat("✓", var, "- VIF =", vif, "（无问题）\n")
  } else if (vif < 10) {
    cat("⚠", var, "- VIF =", vif, "（中度共线性）\n")
  } else {
    cat("✗", var, "- VIF =", vif, "（严重共线性）\n")
  }
}
```

---

## 高级使用技巧

### 技巧 1: 特征工程 - 地理聚类

地理聚类可以帮助捕捉不同区域的票价模式。

```r
# K-means 聚类
set.seed(123)

# 准备地理坐标数据
pickup_coords <- data[, c("pickup_longitude", "pickup_latitude")]
dropoff_coords <- data[, c("dropoff_longitude", "dropoff_latitude")]

# 执行聚类（5个聚类中心）
n_clusters <- 5
pickup_kmeans <- kmeans(pickup_coords, centers = n_clusters)
dropoff_kmeans <- kmeans(dropoff_coords, centers = n_clusters)

# 添加聚类标签到数据
data$pickup_cluster <- pickup_kmeans$cluster
data$dropoff_cluster <- dropoff_kmeans$cluster

# 可视化聚类结果
ggplot(data, aes(x = pickup_longitude, y = pickup_latitude, 
                 color = factor(pickup_cluster))) +
  geom_point(alpha = 0.5) +
  labs(title = "上车位置聚类", 
       x = "经度", 
       y = "纬度", 
       color = "聚类") +
  theme_minimal()

# 分析每个聚类的平均票价
cluster_analysis <- data %>%
  group_by(pickup_cluster) %>%
  summarise(
    avg_fare = mean(fare_amount),
    avg_distance = mean(distance),
    count = n()
  )
print(cluster_analysis)
```

### 技巧 2: 交互项特征

交互项可以捕捉变量之间的联合效应。

```r
# 创建白天/夜间特征
data$daytime <- ifelse(data$hour >= 6 & data$hour < 20, 1, 0)

# 创建交互项
data$distance_daytime <- data$distance * data$daytime
data$distance_passenger <- data$distance * data$passenger_count

# 建立包含交互项的模型
model_interaction <- lm(fare_amount ~ distance + daytime + distance_daytime + 
                        passenger_count + distance_passenger, 
                        data = data)

summary(model_interaction)

# 解释交互项
coef <- coef(model_interaction)
cat("交互项解释:\n")
cat("白天时，距离对票价的影响:", coef["distance"] + coef["distance_daytime"], "\n")
cat("夜间时，距离对票价的影响:", coef["distance"], "\n")
```

### 技巧 3: 逐步回归优化

```r
# 建立包含所有可能变量的完整模型
full_model <- lm(fare_amount ~ pickup_longitude + pickup_latitude + 
                 dropoff_longitude + dropoff_latitude + 
                 distance + passenger_count + 
                 year + month + day + hour + 
                 pickup_cluster + dropoff_cluster + 
                 daytime + distance_daytime, 
                 data = data)

# 向后逐步回归
step_model <- step(full_model, direction = "backward", trace = 0)

# 比较模型
cat("完整模型 R²:", summary(full_model)$r.squared, "\n")
cat("逐步模型 R²:", summary(step_model)$r.squared, "\n")

cat("\n完整模型变量数:", length(coef(full_model)), "\n")
cat("逐步模型变量数:", length(coef(step_model)), "\n")

# 查看保留的变量
cat("\n逐步回归保留的变量:\n")
print(names(coef(step_model)))
```

### 技巧 4: 变量转换处理非正态性

```r
# 检查票价分布
hist(data$fare_amount, breaks = 50, main = "原始票价分布")

# 对数转换
data$log_fare <- log(data$fare_amount)
hist(data$log_fare, breaks = 50, main = "对数转换后的票价分布")

# 建立转换后的模型
model_log <- lm(log_fare ~ distance + passenger_count + hour, data = data)

# 诊断检查
par(mfrow = c(2, 2))
plot(model_log)
par(mfrow = c(1, 1))

# 进行预测（需要反向转换）
new_trip <- data.frame(distance = 5, passenger_count = 2, hour = 14)
log_prediction <- predict(model_log, newdata = new_trip)
actual_prediction <- exp(log_prediction)
cat("预测票价:", round(actual_prediction, 2), "美元\n")
```

### 技巧 5: 处理异常值和影响点

```r
# 识别 Cook's 距离
cooksd <- cooks.distance(model_multiple)

# 绘制 Cook's 距离
plot(cooksd, type = "h", main = "Cook's Distance", ylab = "Cook's Distance")
abline(h = 4/nrow(data), col = "red", lty = 2)

# 识别影响点
threshold <- 4 / nrow(data)
influential_points <- which(cooksd > threshold)
cat("发现", length(influential_points), "个影响点\n")

# 移除影响点并重新拟合
if (length(influential_points) > 0) {
  data_clean <- data[-influential_points, ]
  model_clean <- lm(fare_amount ~ distance + passenger_count + hour, 
                    data = data_clean)
  
  cat("\n原始模型 R²:", summary(model_multiple)$r.squared, "\n")
  cat("清理后模型 R²:", summary(model_clean)$r.squared, "\n")
}
```

---

## 常见场景示例

### 场景 1: 预测单次行程票价

```r
# 场景: 用户想知道一次特定行程的预计费用

# 行程信息
trip_info <- data.frame(
  pickup_longitude = -73.98,
  pickup_latitude = 40.75,
  dropoff_longitude = -73.95,
  dropoff_latitude = 40.78,
  distance = 2.5,
  passenger_count = 1,
  hour = 14,
  month = 6,
  year = 2015
)

# 添加额外特征
trip_info$daytime <- ifelse(trip_info$hour >= 6 & trip_info$hour < 20, 1, 0)
trip_info$distance_daytime <- trip_info$distance * trip_info$daytime

# 使用模型预测
predicted_fare <- predict(step_model, newdata = trip_info)

cat("========== 行程预测 ==========\n")
cat("上车位置:", trip_info$pickup_longitude, ",", trip_info$pickup_latitude, "\n")
cat("下车位置:", trip_info$dropoff_longitude, ",", trip_info$dropoff_latitude, "\n")
cat("距离:", trip_info$distance, "英里\n")
cat("乘客数量:", trip_info$passenger_count, "\n")
cat("时间:", trip_info$hour, ":00\n")
cat("预测票价:", round(predicted_fare, 2), "美元\n")
cat("==============================\n")
```

### 场景 2: 批量预测和分析

```r
# 场景: 出租车公司想预测一天内多个行程的收入

# 创建多个行程
trips <- data.frame(
  distance = c(2.5, 5.0, 10.0, 3.2, 7.5),
  passenger_count = c(1, 2, 1, 4, 2),
  hour = c(8, 12, 18, 14, 22)
)

# 添加特征
trips$daytime <- ifelse(trips$hour >= 6 & trips$hour < 20, 1, 0)
trips$distance_daytime <- trips$distance * trips$daytime

# 预测所有行程
trips$predicted_fare <- predict(step_model, newdata = trips)

# 统计分析
cat("========== 行程分析报告 ==========\n")
cat("总行程数:", nrow(trips), "\n")
cat("总距离:", sum(trips$distance), "英里\n")
cat("总预计收入:", round(sum(trips$predicted_fare), 2), "美元\n")
cat("平均每英里收入:", round(sum(trips$predicted_fare) / sum(trips$distance), 2), "美元/英里\n")
cat("==================================\n")

print(trips)
```

### 场景 3: 最优价格时段分析

```r
# 场景: 司机想知道什么时候开车最赚钱

# 按小时分组分析
hourly_analysis <- data %>%
  group_by(hour) %>%
  summarise(
    avg_fare = mean(fare_amount),
    avg_distance = mean(distance),
    trip_count = n(),
    revenue_per_mile = mean(fare_amount / distance)
  ) %>%
  arrange(desc(revenue_per_mile))

cat("========== 最优时段分析 ==========\n")
cat("最赚钱的5个时段:\n")
print(head(hourly_analysis, 5))

# 可视化
ggplot(hourly_analysis, aes(x = hour, y = revenue_per_mile)) +
  geom_line(color = "blue", size = 1) +
  geom_point(aes(size = trip_count), color = "red") +
  scale_x_continuous(breaks = 0:23) +
  labs(title = "每小时的单位距离收入", 
       x = "小时", 
       y = "每英里收入 ($)", 
       size = "行程数") +
  theme_minimal()
```

### 场景 4: 区域定价策略

```r
# 场景: 分析不同区域的定价模式

# 按聚类分析
cluster_pricing <- data %>%
  group_by(pickup_cluster, dropoff_cluster) %>%
  summarise(
    avg_fare = mean(fare_amount),
    avg_distance = mean(distance),
    trip_count = n(),
    price_per_mile = mean(fare_amount / distance)
  ) %>%
  arrange(desc(price_per_mile))

cat("========== 区域定价分析 ==========\n")
cat("最高价格路线 (前10条):\n")
print(head(cluster_pricing, 10))

# 热图可视化
library(reshape2)
pricing_matrix <- dcast(cluster_pricing, pickup_cluster ~ dropoff_cluster, 
                        value.var = "price_per_mile")

ggplot(melt(pricing_matrix), aes(x = pickup_cluster, y = variable, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "yellow", high = "red", name = "$/英里") +
  labs(title = "不同区域组合的单位距离价格", 
       x = "上车区域", 
       y = "下车区域") +
  theme_minimal()
```

### 场景 5: 季节性分析

```r
# 场景: 分析票价的季节性变化

# 按月份分析
monthly_analysis <- data %>%
  group_by(year, month) %>%
  summarise(
    avg_fare = mean(fare_amount),
    avg_distance = mean(distance),
    trip_count = n()
  ) %>%
  arrange(year, month)

# 添加月份标签
monthly_analysis$month_label <- factor(monthly_analysis$month, 
                                       levels = 1:12, 
                                       labels = c("一月", "二月", "三月", "四月", 
                                                 "五月", "六月", "七月", "八月", 
                                                 "九月", "十月", "十一月", "十二月"))

# 可视化
ggplot(monthly_analysis, aes(x = month, y = avg_fare, group = year, color = factor(year))) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  scale_x_continuous(breaks = 1:12) +
  labs(title = "月度平均票价趋势", 
       x = "月份", 
       y = "平均票价 ($)", 
       color = "年份") +
  theme_minimal()

cat("========== 季节性分析 ==========\n")
cat("最高平均票价月份:\n")
print(monthly_analysis[which.max(monthly_analysis$avg_fare), ])
cat("\n最低平均票价月份:\n")
print(monthly_analysis[which.min(monthly_analysis$avg_fare), ])
```

---

## 故障排除

### 问题 1: 无法加载 Excel 文件

**错误信息**: `Error: 'sample_uber.xlsx' does not exist`

**解决方案**:
```r
# 检查当前工作目录
getwd()

# 设置正确的工作目录
setwd("/path/to/your/data")

# 或者使用完整路径
data <- read_excel("/path/to/sample_uber.xlsx")
```

### 问题 2: 包加载失败

**错误信息**: `Error: package 'xxx' is not installed`

**解决方案**:
```r
# 重新安装包
install.packages("package_name")

# 如果还是失败，尝试指定镜像
install.packages("package_name", repos = "https://cloud.r-project.org/")

# 检查 R 版本
version
```

### 问题 3: 内存不足

**错误信息**: `Error: cannot allocate vector of size XX MB`

**解决方案**:
```r
# 1. 增加内存限制（Windows）
memory.limit(size = 8000)  # 8GB

# 2. 清理工作空间
rm(list = ls())
gc()

# 3. 使用数据采样
data_sample <- data[sample(1:nrow(data), 10000), ]

# 4. 分批处理数据
process_in_batches <- function(data, batch_size = 10000) {
  n_batches <- ceiling(nrow(data) / batch_size)
  results <- list()
  
  for (i in 1:n_batches) {
    start_idx <- (i - 1) * batch_size + 1
    end_idx <- min(i * batch_size, nrow(data))
    batch <- data[start_idx:end_idx, ]
    
    # 处理批次
    results[[i]] <- process_batch(batch)
  }
  
  return(do.call(rbind, results))
}
```

### 问题 4: 模型拟合警告

**警告信息**: `Warning: prediction from a rank-deficient fit`

**解决方案**:
```r
# 检查多重共线性
vif_values <- vif(model)
print(vif_values)

# 移除高 VIF 变量
# 重新构建模型
model_new <- lm(fare_amount ~ distance + passenger_count, data = data)
```

### 问题 5: 日期解析错误

**错误信息**: `Error: All formats failed to parse`

**解决方案**:
```r
# 检查日期格式
head(data$pickup_datetime)

# 尝试不同的解析格式
data$pickup_datetime <- ymd_hms(data$pickup_datetime)  # "2015-06-15 08:30:00"
# 或
data$pickup_datetime <- mdy_hm(data$pickup_datetime)   # "06/15/2015 08:30"
# 或
data$pickup_datetime <- dmy_hms(data$pickup_datetime)  # "15-06-2015 08:30:00"

# 手动指定格式
data$pickup_datetime <- as.POSIXct(data$pickup_datetime, 
                                   format = "%Y-%m-%d %H:%M:%S")
```

---

## 性能优化建议

### 优化 1: 使用 data.table 加速

```r
# 安装并加载 data.table
install.packages("data.table")
library(data.table)

# 转换为 data.table
dt <- as.data.table(data)

# 快速分组操作
result <- dt[, .(avg_fare = mean(fare_amount), 
                 count = .N), 
             by = .(hour, passenger_count)]
```

### 优化 2: 并行处理

```r
# 安装并加载并行处理包
install.packages("parallel")
library(parallel)

# 检测核心数
n_cores <- detectCores() - 1
cl <- makeCluster(n_cores)

# 并行应用函数
results <- parLapply(cl, data_list, process_function)

# 关闭集群
stopCluster(cl)
```

### 优化 3: 缓存中间结果

```r
# 保存处理后的数据
saveRDS(data_processed, "processed_data.rds")

# 加载缓存数据
data_processed <- readRDS("processed_data.rds")

# 保存模型
saveRDS(model, "trained_model.rds")

# 加载模型
model <- readRDS("trained_model.rds")
```

---

## 常见问题解答

### Q1: 为什么我的 R² 值很低？

**A**: R² 值低可能的原因：
1. 缺少重要预测变量
2. 数据质量问题（异常值、缺失值）
3. 变量之间可能不是线性关系

**建议**:
- 添加更多相关特征
- 尝试变量转换
- 考虑非线性模型

### Q2: 如何选择合适的聚类数量？

**A**: 使用肘部法则：

```r
# 计算不同 k 值的 WSS
wss <- sapply(1:10, function(k) {
  kmeans(pickup_coords, centers = k)$tot.withinss
})

# 绘制肘部图
plot(1:10, wss, type = "b", 
     xlab = "聚类数量", 
     ylab = "组内平方和",
     main = "肘部法则")
```

### Q3: 预测值为负数怎么办？

**A**: 
```r
# 方法 1: 使用对数转换
model_log <- lm(log(fare_amount) ~ ..., data = data)
predictions <- exp(predict(model_log, newdata = new_data))

# 方法 2: 设置下限
predictions[predictions < 0] <- 0

# 方法 3: 使用 GLM
model_glm <- glm(fare_amount ~ ..., family = Gamma(link = "log"), data = data)
```

### Q4: 如何处理分类变量？

**A**:
```r
# 创建虚拟变量
data$is_weekend <- factor(data$is_weekend)
model <- lm(fare_amount ~ distance + is_weekend, data = data)

# 或使用 model.matrix
dummy_vars <- model.matrix(~ is_weekend - 1, data = data)
```

### Q5: 如何评估模型在实际应用中的表现？

**A**:
```r
# 时间序列交叉验证
# 使用早期数据训练，晚期数据测试
train_data <- data[data$year == 2014, ]
test_data <- data[data$year == 2015, ]

model <- lm(fare_amount ~ ..., data = train_data)
predictions <- predict(model, newdata = test_data)

# 计算实际性能指标
mae <- mean(abs(predictions - test_data$fare_amount))
rmse <- sqrt(mean((predictions - test_data$fare_amount)^2))

cat("MAE:", mae, "\n")
cat("RMSE:", rmse, "\n")
```

---

## 总结

本指南涵盖了 Uber 票价预测系统的主要使用方法，从基础到高级。记住以下关键点：

1. **始终检查数据质量**
2. **进行全面的探索性数据分析**
3. **创建有意义的特征**
4. **验证模型假设**
5. **使用测试集评估性能**
6. **迭代改进模型**

如有任何问题或需要进一步帮助，请参考 API 文档或函数库文档。

---

**文档版本**: 1.0  
**最后更新**: 2025-11-21  
**作者**: DTS206TC_CW_2144212
