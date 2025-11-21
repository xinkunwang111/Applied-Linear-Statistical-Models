# Uber 票价分析函数库

## 概述

本文档提供了一套完整的 R 函数库，用于 Uber 票价数据分析。所有函数都经过模块化设计，可以独立使用或组合使用。

---

## 目录

1. [数据处理函数](#数据处理函数)
2. [可视化函数](#可视化函数)
3. [特征工程函数](#特征工程函数)
4. [建模函数](#建模函数)
5. [诊断函数](#诊断函数)
6. [实用工具函数](#实用工具函数)

---

## 数据处理函数

### `load_uber_data(file_path)`

加载并进行基础预处理的 Uber 数据

**参数**:
- `file_path` (字符串): Excel 文件路径

**返回值**: 处理后的数据框

**示例**:
```r
load_uber_data <- function(file_path) {
  # 加载数据
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
  
  return(data)
}

# 使用示例
data <- load_uber_data("sample_uber.xlsx")
```

---

### `clean_data(data, remove_na = TRUE, remove_outliers = FALSE)`

清理数据，移除缺失值和异常值

**参数**:
- `data` (数据框): 输入数据
- `remove_na` (逻辑值): 是否移除缺失值
- `remove_outliers` (逻辑值): 是否移除异常值

**返回值**: 清理后的数据框

**示例**:
```r
clean_data <- function(data, remove_na = TRUE, remove_outliers = FALSE) {
  # 移除缺失值
  if (remove_na) {
    data <- na.omit(data)
  }
  
  # 移除异常值（使用 IQR 方法）
  if (remove_outliers) {
    numeric_cols <- names(data)[sapply(data, is.numeric)]
    for (col in numeric_cols) {
      Q1 <- quantile(data[[col]], 0.25, na.rm = TRUE)
      Q3 <- quantile(data[[col]], 0.75, na.rm = TRUE)
      IQR <- Q3 - Q1
      lower_bound <- Q1 - 1.5 * IQR
      upper_bound <- Q3 + 1.5 * IQR
      data <- data[data[[col]] >= lower_bound & data[[col]] <= upper_bound, ]
    }
  }
  
  return(data)
}

# 使用示例
data_clean <- clean_data(data, remove_na = TRUE, remove_outliers = TRUE)
```

---

### `split_train_test(data, train_ratio = 0.8, seed = 123)`

将数据分割为训练集和测试集

**参数**:
- `data` (数据框): 输入数据
- `train_ratio` (数值): 训练集比例 (0-1)
- `seed` (整数): 随机种子

**返回值**: 包含 `train` 和 `test` 的列表

**示例**:
```r
split_train_test <- function(data, train_ratio = 0.8, seed = 123) {
  set.seed(seed)
  train_indices <- sample(1:nrow(data), train_ratio * nrow(data))
  
  return(list(
    train = data[train_indices, ],
    test = data[-train_indices, ]
  ))
}

# 使用示例
split_data <- split_train_test(data, train_ratio = 0.8)
train_data <- split_data$train
test_data <- split_data$test
```

---

## 可视化函数

### `plot_correlation_heatmap(data, title = "Correlation Heatmap")`

绘制数值变量的相关性热图

**参数**:
- `data` (数据框): 输入数据
- `title` (字符串): 图表标题

**返回值**: ggplot 对象

**示例**:
```r
plot_correlation_heatmap <- function(data, title = "Correlation Heatmap") {
  # 计算相关矩阵
  cor_matrix <- cor(data %>% select_if(is.numeric), use = "complete.obs")
  melted_cor_matrix <- melt(cor_matrix)
  
  # 绘制热图
  p <- ggplot(data = melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile(color = "white") +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                         midpoint = 0, limit = c(-1, 1), space = "Lab",
                         name = "Correlation") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1)) +
    coord_fixed() +
    labs(title = title, x = "", y = "")
  
  return(p)
}

# 使用示例
plot_correlation_heatmap(data)
```

---

### `plot_distribution(data, variable, title = NULL, fill_color = "orange")`

绘制单个变量的密度分布图

**参数**:
- `data` (数据框): 输入数据
- `variable` (字符串): 变量名
- `title` (字符串): 图表标题
- `fill_color` (字符串): 填充颜色

**返回值**: ggplot 对象

**示例**:
```r
plot_distribution <- function(data, variable, title = NULL, fill_color = "orange") {
  if (is.null(title)) {
    title <- paste("Distribution of", variable)
  }
  
  p <- ggplot(data, aes_string(x = variable)) +
    geom_density(fill = fill_color, alpha = 0.5) +
    labs(title = title, x = variable, y = "Density") +
    theme_minimal()
  
  return(p)
}

# 使用示例
plot_distribution(data, "fare_amount", "Fare Amount Distribution")
plot_distribution(data, "distance", "Distance Distribution", "blue")
```

---

### `plot_scatter(data, x_var, y_var, title = NULL, color_var = NULL)`

绘制散点图

**参数**:
- `data` (数据框): 输入数据
- `x_var` (字符串): X 轴变量名
- `y_var` (字符串): Y 轴变量名
- `title` (字符串): 图表标题
- `color_var` (字符串): 颜色变量名（可选）

**返回值**: ggplot 对象

**示例**:
```r
plot_scatter <- function(data, x_var, y_var, title = NULL, color_var = NULL) {
  if (is.null(title)) {
    title <- paste(y_var, "vs", x_var)
  }
  
  if (is.null(color_var)) {
    p <- ggplot(data, aes_string(x = x_var, y = y_var)) +
      geom_point(alpha = 0.5, color = "blue")
  } else {
    p <- ggplot(data, aes_string(x = x_var, y = y_var, color = color_var)) +
      geom_point(alpha = 0.6) +
      scale_color_gradient(low = "blue", high = "red")
  }
  
  p <- p + 
    labs(title = title, x = x_var, y = y_var) +
    theme_minimal()
  
  return(p)
}

# 使用示例
plot_scatter(data, "distance", "fare_amount", "Fare vs Distance")
plot_scatter(data, "pickup_longitude", "pickup_latitude", "Pickup Locations", "fare_amount")
```

---

### `plot_boxplot(data, x_var, y_var, title = NULL, fill_color = "blue")`

绘制箱线图

**参数**:
- `data` (数据框): 输入数据
- `x_var` (字符串): X 轴分类变量
- `y_var` (字符串): Y 轴数值变量
- `title` (字符串): 图表标题
- `fill_color` (字符串): 填充颜色

**返回值**: ggplot 对象

**示例**:
```r
plot_boxplot <- function(data, x_var, y_var, title = NULL, fill_color = "blue") {
  if (is.null(title)) {
    title <- paste(y_var, "by", x_var)
  }
  
  p <- ggplot(data, aes_string(x = x_var, y = y_var)) +
    geom_boxplot(alpha = 0.7, fill = fill_color, color = "black") +
    labs(title = title, x = x_var, y = y_var) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  return(p)
}

# 使用示例
data$year <- as.factor(data$year)
plot_boxplot(data, "year", "fare_amount", "Fare Amount by Year")
```

---

### `plot_bar_comparison(data, x_var, y_var, title = NULL)`

绘制分组均值比较条形图

**参数**:
- `data` (数据框): 输入数据
- `x_var` (字符串): X 轴分类变量
- `y_var` (字符串): Y 轴数值变量
- `title` (字符串): 图表标题

**返回值**: ggplot 对象

**示例**:
```r
plot_bar_comparison <- function(data, x_var, y_var, title = NULL) {
  if (is.null(title)) {
    title <- paste("Average", y_var, "by", x_var)
  }
  
  p <- ggplot(data, aes_string(x = paste0("factor(", x_var, ")"), 
                                y = y_var, 
                                fill = paste0("factor(", x_var, ")"))) +
    geom_bar(stat = "summary", fun = "mean", color = "black", alpha = 0.7) +
    labs(title = title, x = x_var, y = paste("Average", y_var)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  return(p)
}

# 使用示例
plot_bar_comparison(data, "passenger_count", "fare_amount", 
                    "Average Fare by Passenger Count")
```

---

## 特征工程函数

### `add_time_features(data)`

添加时间相关特征

**参数**:
- `data` (数据框): 包含 `pickup_datetime` 列的数据

**返回值**: 添加了时间特征的数据框

**示例**:
```r
add_time_features <- function(data) {
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
  
  return(data)
}

# 使用示例
data <- add_time_features(data)
```

---

### `add_daytime_feature(data, day_start = 6, day_end = 20)`

添加白天/夜间特征

**参数**:
- `data` (数据框): 包含 `hour` 列的数据
- `day_start` (整数): 白天开始小时
- `day_end` (整数): 白天结束小时

**返回值**: 添加了 `daytime` 列的数据框

**示例**:
```r
add_daytime_feature <- function(data, day_start = 6, day_end = 20) {
  data <- data %>%
    mutate(
      daytime = ifelse(hour >= day_start & hour < day_end, 1, 0)
    )
  
  return(data)
}

# 使用示例
data <- add_daytime_feature(data)
```

---

### `create_location_clusters(data, n_clusters = 5, seed = 123)`

对上车和下车位置进行 K-means 聚类

**参数**:
- `data` (数据框): 包含经纬度列的数据
- `n_clusters` (整数): 聚类数量
- `seed` (整数): 随机种子

**返回值**: 添加了聚类列的数据框

**示例**:
```r
create_location_clusters <- function(data, n_clusters = 5, seed = 123) {
  set.seed(seed)
  
  # 准备数据
  pickup_data <- data.frame(data$pickup_longitude, data$pickup_latitude)
  dropoff_data <- data.frame(data$dropoff_longitude, data$dropoff_latitude)
  
  # 执行聚类
  pickup_clusters <- kmeans(pickup_data, centers = n_clusters)$cluster
  dropoff_clusters <- kmeans(dropoff_data, centers = n_clusters)$cluster
  
  # 添加到数据
  data <- data %>%
    mutate(
      pickup_cluster = pickup_clusters,
      dropoff_cluster = dropoff_clusters
    )
  
  return(data)
}

# 使用示例
data <- create_location_clusters(data, n_clusters = 5)
```

---

### `plot_clusters(data, cluster_type = "pickup")`

可视化位置聚类结果

**参数**:
- `data` (数据框): 包含聚类结果的数据
- `cluster_type` (字符串): "pickup" 或 "dropoff"

**返回值**: ggplot 对象

**示例**:
```r
plot_clusters <- function(data, cluster_type = "pickup") {
  if (cluster_type == "pickup") {
    p <- ggplot(data, aes(x = pickup_longitude, y = pickup_latitude, 
                          color = factor(pickup_cluster))) +
      geom_point(alpha = 0.5) +
      labs(title = "Pickup Locations Clustering", 
           x = "Pickup Longitude", 
           y = "Pickup Latitude", 
           color = "Cluster") +
      theme_minimal()
  } else {
    p <- ggplot(data, aes(x = dropoff_longitude, y = dropoff_latitude, 
                          color = factor(dropoff_cluster))) +
      geom_point(alpha = 0.5) +
      labs(title = "Dropoff Locations Clustering", 
           x = "Dropoff Longitude", 
           y = "Dropoff Latitude", 
           color = "Cluster") +
      theme_minimal()
  }
  
  return(p)
}

# 使用示例
plot_clusters(data, "pickup")
plot_clusters(data, "dropoff")
```

---

### `add_interaction_terms(data, var1, var2, interaction_name = NULL)`

创建交互项特征

**参数**:
- `data` (数据框): 输入数据
- `var1` (字符串): 第一个变量名
- `var2` (字符串): 第二个变量名
- `interaction_name` (字符串): 交互项名称（可选）

**返回值**: 添加了交互项的数据框

**示例**:
```r
add_interaction_terms <- function(data, var1, var2, interaction_name = NULL) {
  if (is.null(interaction_name)) {
    interaction_name <- paste0(var1, "_", var2, "_interaction")
  }
  
  data[[interaction_name]] <- data[[var1]] * data[[var2]]
  
  return(data)
}

# 使用示例
data <- add_interaction_terms(data, "distance", "daytime", "distance_daytime_interaction")
```

---

### `apply_transformations(data, log_vars = NULL, sqrt_vars = NULL)`

对变量应用数学转换

**参数**:
- `data` (数据框): 输入数据
- `log_vars` (字符向量): 需要对数转换的变量
- `sqrt_vars` (字符向量): 需要平方根转换的变量

**返回值**: 添加了转换变量的数据框

**示例**:
```r
apply_transformations <- function(data, log_vars = NULL, sqrt_vars = NULL) {
  # 对数转换
  if (!is.null(log_vars)) {
    for (var in log_vars) {
      new_var_name <- paste0("log_", var)
      data[[new_var_name]] <- log(data[[var]])
    }
  }
  
  # 平方根转换
  if (!is.null(sqrt_vars)) {
    for (var in sqrt_vars) {
      new_var_name <- paste0("sqrt_", var)
      data[[new_var_name]] <- sqrt(data[[var]])
    }
  }
  
  return(data)
}

# 使用示例
data <- apply_transformations(data, 
                              log_vars = c("fare_amount", "distance"),
                              sqrt_vars = c("distance"))
```

---

## 建模函数

### `build_linear_model(data, formula, summary_output = TRUE)`

构建线性回归模型

**参数**:
- `data` (数据框): 训练数据
- `formula` (公式): 模型公式
- `summary_output` (逻辑值): 是否输出摘要

**返回值**: lm 模型对象

**示例**:
```r
build_linear_model <- function(data, formula, summary_output = TRUE) {
  model <- lm(formula, data = data)
  
  if (summary_output) {
    print(summary(model))
  }
  
  return(model)
}

# 使用示例
formula <- fare_amount ~ pickup_longitude + pickup_latitude + 
           dropoff_longitude + dropoff_latitude + distance + passenger_count
model <- build_linear_model(data, formula)
```

---

### `stepwise_selection(model, direction = "backward", trace = 1)`

执行逐步回归

**参数**:
- `model` (lm 对象): 初始模型
- `direction` (字符串): "backward", "forward", 或 "both"
- `trace` (整数): 详细程度

**返回值**: 优化后的 lm 模型对象

**示例**:
```r
stepwise_selection <- function(model, direction = "backward", trace = 1) {
  stepwise_model <- step(model, direction = direction, trace = trace)
  
  return(stepwise_model)
}

# 使用示例
stepwise_model <- stepwise_selection(model, direction = "backward")
```

---

### `evaluate_model(model, test_data = NULL)`

评估模型性能

**参数**:
- `model` (lm 对象): 训练好的模型
- `test_data` (数据框): 测试数据（可选）

**返回值**: 包含性能指标的列表

**示例**:
```r
evaluate_model <- function(model, test_data = NULL) {
  # 训练集性能
  summary_stats <- summary(model)
  r_squared <- summary_stats$r.squared
  adj_r_squared <- summary_stats$adj.r.squared
  residual_std_error <- summary_stats$sigma
  
  results <- list(
    r_squared = r_squared,
    adj_r_squared = adj_r_squared,
    residual_std_error = residual_std_error
  )
  
  # 测试集性能（如果提供）
  if (!is.null(test_data)) {
    predictions <- predict(model, newdata = test_data)
    actual <- test_data[[as.character(formula(model)[[2]])]]
    
    mae <- mean(abs(predictions - actual))
    rmse <- sqrt(mean((predictions - actual)^2))
    mape <- mean(abs((predictions - actual) / actual)) * 100
    
    results$test_mae <- mae
    results$test_rmse <- rmse
    results$test_mape <- mape
  }
  
  return(results)
}

# 使用示例
performance <- evaluate_model(model, test_data = test_data)
print(performance)
```

---

### `make_predictions(model, new_data)`

使用模型进行预测

**参数**:
- `model` (lm 对象): 训练好的模型
- `new_data` (数据框): 新数据

**返回值**: 预测值向量

**示例**:
```r
make_predictions <- function(model, new_data) {
  predictions <- predict(model, newdata = new_data)
  
  return(predictions)
}

# 使用示例
predictions <- make_predictions(model, test_data)
```

---

## 诊断函数

### `plot_diagnostic_plots(model)`

绘制模型诊断图

**参数**:
- `model` (lm 对象): 线性模型

**返回值**: 无（直接绘图）

**示例**:
```r
plot_diagnostic_plots <- function(model) {
  par(mfrow = c(2, 2))
  plot(model)
  par(mfrow = c(1, 1))
}

# 使用示例
plot_diagnostic_plots(model)
```

---

### `test_homoscedasticity(model)`

测试同方差性

**参数**:
- `model` (lm 对象): 线性模型

**返回值**: Breusch-Pagan 检验结果

**示例**:
```r
test_homoscedasticity <- function(model) {
  bp_test <- bptest(model)
  
  cat("Breusch-Pagan 检验:\n")
  print(bp_test)
  
  if (bp_test$p.value > 0.05) {
    cat("\n结论: 同方差性假设成立 (p > 0.05)\n")
  } else {
    cat("\n结论: 存在异方差性 (p ≤ 0.05)\n")
  }
  
  return(bp_test)
}

# 使用示例
test_homoscedasticity(model)
```

---

### `test_autocorrelation(model)`

测试自相关性

**参数**:
- `model` (lm 对象): 线性模型

**返回值**: Durbin-Watson 检验结果

**示例**:
```r
test_autocorrelation <- function(model) {
  dw_test <- durbinWatsonTest(model)
  
  cat("Durbin-Watson 检验:\n")
  print(dw_test)
  
  if (abs(dw_test$dw - 2) < 0.5) {
    cat("\n结论: 无显著自相关 (DW ≈ 2)\n")
  } else {
    cat("\n结论: 存在自相关\n")
  }
  
  return(dw_test)
}

# 使用示例
test_autocorrelation(model)
```

---

### `test_normality(model)`

测试残差正态性

**参数**:
- `model` (lm 对象): 线性模型

**返回值**: Anderson-Darling 检验结果

**示例**:
```r
test_normality <- function(model) {
  residuals_model <- residuals(model)
  ad_test <- ad.test(residuals_model)
  
  cat("Anderson-Darling 正态性检验:\n")
  print(ad_test)
  
  if (ad_test$p.value > 0.05) {
    cat("\n结论: 残差服从正态分布 (p > 0.05)\n")
  } else {
    cat("\n结论: 残差不服从正态分布 (p ≤ 0.05)\n")
  }
  
  # 绘制 Q-Q 图
  qqnorm(residuals_model)
  qqline(residuals_model, col = "red")
  
  return(ad_test)
}

# 使用示例
test_normality(model)
```

---

### `check_multicollinearity(model, threshold = 10)`

检查多重共线性

**参数**:
- `model` (lm 对象): 线性模型
- `threshold` (数值): VIF 阈值

**返回值**: VIF 值

**示例**:
```r
check_multicollinearity <- function(model, threshold = 10) {
  vif_values <- vif(model)
  
  cat("方差膨胀因子 (VIF):\n")
  print(vif_values)
  
  high_vif <- vif_values[vif_values > threshold]
  
  if (length(high_vif) > 0) {
    cat("\n警告: 以下变量存在严重多重共线性 (VIF >", threshold, "):\n")
    print(high_vif)
  } else {
    cat("\n结论: 无严重多重共线性\n")
  }
  
  return(vif_values)
}

# 使用示例
check_multicollinearity(model, threshold = 10)
```

---

### `identify_influential_points(model, data, threshold_multiplier = 20)`

识别高影响点

**参数**:
- `model` (lm 对象): 线性模型
- `data` (数据框): 数据
- `threshold_multiplier` (数值): 阈值乘数

**返回值**: 包含影响点索引的向量

**示例**:
```r
identify_influential_points <- function(model, data, threshold_multiplier = 20) {
  cooksd <- cooks.distance(model)
  threshold <- threshold_multiplier / nrow(data)
  
  influential_points <- which(cooksd > threshold)
  
  cat("发现", length(influential_points), "个高影响点\n")
  cat("Cook's 距离阈值:", threshold, "\n")
  
  # 绘制 Cook's 距离图
  plot(cooksd, type = "h", main = "Cook's Distance",
       ylab = "Cook's Distance", xlab = "Observation Index")
  abline(h = threshold, col = "red", lty = 2)
  
  return(influential_points)
}

# 使用示例
influential <- identify_influential_points(model, data)
```

---

### `perform_comprehensive_diagnostics(model, data)`

执行全面的模型诊断

**参数**:
- `model` (lm 对象): 线性模型
- `data` (数据框): 数据

**返回值**: 包含所有诊断结果的列表

**示例**:
```r
perform_comprehensive_diagnostics <- function(model, data) {
  cat("=" %R% 50, "\n")
  cat("综合模型诊断报告\n")
  cat("=" %R% 50, "\n\n")
  
  # 1. 基本模型信息
  cat("1. 模型摘要:\n")
  print(summary(model))
  cat("\n")
  
  # 2. 诊断图
  cat("2. 诊断图:\n")
  plot_diagnostic_plots(model)
  cat("\n")
  
  # 3. 同方差性
  cat("3. 同方差性检验:\n")
  bp_result <- test_homoscedasticity(model)
  cat("\n")
  
  # 4. 自相关
  cat("4. 自相关检验:\n")
  dw_result <- test_autocorrelation(model)
  cat("\n")
  
  # 5. 正态性
  cat("5. 正态性检验:\n")
  ad_result <- test_normality(model)
  cat("\n")
  
  # 6. 多重共线性
  cat("6. 多重共线性检验:\n")
  vif_result <- check_multicollinearity(model)
  cat("\n")
  
  # 7. 影响点
  cat("7. 影响点分析:\n")
  influential <- identify_influential_points(model, data)
  cat("\n")
  
  results <- list(
    summary = summary(model),
    bp_test = bp_result,
    dw_test = dw_result,
    ad_test = ad_result,
    vif = vif_result,
    influential_points = influential
  )
  
  return(results)
}

# 使用示例
diagnostics <- perform_comprehensive_diagnostics(model, data)
```

---

## 实用工具函数

### `remove_influential_observations(data, model, threshold_multiplier = 20)`

移除高影响点观测

**参数**:
- `data` (数据框): 原始数据
- `model` (lm 对象): 模型
- `threshold_multiplier` (数值): 阈值乘数

**返回值**: 移除影响点后的数据框

**示例**:
```r
remove_influential_observations <- function(data, model, threshold_multiplier = 20) {
  cooksd <- cooks.distance(model)
  threshold <- threshold_multiplier / nrow(data)
  influential_points <- which(cooksd > threshold)
  
  cat("移除", length(influential_points), "个高影响点观测\n")
  
  data_clean <- data[-influential_points, ]
  
  return(data_clean)
}

# 使用示例
data_clean <- remove_influential_observations(data, model)
```

---

### `get_model_coefficients(model, p_value_threshold = 0.05)`

获取显著的模型系数

**参数**:
- `model` (lm 对象): 模型
- `p_value_threshold` (数值): p 值阈值

**返回值**: 数据框包含系数和 p 值

**示例**:
```r
get_model_coefficients <- function(model, p_value_threshold = 0.05) {
  coef_summary <- summary(model)$coefficients
  coef_df <- as.data.frame(coef_summary)
  colnames(coef_df) <- c("Estimate", "Std.Error", "t.value", "p.value")
  coef_df$Variable <- rownames(coef_df)
  rownames(coef_df) <- NULL
  
  # 标记显著性
  coef_df$Significant <- ifelse(coef_df$p.value < p_value_threshold, "Yes", "No")
  
  # 重新排列列
  coef_df <- coef_df[, c("Variable", "Estimate", "Std.Error", "t.value", "p.value", "Significant")]
  
  return(coef_df)
}

# 使用示例
coefficients <- get_model_coefficients(model)
print(coefficients)
```

---

### `compare_models(model_list, model_names = NULL)`

比较多个模型的性能

**参数**:
- `model_list` (列表): 模型对象列表
- `model_names` (字符向量): 模型名称

**返回值**: 比较结果数据框

**示例**:
```r
compare_models <- function(model_list, model_names = NULL) {
  if (is.null(model_names)) {
    model_names <- paste("Model", 1:length(model_list))
  }
  
  comparison <- data.frame(
    Model = model_names,
    R_squared = sapply(model_list, function(m) summary(m)$r.squared),
    Adj_R_squared = sapply(model_list, function(m) summary(m)$adj.r.squared),
    AIC = sapply(model_list, AIC),
    BIC = sapply(model_list, BIC),
    RMSE = sapply(model_list, function(m) sqrt(mean(residuals(m)^2)))
  )
  
  return(comparison)
}

# 使用示例
model1 <- lm(fare_amount ~ distance, data = data)
model2 <- lm(fare_amount ~ distance + passenger_count, data = data)
model3 <- lm(fare_amount ~ distance + passenger_count + hour, data = data)

comparison <- compare_models(list(model1, model2, model3), 
                             c("Simple", "Medium", "Complex"))
print(comparison)
```

---

### `export_predictions(predictions, actual, file_path)`

导出预测结果

**参数**:
- `predictions` (向量): 预测值
- `actual` (向量): 实际值
- `file_path` (字符串): 输出文件路径

**返回值**: 无（保存文件）

**示例**:
```r
export_predictions <- function(predictions, actual, file_path) {
  results <- data.frame(
    Actual = actual,
    Predicted = predictions,
    Residual = actual - predictions,
    Abs_Error = abs(actual - predictions),
    Pct_Error = abs((actual - predictions) / actual) * 100
  )
  
  write.csv(results, file = file_path, row.names = FALSE)
  cat("预测结果已导出至:", file_path, "\n")
}

# 使用示例
export_predictions(predictions, test_data$fare_amount, "predictions.csv")
```

---

### `create_model_report(model, data, output_file = "model_report.txt")`

生成模型报告

**参数**:
- `model` (lm 对象): 模型
- `data` (数据框): 数据
- `output_file` (字符串): 输出文件名

**返回值**: 无（保存报告）

**示例**:
```r
create_model_report <- function(model, data, output_file = "model_report.txt") {
  sink(output_file)
  
  cat("=" %R% 80, "\n")
  cat("模型分析报告\n")
  cat("生成时间:", as.character(Sys.time()), "\n")
  cat("=" %R% 80, "\n\n")
  
  cat("1. 模型摘要\n")
  cat("-" %R% 80, "\n")
  print(summary(model))
  
  cat("\n\n2. 模型系数\n")
  cat("-" %R% 80, "\n")
  print(get_model_coefficients(model))
  
  cat("\n\n3. 诊断检验\n")
  cat("-" %R% 80, "\n")
  
  cat("\n3.1 同方差性检验\n")
  print(bptest(model))
  
  cat("\n3.2 自相关检验\n")
  print(durbinWatsonTest(model))
  
  cat("\n3.3 正态性检验\n")
  print(ad.test(residuals(model)))
  
  cat("\n3.4 多重共线性检验 (VIF)\n")
  print(vif(model))
  
  cat("\n\n4. 影响点分析\n")
  cat("-" %R% 80, "\n")
  cooksd <- cooks.distance(model)
  threshold <- 20 / nrow(data)
  influential <- which(cooksd > threshold)
  cat("高影响点数量:", length(influential), "\n")
  
  sink()
  
  cat("报告已生成:", output_file, "\n")
}

# 使用示例
create_model_report(model, data, "uber_fare_model_report.txt")
```

---

## 完整工作流程示例

### 示例: 端到端分析流程

```r
# ==========================================
# Uber 票价预测完整分析流程
# ==========================================

# 加载所有必要的库
library(readxl)
library(ggplot2)
library(dplyr)
library(lubridate)
library(reshape2)
library(car)
library(lmtest)
library(nortest)

set.seed(123)

# ------------------------------------------
# 步骤 1: 数据加载和预处理
# ------------------------------------------
cat("步骤 1: 加载数据...\n")
data <- load_uber_data("sample_uber.xlsx")

cat("步骤 2: 清理数据...\n")
data <- clean_data(data, remove_na = TRUE, remove_outliers = FALSE)

# ------------------------------------------
# 步骤 2: 探索性数据分析
# ------------------------------------------
cat("步骤 3: 进行探索性数据分析...\n")

# 相关性分析
plot_correlation_heatmap(data)

# 分布分析
plot_distribution(data, "fare_amount", "Fare Amount Distribution")
plot_distribution(data, "distance", "Distance Distribution")

# 关系分析
plot_scatter(data, "distance", "fare_amount", "Fare vs Distance")

# ------------------------------------------
# 步骤 3: 特征工程
# ------------------------------------------
cat("步骤 4: 特征工程...\n")

# 添加时间特征
data <- add_time_features(data)
data <- add_daytime_feature(data)

# 地理聚类
data <- create_location_clusters(data, n_clusters = 5)

# 交互项
data <- add_interaction_terms(data, "distance", "daytime")

# 可视化聚类
plot_clusters(data, "pickup")
plot_clusters(data, "dropoff")

# ------------------------------------------
# 步骤 4: 数据分割
# ------------------------------------------
cat("步骤 5: 分割训练集和测试集...\n")
split_data <- split_train_test(data, train_ratio = 0.8)
train_data <- split_data$train
test_data <- split_data$test

# ------------------------------------------
# 步骤 5: 建立模型
# ------------------------------------------
cat("步骤 6: 建立初始模型...\n")
formula <- fare_amount ~ pickup_longitude + pickup_latitude + 
           dropoff_longitude + dropoff_latitude + distance + passenger_count +
           distance_daytime_interaction + 
           year + month + day + hour + 
           dropoff_cluster + pickup_cluster

model <- build_linear_model(train_data, formula)

# ------------------------------------------
# 步骤 6: 模型优化
# ------------------------------------------
cat("步骤 7: 逐步回归优化...\n")
stepwise_model <- stepwise_selection(model, direction = "backward")

# ------------------------------------------
# 步骤 7: 模型评估
# ------------------------------------------
cat("步骤 8: 评估模型性能...\n")
performance <- evaluate_model(stepwise_model, test_data)
print(performance)

# ------------------------------------------
# 步骤 8: 模型诊断
# ------------------------------------------
cat("步骤 9: 进行模型诊断...\n")
diagnostics <- perform_comprehensive_diagnostics(stepwise_model, train_data)

# ------------------------------------------
# 步骤 9: 修正措施（如需要）
# ------------------------------------------
if (length(diagnostics$influential_points) > 0) {
  cat("步骤 10: 应用修正措施...\n")
  
  # 移除高影响点
  train_data_clean <- remove_influential_observations(train_data, stepwise_model)
  
  # 应用转换
  train_data_clean <- apply_transformations(train_data_clean, 
                                            log_vars = c("fare_amount"),
                                            sqrt_vars = c("distance"))
  
  # 重新拟合
  formula_transformed <- log_fare_amount ~ pickup_longitude + dropoff_longitude + 
                         sqrt_distance + distance_daytime_interaction + 
                         year + month + pickup_cluster
  
  model_final <- build_linear_model(train_data_clean, formula_transformed)
  
  # 重新诊断
  plot_diagnostic_plots(model_final)
}

# ------------------------------------------
# 步骤 10: 生成报告
# ------------------------------------------
cat("步骤 11: 生成最终报告...\n")
create_model_report(stepwise_model, train_data, "final_model_report.txt")

# 导出系数
coefficients <- get_model_coefficients(stepwise_model)
write.csv(coefficients, "model_coefficients.csv", row.names = FALSE)

cat("\n分析完成！\n")
```

---

## 快速参考表

### 常用函数速查

| 功能类别 | 函数名 | 主要用途 |
|---------|--------|---------|
| **数据处理** | `load_uber_data()` | 加载 Uber 数据 |
| | `clean_data()` | 清理数据 |
| | `split_train_test()` | 分割数据集 |
| **可视化** | `plot_correlation_heatmap()` | 相关性热图 |
| | `plot_distribution()` | 密度分布图 |
| | `plot_scatter()` | 散点图 |
| | `plot_boxplot()` | 箱线图 |
| **特征工程** | `add_time_features()` | 添加时间特征 |
| | `create_location_clusters()` | 地理聚类 |
| | `add_interaction_terms()` | 交互项 |
| **建模** | `build_linear_model()` | 建立线性模型 |
| | `stepwise_selection()` | 逐步回归 |
| | `evaluate_model()` | 模型评估 |
| **诊断** | `test_homoscedasticity()` | 同方差性检验 |
| | `test_autocorrelation()` | 自相关检验 |
| | `test_normality()` | 正态性检验 |
| | `check_multicollinearity()` | 多重共线性检验 |
| | `identify_influential_points()` | 识别影响点 |

---

## 注意事项

1. **数据要求**: 所有函数假设输入数据格式正确且包含必要的列
2. **错误处理**: 建议在生产环境中添加更完善的错误处理
3. **性能**: 对于大型数据集，某些函数可能需要优化
4. **依赖**: 确保安装所有必要的 R 包

---

## 贡献和反馈

如有改进建议或发现问题，欢迎提交反馈。

**文档版本**: 1.0  
**最后更新**: 2025-11-21
