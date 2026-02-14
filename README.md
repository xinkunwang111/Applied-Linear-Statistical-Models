# Applied Linear Statistical Models

This project explores the application of multiple linear regression (MLR) using R to predict Uber fare prices, focusing on advanced statistical modeling techniques, exploratory data analysis (EDA), and regression diagnostics. The work demonstrates expertise in R programming for data analysis, feature engineering, and model optimization, aiming to derive actionable insights for business applications.

---

## 📚 完整文档

本项目提供了全面的技术文档，包括：

- **[文档索引](DOCUMENTATION_INDEX.md)** - 快速导航到所有文档
- **[API 文档](API_DOCUMENTATION.md)** - 完整的 API 和函数参考
- **[函数库](FUNCTION_LIBRARY.md)** - 可重用的函数库和代码示例
- **[用户指南](USER_GUIDE.md)** - 详细的使用教程和最佳实践

**快速开始**: 如果您是新用户，请先阅读 [用户指南的快速开始部分](USER_GUIDE.md#快速开始)。  

---

## Key Features of the Project  

### 1. Exploratory Data Analysis (EDA) and Feature Engineering  
- Utilized R for comprehensive EDA, including visualizing relationships between fare prices and key predictors like distance, time, and location.  
- Created engineered features such as:
  - **`distance_daytime_interaction`**: Captures the combined effect of trip distance and time of day on fare prices.  
  - **`pickup_clusters` and `dropoff_clusters`**: Applied K-means clustering to group geographic locations, uncovering fare concentration patterns.  
- Analyzed and visualized the effects of time variables (year, month, and hour) and passenger count on fare prices, providing deeper context for regression analysis.  

### 2. Linear Regression Implementation and Optimization  
- Built a multiple linear regression model in R to predict Uber fares, incorporating both original and engineered features.  
- Enhanced model performance by:
  - Applying **stepwise regression** to reduce complexity while retaining predictive accuracy.  
  - Achieving a high R² value of **0.7646**, indicating strong explanatory power for fare variation.  
- Conducted variable selection based on statistical significance (p-values) and multicollinearity checks using Variance Inflation Factor (VIF).  

### 3. Diagnostics and Remedial Measures  
- Verified regression assumptions (linearity, independence, homoscedasticity, and normality) using R diagnostic tools:
  - Residual plots, QQ-plots, and Durbin-Watson tests for independence.  
  - Breusch-Pagan test for homoscedasticity and VIF checks for multicollinearity.  
- Addressed issues such as:  
  - **High leverage points**: Removed influential data points to improve model stability.  
  - **Normality deviation**: Applied log transformation to the response variable (`fare_amount`) to correct skewed distributions.  

---

## Project Results  
- The optimized model explains **76.46% of the variation** in Uber fare prices, demonstrating the efficacy of feature engineering and diagnostics in improving regression models.  
- **Key findings**:  
  - Trip distance is the most significant predictor of fare prices.  
  - Fares exhibit geographic clustering, with specific pickup and drop-off locations linked to higher fares.  
  - Time variables (year, month, and daytime) significantly influence pricing, highlighting seasonal and temporal trends.  

---

## Insights and Applications  
### For Users:  
- Gain a better understanding of Uber's pricing rules to make cost-effective travel decisions, such as optimizing pickup/drop-off locations and ride times.  

### For Uber:  
- Leverage predictive insights to refine pricing strategies, optimize fleet allocation, and improve revenue forecasting by targeting high-demand times and locations.  

---

## Technical Skills Demonstrated  
- Proficient use of **R programming** for statistical analysis, visualization, and model development.  
- Advanced knowledge of feature engineering, regression modeling, and diagnostic testing.  
- Practical application of stepwise regression, clustering algorithms, and interaction terms in predictive modeling.  

This project exemplifies how statistical modeling can be combined with feature engineering and diagnostics to provide actionable insights for real-world applications. Feel free to explore the code and methodologies in this repository to learn more!  


