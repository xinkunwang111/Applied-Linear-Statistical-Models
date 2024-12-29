#install.packages("readxl")
#install.packages("ggplot2")
#install.packages("dplyr")
#install.packages("lubridate")
#install.packages("reshape2")
#install.packages("car")
#install.packages("lmtest")
#install.packages("nortest")


library(readxl)
library(ggplot2)
library(dplyr)
library(lubridate)
library(reshape2)
library(car)
library(lmtest)
library(nortest)
###################################### Dataset and EDA ##############################################
set.seed(123) 
# load the dataset
file_path <- "sample_uber.xlsx"
data <- read_excel(file_path)

# Extract new time-related variable
data <- data %>%
  mutate(
    pickup_datetime = ymd_hms(pickup_datetime),
    year = year(pickup_datetime),
    month = month(pickup_datetime),
    day = day(pickup_datetime),
    hour = hour(pickup_datetime)
  )

data <- subset(data, select = -c(Code, key))

# look through the data
head(data)
summary(data)

cor_matrix <- cor(data %>% select_if(is.numeric), use = "complete.obs")
melted_cor_matrix <- melt(cor_matrix)

# draw the heatmap
ggplot(data = melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       midpoint = 0, limit = c(-1, 1), space = "Lab",
                       name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1)) +
  coord_fixed() +
  labs(title = "Correlation Heatmap", x = "", y = "")


######################################## 1.2.3  Fare distribution ####################
ggplot(data, aes(x = fare_amount)) +
  geom_density(fill = "orange", alpha = 0.5) +  
  labs(title = "Density Plot of Trip Fare", x = "Fares", y = "Density") +
  theme_minimal()

#################################1.2.4 Distance distribution##############################################


# Distance distribution
ggplot(data, aes(x = distance)) +
  geom_density(fill = "orange", alpha = 0.5) +  
  labs(title = "Distance Distribution", x = "Distance", y = "Frequency") +
  theme_minimal()

# Plot the relation between fare and distance
ggplot(data, aes(x = distance, y = fare_amount)) +
  geom_point(alpha = 0.5, color = "blue") +
  labs(title = "Scatterplot of Fare Amount vs Distance", x = "Distance", y = "Fare Amount ($)") +
  theme_minimal()







###################1.2.5 The pickup  and dropoff locaiton ####################################################

# Plot the relation between fare and dropoff_location 
ggplot(data, aes(x = dropoff_longitude, y = dropoff_latitude, color = fare_amount)) +
  geom_point(alpha = 0.6) +
  scale_color_gradient(low = "blue", high = "red") +
  labs(title = "Scatterplot of Fare Amount by Dropoff Longitude and Latitude", x = "Dropoff Longitude", y = "Dropoff Latitude", color = "Fare Amount ($)") +
  theme_minimal()

# Plot the relation between fare and pickup_location 
ggplot(data, aes(x = pickup_longitude, y = pickup_latitude, color = fare_amount)) +
  geom_point(alpha = 0.6) +
  scale_color_gradient(low = "blue", high = "red") +
  labs(title = "Scatterplot of Fare Amount by Pickup Longitude and Latitude", x = "Pickup Longitude", y = "Pickup Latitude", color = "Fare Amount ($)") +
  theme_minimal() +
  coord_cartesian(xlim = c(min(data$pickup_longitude), max(data$pickup_longitude)),
                  ylim = c(min(data$pickup_latitude), max(data$pickup_latitude)))

# Adding  cluster using kmeans
pickup_data <- data.frame(data$pickup_longitude, data$pickup_latitude)
dropoff_data <- data.frame(data$dropoff_longitude, data$dropoff_latitude)

pickup_clusters <- kmeans(pickup_data, centers = 5)$cluster
dropoff_clusters <- kmeans(dropoff_data, centers = 5)$cluster

data <- data %>%
  mutate(
    dropoff_cluster = dropoff_clusters,
    pickup_cluster = pickup_clusters
  )
#vvisualize the cluster results
ggplot(data, aes(x = pickup_longitude, y = pickup_latitude, color = factor(pickup_cluster))) +
  geom_point(alpha = 0.5) +
  labs(title = "Pickup Locations Clustering", x = "Pickup Longitude", y = "Pickup Latitude", color = "Pickup Cluster") +
  theme_minimal()

ggplot(data, aes(x = dropoff_longitude, y = dropoff_latitude, color = factor(dropoff_cluster))) +
  geom_point(alpha = 0.5) +
  labs(title = "Dropoff Locations Clustering", x = "Dropoff Longitude", y = "Dropoff Latitude", color = "Dropoff Cluster") +
  theme_minimal()


###################################### 1.2.6 Time Analysis ##############################################
# set new daytime variable
data <- data %>%
  mutate(
    daytime = ifelse(hour >= 6 & hour < 20, 1, 0)
  )
   
daytime_distribution <- data %>%
  group_by(daytime) %>%
  summarise(order_count = n())

ggplot(data, aes(x = factor(daytime), y = fare_amount, fill = factor(daytime))) +
  geom_bar(stat = "summary", fun = "mean", color = "black", alpha = 0.7) +
  labs(title = "Average Fare Amount: Night vs Day", x = "Daytime (1: Day, 0: Night)", y = "Average Fare Amount ($)", fill = "Time of Day") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c("orange", "blue"), 
                    labels = c("Night", "Day"))


# Distance comparison: night and day using bar plot
ggplot(data, aes(x = factor(daytime), y = distance, fill = factor(daytime))) +
  geom_bar(stat = "summary", fun = "mean", color = "black", alpha = 0.7) +
  labs(title = "Average Distance: Night vs Day", x = "Daytime (1: Day, 0: Night)", y = "Average Distance") +
  theme_minimal() +theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c("green", "purple"),labels = c("Night", "Day"))

# relation between distance and  fare amount group by  daytime 
ggplot(data, aes(x = distance, y = fare_amount, color = factor(daytime))) +
  geom_point(alpha = 0.5) + geom_smooth(method = "lm", se = FALSE) +
  labs(color = "Daytime", x = "Distance", y = "Fare Amount",
       title = "Relationship between Distance and Fare mount group by Daytime" )




# Plot the relation between fare amount and year using a boxplot
data$year <- as.factor(data$year)
ggplot(data, aes(x = year, y = fare_amount)) +
  geom_boxplot(alpha = 0.7, fill = "blue", color = "black") +  # Create boxplot
  labs(title = "Boxplot of Fare Amount vs Year", 
       x = "Year", 
       y = "Fare Amount ($)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



# Plot the relation between fare amount and month using a boxplot
data$month <- factor(data$month, levels = 1:12)
ggplot(data, aes(x = month, y = fare_amount)) +
  geom_boxplot(alpha = 0.7, fill = "blue", color = "black") +  # Create boxplot
  labs(title = "Boxplot of Fare Amount vs Month", 
       x = "Month", 
       y = "Fare Amount ($)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

###################################### 1.2.7 The analysis of Passenger Count ##############################################
# Plot the relation between fare and passenger count using bar plot
ggplot(data, aes(x = factor(passenger_count), y = fare_amount, fill = factor(passenger_count))) +
  geom_bar(stat = "summary", fun = "mean", color = "black", alpha = 0.7) +
  labs(title = "Average Fare Amount vs Passenger Count", x = "Passenger Count", y = "Average Fare Amount ($)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_brewer(palette = "Blues")



###########################################linear Model ##############################################################
#Re-read the file and do the variable operation
file_path <- "sample_uber.xlsx"
data <- read_excel(file_path)

#put the adding variable into the data 
data <- data %>%
  mutate(
    pickup_datetime = ymd_hms(pickup_datetime),
    year = year(pickup_datetime),
    month = month(pickup_datetime),
    day = day(pickup_datetime),
    hour = hour(pickup_datetime),
    daytime = ifelse(hour >= 6 & hour < 20, 1, 0),
    distance_daytime_interaction = distance * daytime,
    dropoff_cluster = dropoff_clusters,
    pickup_cluster = pickup_clusters
      )
    
# Linear  Regression
model <- lm(fare_amount ~ pickup_longitude + pickup_latitude + 
              dropoff_longitude + dropoff_latitude+distance+passenger_count
            +distance_daytime_interaction 
            +year+month+day+hour
            + dropoff_clusters+pickup_clusters
              , data = data)
summary(model)


#use stepwise to remove variable
stepwise_model <- step(model, direction = "backward")
summary(stepwise_model)

#print the coefficient and p values
coefficients <- coef(stepwise_model)
p_values <- summary(stepwise_model)$coefficients[, 4]

print("Coefficients:")
print(coefficients)
print("P-values:")
print(p_values)



# 2.4 Judge the goodness-of-fit of the model  
r_squared <- summary(stepwise_model)$r.squared
adjusted_r_squared <- summary(stepwise_model)$adj.r.squared
print(paste("R-squared:", r_squared))
print(paste("Adjusted R-squared:", adjusted_r_squared))


####################################Diagnostics & Remedial Measures######################################
par(mfrow = c(2, 2))
plot(stepwise_model)
#check the linear regression model 

plot(stepwise_model,1)


#2.	Independence Assumption Check

# use Breusch-Pagan test
bptest(stepwise_model)

#3.  Independence Assumption Check 
plot(stepwise_model,3)
durbinWatsonTest(stepwise_model)

##4. Normality Assumption Check 
plot(stepwise_model,2)

ad.test(residuals(stepwise_model))


#5. No Multicollinearity Assumption

library(car)
vif_values <- vif(stepwise_model)
print(vif_values)


#Leverage points
par(mfrow = c(2, 2))
plot(stepwise_model)
plot(stepwise_model,4)




#Remedial Measures



# Calculate Cook's distance
cooksd<- cooks.distance(model)

# set the threshold
threshold <- 20 / nrow(data)

# Identify high influence points
high_influence_points <-which(cooksd>threshold)

# create a new dataset without high influence points
data_without_high_influence_points<-data[-high_influence_points, ]

# Apply log transformation to the dependent variable
data_without_high_influence_points$log_fare_amount <- log(data_without_high_influence_points$fare_amount)
data_without_high_influence_points$sqrt_distance <- sqrt(data_without_high_influence_points$distance)

# Refit the model with the transformed dependent variable
model_transformed <- lm(log_fare_amount ~ pickup_longitude +  
                              dropoff_longitude + sqrt_distance + distance_daytime_interaction + 
                              year + month + pickup_cluster, 
                            data = data_without_high_influence_points)


# Summary of the refitted model
summary(model_transformed)

# Diagnostic plots for the new model
par(mfrow = c(2, 2))
plot(model_transformed)
plot(model_transformed, which = 2)  # Q-Q plot
plot(model_transformed, which = 5)  # Scale-Location plot




