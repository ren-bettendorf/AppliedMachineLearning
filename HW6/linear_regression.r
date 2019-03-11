require(MASS)
setwd('C:/Users/Ren Bettendorf/Documents/AppliedMachineLearning/HW6/')
house_data <- read.table("data/house_data.txt", header=FALSE, col.names = c("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"))

# https://stats.stackexchange.com/questions/29477/how-to-write-a-linear-model-formula-with-100-variables-in-r
linear_model <- lm(MEDV ~ ., data = house_data)

# https://data.library.virginia.edu/diagnostic-plots/
par(mfrow=c(2,2))
plot(linear_model)
par(mfrow=c(1,1))

# Using 3 > p/n
# https://piazza.com/class/jqo7mhnnyiy698?cid=738
residual <- resid(linear_model)[cooks.distance(linear_model) > 2 * 13  / length(linear_model)]
named_residuals <- names(residual)
outliers <- as.numeric(named_residuals)

removed_outliers_data <- house_data[-outliers, ]
removed_outliers_model <- lm(MEDV ~ ., data = removed_outliers_data)

par(mfrow=c(2,2))
plot(removed_outliers_model)
par(mfrow=c(1,1))

# https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/boxcox.html
box_cox <- boxcox(removed_outliers_model, plotit = TRUE)
lambda <- box_cox$x[which.max(box_cox$y)]

# https://piazza.com/class/jqo7mhnnyiy698?cid=806
boxcox_model <- lm((MEDV ^ lambda - 1)/lambda ~ ., data = removed_outliers_data)

par(mfrow=c(2,2))
plot(boxcox_model)
par(mfrow=c(1,1))

# https://piazza.com/class/jqo7mhnnyiy698?cid=806
undo_transformation_model <- ((boxcox_model$fitted.values)*lambda)^(1/lambda)
plot(undo_transformation_model, removed_outliers_data$MEDV)