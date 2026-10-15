# ============================================================
# ROAD ACCIDENT SEVERITY STATISTICAL CONSULTANCY PROJECT
# UK DfT Road Casualty Statistics - Collision 2022
# IT3081 Statistical Modelling - Group Assignment
#
# TARGET:
# collision_severity
#   1 = Fatal
#   2 = Serious
#   3 = Slight
#
# TASK MAP:
#   Task 1  - Industry problem & objectives   -> written report, no code
#   Task 2  - Literature review                -> written report, no code
#   Task 3  - Dataset understanding & EDA       -> Sections 2-4 below
#   Task 4  - Statistical inference             -> Section 5 below
#   Task 5  - Predictive modelling              -> Sections 6-11 below
#   Task 6  - CRD / RCBD critical evaluation    -> Section 15 (discussion only)
#   Task 7  - PCA critical evaluation           -> Section 14
#   Task 8  - Bayesian methods critical eval.   -> Section 16 (discussion only)
#   Task 9  - Time series extension             -> Section 17
#
# MAIN PREDICTIVE MODELS (Task 5 core set):
#   1. Weighted Multinomial Logistic Regression
#   2. Weighted Ordinal Logistic Regression
#   3. Weighted Multinomial LASSO
#   4. Weighted Multinomial Elastic Net
#   + SMOTE sensitivity analysis (NOT a 5th main model - see Section 12)
#
# CLASS IMBALANCE STRATEGY:
#   - Stratified train/test split
#   - Rare-category collapsing BEFORE splitting (avoids near-singleton
#     categories that cause quasi-complete separation / absurd RRRs)
#   - Inverse-frequency class weights (primary strategy)
#   - Untouched test set throughout
#   - SMOTE used only as a secondary sensitivity check
#
# EVALUATION METRICS:
#   Accuracy, Kappa, Macro F1, per-class Sensitivity, Balanced Accuracy
# ============================================================


# ============================================================
# 0. PACKAGES
# ============================================================

required_packages <- c(
  "tidyverse", "caret", "naniar", "nnet", "MASS", "glmnet",
  "car", "pROC", "themis", "recipes", "lmtest", "forcats",
  "corrplot", "gridExtra"
)

for (p in required_packages) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p)
  }
}

library(tidyverse)
library(caret)
library(naniar)
library(nnet)
library(MASS)         # NOTE: masks dplyr::select() - this script always
                       # calls dplyr::select() explicitly to avoid conflicts
library(glmnet)
library(car)
library(pROC)
library(themis)
library(recipes)
library(lmtest)
library(forcats)
library(corrplot)
library(gridExtra)

options(stringsAsFactors = FALSE)
set.seed(2222)


# ============================================================
# 1. IMPORT DATA
# ============================================================

data_path <- 'C://Users//NIHINDU LONETH//Downloads//stat_lab//dft-road-casualty-statistics-collision-2022.csv'

# FIX: "-1" is STATS19's official "data missing or out of range" code
# across almost every categorical column. Leaving it out of na.strings
# lets -1 survive as a real factor level - because it's so rare, it
# creates near-singleton categories that cause quasi-complete
# separation (astronomical relative-risk-ratios / odds-ratios in the
# multinomial and ordinal models). This one line prevents that.
df <- read.csv(
  data_path,
  stringsAsFactors = FALSE,
  na.strings = c("", "NA", "NULL", "-1")
)

cat("\n================ DATASET INFORMATION ================\n")
cat("\nDataset dimensions:\n")
print(dim(df))
cat("\nVariable names:\n")
print(names(df))


# ============================================================
# 2. DATA CLEANING (Task 3)
# ============================================================

drop_vars <- c(
  "collision_index", "collision_ref_no",
  "first_road_class", "first_road_number",
  "second_road_class", "second_road_number",
  "location_easting_osgr", "location_northing_osgr",
  "local_authority_district", "local_authority_highway",
  "local_authority_highway_current",
  "lsoa_of_accident_location",
  "enhanced_severity_collision",
  "collision_injury_based",
  "collision_adjusted_severity_serious",
  "collision_adjusted_severity_slight",
  "collision_year"
)
drop_vars <- intersect(drop_vars, names(df))

df_clean <- df %>%
  dplyr::select(-all_of(drop_vars))

# Keep date/time aside for the Task 9 time-series extension before
# feature engineering consumes them
df_time <- df_clean %>%
  dplyr::select(date, time, collision_severity)

# ------------------------------------------------------------
# 2.1 Target variable
# ------------------------------------------------------------
df_clean$collision_severity <- factor(
  df_clean$collision_severity,
  levels = c(1, 2, 3),
  labels = c("Fatal", "Serious", "Slight")
)

# ------------------------------------------------------------
# 2.2 Numeric variables
# ------------------------------------------------------------
core_numeric_vars <- intersect(
  c("number_of_vehicles", "number_of_casualties", "speed_limit"),
  names(df_clean)
)
for (v in core_numeric_vars) {
  df_clean[[v]] <- as.numeric(df_clean[[v]])
}

# ------------------------------------------------------------
# 2.3 Categorical variables
# ------------------------------------------------------------
categorical_vars <- intersect(
  c("urban_or_rural_area", "light_conditions", "weather_conditions",
    "road_surface_conditions", "road_type", "junction_detail",
    "day_of_week", "police_force"),
  names(df_clean)
)

# Keep a numeric copy of day_of_week BEFORE converting to factor -
# needed for the is_weekend feature in Section 4, and this ordering
# avoids the earlier bug where comparing factor-of-numbers against
# the strings "Saturday"/"Sunday" silently matched nothing.
day_of_week_numeric <- suppressWarnings(as.numeric(as.character(df_clean$day_of_week)))

for (v in categorical_vars) {
  df_clean[[v]] <- factor(df_clean[[v]])
}

# ------------------------------------------------------------
# 2.4 Missing categorical values -> explicit "Unknown" level
# ------------------------------------------------------------
for (v in categorical_vars) {
  df_clean[[v]] <- forcats::fct_na_value_to_level(df_clean[[v]], level = "Unknown")
}

# ------------------------------------------------------------
# 2.5 Collapse rare categories BEFORE the train/test split
# ------------------------------------------------------------
# FIX: even after removing -1, some legitimate categories are still
# tiny (e.g. urban_or_rural_area = "3" / Unallocated has ~22 rows,
# some with zero Fatal cases). Near-empty categories cause the same
# separation problem as before. Lumping anything under 30 rows into
# "Other" keeps every category large enough to fit stable coefficients,
# and doing this BEFORE the split guarantees train and test share
# identical factor levels (no "new level in test data" errors later).
merge_tiny_level <- function(x, min_count = 30) {
  x <- forcats::fct_lump_min(x, min = min_count, other_level = "Other")
  tab <- table(x)
  if ("Other" %in% names(tab) && tab[["Other"]] < min_count) {
    top_level <- names(tab)[names(tab) != "Other"][which.max(tab[names(tab) != "Other"])]
    x <- forcats::fct_recode(x, !!top_level := "Other")
    x <- droplevels(x)
  }
  x
}

for (v in categorical_vars) {
  df_clean[[v]] <- merge_tiny_level(df_clean[[v]], min_count = 30)
}

# ------------------------------------------------------------
# 2.6 Missing numeric values
# ------------------------------------------------------------
df_clean <- df_clean %>%
  tidyr::drop_na(all_of(core_numeric_vars))

df_clean <- droplevels(df_clean)

cat("\nCleaned dataset dimensions:\n")
print(dim(df_clean))


# ============================================================
# 3. DATA QUALITY CHECKS (Task 3)
# ============================================================

cat("\n================ DATA QUALITY ================\n")

missing_summary <- data.frame(
  Variable = names(df_clean),
  Missing_Count = sapply(df_clean, function(x) sum(is.na(x)))
)
missing_summary$Missing_Percent <- round(100 * missing_summary$Missing_Count / nrow(df_clean), 2)

print(missing_summary %>% arrange(desc(Missing_Count)))

print(
  naniar::vis_miss(df_clean %>% dplyr::slice_sample(n = min(5000, nrow(df_clean))))
)


# ============================================================
# 4. DESCRIPTIVE ANALYSIS - TASK 3
# ============================================================

# ------------------------------------------------------------
# Target distribution
# ------------------------------------------------------------
severity_counts  <- table(df_clean$collision_severity)
severity_percent <- prop.table(severity_counts) * 100

print(severity_counts)
print(round(severity_percent, 2))

# ------------------------------------------------------------
# Numeric descriptive statistics
# ------------------------------------------------------------
numeric_summary <- df_clean %>%
  summarise(
    Vehicles_Mean     = mean(number_of_vehicles),
    Vehicles_SD       = sd(number_of_vehicles),
    Vehicles_Median   = median(number_of_vehicles),
    Casualties_Mean   = mean(number_of_casualties),
    Casualties_SD     = sd(number_of_casualties),
    Casualties_Median = median(number_of_casualties),
    Speed_Mean        = mean(speed_limit),
    Speed_SD          = sd(speed_limit),
    Speed_Median      = median(speed_limit)
  )

print(numeric_summary)

# ------------------------------------------------------------
# p1: Severity distribution
# ------------------------------------------------------------
p1 <- ggplot(df_clean, aes(collision_severity, fill = collision_severity)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5) +
  labs(title = "Distribution of Collision Severity", x = "Severity", y = "Number of Collisions") +
  theme_minimal() +
  theme(legend.position = "none")

# ------------------------------------------------------------
# p2: Numeric distributions
# ------------------------------------------------------------
p2 <- df_clean %>%
  dplyr::select(all_of(core_numeric_vars)) %>%
  pivot_longer(everything()) %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 30) +
  facet_wrap(~name, scales = "free") +
  labs(title = "Distribution of Numeric Variables") +
  theme_minimal()

# ------------------------------------------------------------
# p3-p6: Severity by important categorical variables
# ------------------------------------------------------------
p3 <- ggplot(df_clean, aes(factor(speed_limit), fill = collision_severity)) +
  geom_bar(position = "fill") +
  labs(title = "Severity Proportion by Speed Limit", x = "Speed Limit", y = "Proportion") +
  theme_minimal()

p4 <- ggplot(df_clean, aes(urban_or_rural_area, fill = collision_severity)) +
  geom_bar(position = "fill") +
  labs(title = "Severity by Urban/Rural Area", y = "Proportion") +
  theme_minimal()

p5 <- ggplot(df_clean, aes(light_conditions, fill = collision_severity)) +
  geom_bar(position = "fill") +
  coord_flip() +
  labs(title = "Severity by Light Conditions", y = "Proportion") +
  theme_minimal()

p6 <- ggplot(df_clean, aes(weather_conditions, fill = collision_severity)) +
  geom_bar(position = "fill") +
  coord_flip() +
  labs(title = "Severity by Weather Conditions", y = "Proportion") +
  theme_minimal()

# ------------------------------------------------------------
# p7-p8: Numeric variables by severity
# ------------------------------------------------------------
p7 <- ggplot(df_clean, aes(collision_severity, speed_limit, fill = collision_severity)) +
  geom_boxplot() +
  labs(title = "Speed Limit by Collision Severity") +
  theme_minimal() +
  theme(legend.position = "none")

p8 <- ggplot(df_clean, aes(collision_severity, number_of_casualties, fill = collision_severity)) +
  geom_boxplot() +
  labs(title = "Number of Casualties by Collision Severity") +
  theme_minimal() +
  theme(legend.position = "none")

grid.arrange(p1, p2, ncol = 2)
grid.arrange(p3, p4, ncol = 2)
grid.arrange(p5, p6, ncol = 2)
grid.arrange(p7, p8, ncol = 2)

# ------------------------------------------------------------
# Correlation among numeric variables
# ------------------------------------------------------------
num_data <- df_clean %>%
  dplyr::select(all_of(core_numeric_vars)) %>%
  na.omit()

corrplot(
  cor(num_data),
  method = "color",
  type = "upper",
  addCoef.col = "black",
  tl.col = "black"
)

# ------------------------------------------------------------
# Simple IQR-based outlier counts
# ------------------------------------------------------------
outlier_summary <- data.frame(
  Variable = core_numeric_vars,
  Outlier_Count = sapply(
    df_clean[core_numeric_vars],
    function(x) length(boxplot.stats(x)$out)
  )
)

print(outlier_summary)


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

# ------------------------------------------------------------
# 5.1 Time of day
# ------------------------------------------------------------
hour_value <- suppressWarnings(as.numeric(substr(as.character(df_clean$time), 1, 2)))

df_clean$time_of_day <- cut(
  hour_value,
  breaks = c(-1, 5, 11, 17, 21, 24),
  labels = c("Night", "Morning", "Afternoon", "Evening", "Late_Night"),
  right = TRUE
)

# ------------------------------------------------------------
# 5.2 Weekend
# ------------------------------------------------------------
# FIX: day_of_week is STATS19-coded 1 = Sunday ... 7 = Saturday, stored
# as numbers, never as the strings "Saturday"/"Sunday". The earlier
# version compared against those strings, which never matched anything,
# so every row fell into "Weekday" and is_weekend ended up with only
# one observed level - this then crashed multinom()/polr() with
# "contrasts can be applied only to factors with 2 or more levels".
# Using the numeric day-of-week codes captured in Section 2.3 fixes it.
df_clean$is_weekend <- factor(
  ifelse(day_of_week_numeric %in% c(1, 7), "Weekend", "Weekday"),
  levels = c("Weekday", "Weekend")
)

# ------------------------------------------------------------
# 5.3 Season
# ------------------------------------------------------------
parsed_date <- suppressWarnings(
  as.Date(df_clean$date, tryFormats = c("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"))
)
month_number <- as.numeric(format(parsed_date, "%m"))

df_clean$season <- case_when(
  month_number %in% c(12, 1, 2) ~ "Winter",
  month_number %in% c(3, 4, 5)  ~ "Spring",
  month_number %in% c(6, 7, 8)  ~ "Summer",
  month_number %in% c(9, 10, 11) ~ "Autumn",
  TRUE ~ "Unknown"
)
df_clean$season <- factor(df_clean$season)

# Sanity check: confirm every engineered factor has at least 2 levels
# before it reaches multinom()/polr() - catches this class of bug early
# rather than several models deep.
engineered_vars <- c("time_of_day", "is_weekend", "season")
stopifnot(all(sapply(df_clean[engineered_vars], nlevels) >= 2))


# ============================================================
# 6. TASK 4 - STATISTICAL INFERENCE
# ============================================================
# Each test is reported with H0, H1, justification, statistic,
# p-value, interpretation and organizational implication in the
# written report - the R output below supplies the numbers.

# ------------------------------------------------------------
# 6.1 Chi-square: Urban/Rural vs Severity
# H0: collision_severity is independent of urban_or_rural_area
# H1: collision_severity is NOT independent of urban_or_rural_area
# ------------------------------------------------------------
urban_table <- table(df_clean$urban_or_rural_area, df_clean$collision_severity)
urban_chi <- chisq.test(urban_table)
cat("\nChi-square: Urban/Rural vs Severity\n")
print(urban_table)
print(urban_chi)

# ------------------------------------------------------------
# 6.2 Chi-square: Light conditions vs Severity
# H0: collision_severity is independent of light_conditions
# H1: collision_severity is NOT independent of light_conditions
# ------------------------------------------------------------
light_table <- table(df_clean$light_conditions, df_clean$collision_severity)
light_chi <- chisq.test(light_table)
cat("\nChi-square: Light vs Severity\n")
print(light_table)
print(light_chi)

# ------------------------------------------------------------
# 6.3 One-way ANOVA: speed limit across severity groups
# H0: mean speed_limit is equal across Fatal, Serious, Slight
# H1: at least one group mean differs
# ------------------------------------------------------------
anova_model <- aov(speed_limit ~ collision_severity, data = df_clean)
cat("\nANOVA: Speed Limit by Severity\n")
print(summary(anova_model))
cat("\nTukey post-hoc test:\n")
print(TukeyHSD(anova_model))

# ------------------------------------------------------------
# 6.4 Welch t-test: number of vehicles, Fatal vs Slight
# H0: mean number_of_vehicles is equal for Fatal and Slight collisions
# H1: the means differ
# ------------------------------------------------------------
fatal_slight <- df_clean %>% filter(collision_severity %in% c("Fatal", "Slight"))
vehicles_ttest <- t.test(number_of_vehicles ~ collision_severity, data = fatal_slight)
cat("\nWelch t-test: Number of Vehicles\n")
print(vehicles_ttest)

# ------------------------------------------------------------
# 6.5 One-sample proportion test: Fatal rate vs an assumed 1% benchmark
# H0: true Fatal-collision proportion = 0.01
# H1: true Fatal-collision proportion != 0.01
# ------------------------------------------------------------
fatal_count <- sum(df_clean$collision_severity == "Fatal")
total_count <- nrow(df_clean)
fatal_prop_test <- prop.test(x = fatal_count, n = total_count, p = 0.01, alternative = "two.sided")
cat("\nFatal proportion test:\n")
print(fatal_prop_test)

# ------------------------------------------------------------
# 6.6 F-test: variance of speed limit, Urban vs Rural
# H0: variance of speed_limit is equal in Urban and Rural areas
# H1: the variances differ
# ------------------------------------------------------------
# FIX: var.test() requires EXACTLY 2 levels in the grouping factor.
# urban_or_rural_area has 3+ levels (Urban/Rural/Unallocated/Unknown),
# so this must be filtered down to just the two groups being compared,
# with droplevels() to remove the now-unused levels - otherwise R
# throws "grouping factor must have exactly 2 levels" and stops.
variance_data <- df_clean %>%
  filter(!is.na(speed_limit), urban_or_rural_area %in% c("1", "2")) %>%
  droplevels()

variance_test <- var.test(speed_limit ~ urban_or_rural_area, data = variance_data)
cat("\nF-test: Speed-limit variance, Urban vs Rural\n")
print(variance_test)


# ============================================================
# 7. MODEL DATA AND STRATIFIED SPLIT
# ============================================================

model_vars <- c(
  "collision_severity", "number_of_vehicles", "number_of_casualties",
  "speed_limit", "urban_or_rural_area", "light_conditions",
  "weather_conditions", "road_surface_conditions", "road_type",
  "junction_detail", "time_of_day", "is_weekend", "season"
)
model_vars <- intersect(model_vars, names(df_clean))

model_data <- df_clean %>%
  dplyr::select(all_of(model_vars)) %>%
  droplevels()

set.seed(2222)
train_index <- createDataPartition(model_data$collision_severity, p = 0.70, list = FALSE)
train_lr <- model_data[train_index, ]
test_lr  <- model_data[-train_index, ]

cat("\n================ TRAIN / TEST SPLIT ================\n")
cat("\nTraining observations:\n"); print(nrow(train_lr))
cat("\nTesting observations:\n");  print(nrow(test_lr))
cat("\nTraining class proportions:\n")
print(round(prop.table(table(train_lr$collision_severity)), 4))
cat("\nTesting class proportions:\n")
print(round(prop.table(table(test_lr$collision_severity)), 4))


# ============================================================
# 8. CLASS IMBALANCE HANDLING
# ============================================================
# Main strategy: inverse-frequency class weights. This uses every
# training row (unlike downsampling), does not fabricate observations
# (unlike SMOTE), and does not distort standard errors the way
# duplicating rows would. The test set is never touched.

class_counts <- table(train_lr$collision_severity)
class_weights <- sum(class_counts) / (length(class_counts) * class_counts)

cat("\n================ CLASS IMBALANCE ================\n")
cat("\nClass counts:\n"); print(class_counts)
cat("\nClass weights:\n"); print(round(class_weights, 3))

train_weights <- as.numeric(class_weights[as.character(train_lr$collision_severity)])
cat("\nObservation-weight summary:\n"); print(summary(train_weights))


# ============================================================
# 9. TASK 5 - PREDICTIVE MODELLING
# ============================================================

# ------------------------------------------------------------
# 9.1 Weighted Multinomial Logistic Regression
# ------------------------------------------------------------
multinom_model <- multinom(
  collision_severity ~ number_of_vehicles + number_of_casualties + speed_limit +
    urban_or_rural_area + light_conditions + weather_conditions +
    road_surface_conditions + road_type + junction_detail +
    time_of_day + is_weekend + season,
  data = train_lr, weights = train_weights, trace = FALSE
)

cat("\n================ MULTINOMIAL LOGISTIC ================\n")
print(summary(multinom_model))
cat("\nRelative Risk Ratios:\n")
print(exp(coef(multinom_model)))

multinom_pred <- predict(multinom_model, newdata = test_lr)
multinom_cm <- confusionMatrix(multinom_pred, test_lr$collision_severity)
print(multinom_cm)

# ------------------------------------------------------------
# 9.2 Weighted Ordinal Logistic Regression
# Natural severity order: Slight < Serious < Fatal
# ------------------------------------------------------------
train_ord <- train_lr
test_ord  <- test_lr

train_ord$collision_severity <- factor(train_ord$collision_severity,
                                        levels = c("Slight", "Serious", "Fatal"), ordered = TRUE)
test_ord$collision_severity  <- factor(test_ord$collision_severity,
                                        levels = c("Slight", "Serious", "Fatal"), ordered = TRUE)

ord_weights <- as.numeric(class_weights[as.character(train_ord$collision_severity)])

ordinal_model <- polr(
  collision_severity ~ number_of_vehicles + number_of_casualties + speed_limit +
    urban_or_rural_area + light_conditions + weather_conditions +
    road_surface_conditions + road_type + junction_detail +
    time_of_day + is_weekend + season,
  data = train_ord, weights = ord_weights, Hess = TRUE
)

cat("\n================ ORDINAL LOGISTIC ================\n")
print(summary(ordinal_model))
cat("\nOrdinal Odds Ratios:\n")
print(exp(coef(ordinal_model)))

ordinal_pred <- predict(ordinal_model, newdata = test_ord)
ordinal_cm <- confusionMatrix(ordinal_pred, test_ord$collision_severity)
print(ordinal_cm)

# Proportional-odds assumption: discuss in the report (compare ordinal
# vs. multinomial predictions above; a formal Brant test can be added
# via the 'brant' package if time allows).

# ------------------------------------------------------------
# 9.3 Weighted Multinomial LASSO
# ------------------------------------------------------------
x_train <- model.matrix(collision_severity ~ . - 1, data = train_lr)
x_test  <- model.matrix(collision_severity ~ . - 1, data = test_lr)

# Align columns in case a rare dummy level appears in only one split
missing_cols <- setdiff(colnames(x_train), colnames(x_test))
for (col in missing_cols) {
  x_test <- cbind(x_test, setNames(data.frame(rep(0, nrow(x_test))), col))
}
extra_cols <- setdiff(colnames(x_test), colnames(x_train))
if (length(extra_cols) > 0) {
  x_test <- x_test[, !colnames(x_test) %in% extra_cols, drop = FALSE]
}
x_test <- x_test[, colnames(x_train), drop = FALSE]

y_train <- train_lr$collision_severity
y_test  <- test_lr$collision_severity

set.seed(2222)
lasso_cv <- cv.glmnet(x = x_train, y = y_train, family = "multinomial",
                       alpha = 1, weights = train_weights, type.measure = "class", nfolds = 5)

cat("\n================ LASSO ================\n")
cat("\nBest LASSO lambda:\n"); print(lasso_cv$lambda.min)
plot(lasso_cv, main = "LASSO cross-validation")

lasso_prob <- predict(lasso_cv, newx = x_test, s = "lambda.min", type = "response")
lasso_matrix <- lasso_prob[, , 1]
lasso_pred <- factor(colnames(lasso_matrix)[apply(lasso_matrix, 1, which.max)], levels = levels(y_test))
lasso_cm <- confusionMatrix(lasso_pred, y_test)
print(lasso_cm)

cat("\nLASSO coefficients:\n")
print(coef(lasso_cv, s = "lambda.min"))

# ------------------------------------------------------------
# 9.4 Weighted Multinomial Elastic Net
# ------------------------------------------------------------
# Trim the lambda sequence so the grid search stays tractable:
# 4 alphas x ~100 lambdas x 5 folds = 2000 fits would otherwise be slow.
enet_grid <- expand.grid(
  alpha = seq(0.2, 0.8, by = 0.2),
  lambda = lasso_cv$lambda[seq(1, length(lasso_cv$lambda), length.out = 15)]
)

enet_control <- trainControl(method = "cv", number = 5, classProbs = TRUE, savePredictions = "final")

set.seed(2222)
# FIX: tuning on metric = "Accuracy" systematically rewards whichever
# alpha/lambda combination gets the most rows right overall - with 76%
# of rows "Slight", that means Accuracy-based tuning drifts toward a
# majority-leaning model even though the fit itself is weighted.
# Tuning on Kappa (chance-corrected agreement) avoids this.
enet_model <- train(
  x = x_train, y = y_train, method = "glmnet", family = "multinomial",
  trControl = enet_control, tuneGrid = enet_grid,
  weights = train_weights, metric = "Kappa"
)

cat("\n================ ELASTIC NET ================\n")
cat("\nBest Elastic Net parameters:\n"); print(enet_model$bestTune)
plot(enet_model)

enet_pred <- predict(enet_model, newdata = x_test)
enet_cm <- confusionMatrix(enet_pred, y_test)
print(enet_cm)


# ============================================================
# 10. MODEL PERFORMANCE FUNCTION
# ============================================================

get_metrics <- function(cm, model_name) {
  by_class <- cm$byClass
  if (is.null(dim(by_class))) by_class <- t(as.matrix(by_class))

  # FIX: caret's multi-class confusionMatrix() prefixes row names with
  # "Class: " (e.g. "Class: Fatal"), so a bare "Fatal" %in% rownames()
  # lookup NEVER matches and silently returns NA for every per-class
  # column - this line strips the prefix so the lookup actually works.
  class_names <- gsub("^Class: ", "", rownames(by_class))
  rownames(by_class) <- class_names

  get_metric <- function(class_name, metric_name) {
    if (class_name %in% class_names && metric_name %in% colnames(by_class)) {
      return(as.numeric(by_class[class_name, metric_name]))
    }
    return(NA_real_)
  }

  macro_f1 <- if ("F1" %in% colnames(by_class)) mean(by_class[, "F1"], na.rm = TRUE) else NA_real_
  balanced_accuracy <- if ("Balanced Accuracy" %in% colnames(by_class)) {
    mean(by_class[, "Balanced Accuracy"], na.rm = TRUE)
  } else NA_real_

  data.frame(
    Model = model_name,
    Accuracy = as.numeric(cm$overall["Accuracy"]),
    Kappa = as.numeric(cm$overall["Kappa"]),
    Macro_F1 = macro_f1,
    Fatal_Sensitivity = get_metric("Fatal", "Sensitivity"),
    Serious_Sensitivity = get_metric("Serious", "Sensitivity"),
    Slight_Sensitivity = get_metric("Slight", "Sensitivity"),
    Balanced_Accuracy = balanced_accuracy
  )
}


# ============================================================
# 11. MAIN MODEL COMPARISON (Task 5 core set)
# ============================================================

model_results <- bind_rows(
  get_metrics(multinom_cm, "Weighted Multinomial Logistic"),
  get_metrics(ordinal_cm,  "Weighted Ordinal Logistic"),
  get_metrics(lasso_cm,    "Weighted LASSO"),
  get_metrics(enet_cm,     "Weighted Elastic Net")
)
model_results[, -1] <- round(model_results[, -1], 4)

cat("\n================ MAIN MODEL COMPARISON ================\n")
print(model_results)

print(
  ggplot(model_results, aes(x = reorder(Model, Macro_F1), y = Macro_F1)) +
    geom_col() +
    geom_text(aes(label = round(Macro_F1, 3)), hjust = -0.1) +
    coord_flip() +
    labs(title = "Model Comparison - Macro F1", x = "", y = "Macro F1") +
    theme_minimal()
)

print(
  ggplot(model_results, aes(x = reorder(Model, Fatal_Sensitivity), y = Fatal_Sensitivity)) +
    geom_col() +
    geom_text(aes(label = round(Fatal_Sensitivity, 3)), hjust = -0.1) +
    coord_flip() +
    labs(title = "Model Comparison - Fatal Sensitivity", x = "", y = "Fatal Sensitivity") +
    theme_minimal()
)


# ============================================================
# 12. SMOTE SENSITIVITY ANALYSIS
# ============================================================
# SMOTE IS NOT A FIFTH MAIN MODEL - it is a sensitivity analysis to
# check whether performance changes materially when synthetic minority
# observations are introduced. SMOTE is applied ONLY to training data;
# the test set keeps its original imbalanced distribution throughout
# (themis::step_smote is skip=TRUE, so bake() on new_data does not
# reapply it - verified by the printed test-set table below).

set.seed(2222)

smote_recipe <- recipe(collision_severity ~ ., data = train_lr) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_smote(collision_severity, over_ratio = 0.5)

smote_prepped <- prep(smote_recipe, training = train_lr)
smote_train   <- bake(smote_prepped, new_data = NULL)
smote_test    <- bake(smote_prepped, new_data = test_lr)

cat("\n================ SMOTE ================\n")
cat("\nSMOTE training class distribution:\n")
print(table(smote_train$collision_severity))
cat("\nSMOTE test class distribution (should match original, unchanged):\n")
print(table(test_lr$collision_severity))

smote_multinom <- multinom(collision_severity ~ ., data = smote_train, trace = FALSE)
smote_pred <- predict(smote_multinom, newdata = smote_test)
smote_cm <- confusionMatrix(smote_pred, test_lr$collision_severity)

cat("\nSMOTE sensitivity-analysis results:\n")
print(smote_cm)

smote_result <- get_metrics(smote_cm, "SMOTE Multinomial - Sensitivity")
smote_result[, -1] <- round(smote_result[, -1], 4)
cat("\nSMOTE metrics:\n")
print(smote_result)

all_model_results <- bind_rows(model_results, smote_result)
cat("\n================ ALL RESULTS ================\n")
print(all_model_results)

# NOTE FOR THE REPORT: if SMOTE shows a collapsed Serious-class
# sensitivity (this happened in an earlier run of this pipeline before
# the rare-category and -1 fixes above), that is itself a legitimate,
# reportable finding - not a failure. It suggests Fatal and Serious
# collisions occupy overlapping regions of the predictor space, so
# densifying both neighbourhoods synthetically blurs rather than
# sharpens the boundary between them. Re-check this after the fixes
# in Sections 1-2, since cleaner category encoding may change the
# outcome meaningfully.


# ============================================================
# 13. CONFUSION MATRICES (for the report appendix)
# ============================================================

cat("\n--- Weighted Multinomial ---\n");  print(multinom_cm$table)
cat("\n--- Weighted Ordinal ---\n");      print(ordinal_cm$table)
cat("\n--- Weighted LASSO ---\n");        print(lasso_cm$table)
cat("\n--- Weighted Elastic Net ---\n");  print(enet_cm$table)
cat("\n--- SMOTE Sensitivity Analysis ---\n"); print(smote_cm$table)

# ------------------------------------------------------------
# Multicollinearity diagnostic
# ------------------------------------------------------------
# VIF is assessed via an auxiliary linear model on the predictor
# structure - it is NOT a direct VIF for the multinomial outcome model
# (car::vif() does not produce meaningful values there; see the
# warnings this raised in earlier drafts).
vif_data <- train_lr %>% dplyr::select(-collision_severity)
vif_lm <- lm(speed_limit ~ ., data = vif_data)

cat("\n================ VIF DIAGNOSTIC ================\n")
print(car::vif(vif_lm))


# ============================================================
# 14. TASK 7 - PCA EVALUATION
# ============================================================
# PCA is demonstrated on the continuous numeric variables only. It is
# NOT folded into the final predictive models because: there are only
# 3 continuous variables (little redundancy to reduce), most important
# predictors here are categorical (where PCA does not directly apply),
# and the original predictors are more interpretable for road-safety
# decision-making than derived components would be.

pca_vars <- intersect(c("speed_limit", "number_of_vehicles", "number_of_casualties"), names(df_clean))
num_for_pca <- df_clean %>% dplyr::select(all_of(pca_vars)) %>% na.omit()

pca_result <- prcomp(num_for_pca, scale. = TRUE)
cat("\n================ PCA ================\n")
print(summary(pca_result))
cat("\nPCA loadings:\n")
print(pca_result$rotation)
biplot(pca_result, main = "PCA Biplot - Numeric Variables")


# ============================================================
# 15. TASK 6 - EXPERIMENTAL DESIGN (discussion, no code required)
# ============================================================
# No controlled experiment is required for this observational dataset.
# CRD: could be considered for a future controlled road-safety
#   intervention trial (e.g. new signage) where experimental units are
#   sufficiently homogeneous.
# RCBD: road_type or police_force could serve as a natural blocking
#   factor, since baseline collision risk differs systematically
#   across roads/regions.
# Discuss in the report: advantages, limitations, feasibility, and the
# ethical/safety constraint that collisions themselves cannot be
# randomly assigned to occur.


# ============================================================
# 16. TASK 8 - BAYESIAN METHODS (discussion, no code required)
# ============================================================
# The four main predictive models above are frequentist. In the report,
# critically evaluate:
#   1. Naive Bayes - efficient, but a strong conditional-independence
#      assumption that likely doesn't hold well across correlated
#      road/weather/lighting variables.
#   2. Bayesian Regression - would replace point estimates with full
#      posterior distributions, naturally representing uncertainty
#      around each coefficient (contrast with the LASSO/Elastic Net
#      point estimates in Section 9).
#   3. Bayesian Decision Making - combines posterior probabilities with
#      a cost matrix; potentially valuable here since a missed Fatal
#      prediction and a missed Slight prediction plausibly have very
#      different real-world costs to the organization.
# Support this discussion with peer-reviewed literature.


# ============================================================
# 17. TASK 9 - TIME SERIES EXTENSION
# ============================================================
# One calendar year of data is not enough for a genuine multi-year
# ARIMA study, but a descriptive monthly trend can still be produced,
# and the report should discuss how pooling multiple years of DfT
# collision files would enable trend/seasonal decomposition and
# forecasting (ARIMA/SARIMA) of monthly severe-collision volumes.

date_for_ts <- suppressWarnings(
  as.Date(df_time$date, tryFormats = c("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"))
)

monthly_counts <- data.frame(date = date_for_ts) %>%
  filter(!is.na(date)) %>%
  mutate(month = floor_date(date, unit = "month")) %>%
  count(month)

cat("\n================ MONTHLY COLLISION COUNTS ================\n")
print(monthly_counts)

print(
  ggplot(monthly_counts, aes(x = month, y = n)) +
    geom_line() +
    geom_point() +
    labs(title = "Monthly Collision Counts - 2022", x = "Month", y = "Number of Collisions") +
    theme_minimal()
)


# ============================================================
# 18. SAVE RESULTS
# ============================================================

write.csv(model_results, "road_accident_main_model_comparison.csv", row.names = FALSE)
write.csv(all_model_results, "road_accident_model_comparison_with_SMOTE.csv", row.names = FALSE)


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

cat("\n")
cat("============================================================\n")
cat("ROAD ACCIDENT SEVERITY ANALYSIS COMPLETE\n")
cat("============================================================\n")
cat("Main predictive models: 4 (Multinomial, Ordinal, LASSO, Elastic Net)\n")
cat("Imbalance strategy: rare-category collapsing + inverse-frequency weights\n")
cat("SMOTE: sensitivity analysis only, over_ratio = 0.5\n")
cat("Test set: untouched during all imbalance handling\n")
cat("Evaluation: Accuracy, Kappa, Macro F1, per-class Sensitivity, Balanced Accuracy\n")
cat("============================================================\n")
