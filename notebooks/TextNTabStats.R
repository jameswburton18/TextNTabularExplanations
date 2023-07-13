library(ggstatsplot)
library(tidyverse)
# df <- read.csv("/home/james/CodingProjects/TextNTabularExplanations/notebooks/instance_df.csv")
df <- read.csv("/home/james/CodingProjects/TextNTabularExplanations/models/shap_vals_instance_df.csv")
sal <- subset(df, ds_name = salary)

# Create a list to store the separate dataframes
df_list <- list()

# Get the unique values of ds_name column
unique_values <- unique(df$ds_name)

# Loop through each unique value and create separate dataframes
for (value in unique_values) {
  df_subset <- df[df$ds_name == value, ]
  df_subset <- df_subset[, !names(df_subset) %in% "ds_name"] # Drop the ds_name column
  df_list[[value]] <- df_subset
}

# Creating a list of strings of unique values
unique_values_list <- as.character(unique_values)
ds <- 'imdb_genre'
ggwithinstats(
  data = df_list[[ds]],
  x = text_model,
  y = text.tab,
  type = "nonparametric", # ANOVA or Kruskal-Wallis
  # plot.type = "box",
  package = "ggsci",
  palette = "default_jco",
  pairwise.comparisons = TRUE,
  pairwise.display = "all",
  ylab = "med(|Text F.I.|) - med(|Tabular F.I.|)",
  xlab = "Combination Method",
  title = "Do Models Assign More Importance to Text or Tabular Features?
Difference in Median Absolute Feature Importance, by Combination Method",
  caption = paste("Dataset: ", ds),
  mean.plotting = FALSE,
)


ggbetweenstats(
  data = df_list[[ds]],
  x = method,
  y = text.tab,
  type = "nonparametric", # ANOVA or Kruskal-Wallis
  # plot.type = "box",
  package = "ggsci",
  palette = "default_jco",
  pairwise.comparisons = TRUE,
  pairwise.display = "ns",
  ylab = "med(|Text F.I.|) - med(|Tabular F.I.|)",
  xlab = "Combination Method",
  title = "Do Models Assign More Importance to Text or Tabular Features?
Difference in Median Absolute Feature Importance, by Combination Method",
  caption = paste("Dataset: ", ds),
  mean.plotting = FALSE,
)


for (ds in unique_values_list) {
  p <- ggwithinstats(
    data = df_list[[ds]],
    x = method,
    y = text.tab,
    type = "nonparametric", # ANOVA or Kruskal-Wallis
    # plot.type = "box",
    package = "ggsci",
    palette = "default_jco",
    pairwise.comparisons = TRUE,
    pairwise.display = "ns",
    ylab = "med(|Text F.I.|) - med(|Tabular F.I.|)",
    xlab = "Combination Method",
    title = "Do Models Assign More Importance to Text or Tabular Features?
Difference in Median Absolute Feature Importance, by Combination Method",
    caption = paste("Dataset: ", ds),
    mean.plotting = FALSE,
  )
  print(p)
}

for (ds in unique_values_list) {
  p <- ggbetweenstats(
    data = df_list[[ds]],
    x = text_model,
    y = text.tab,
    type = "nonparametric", # ANOVA or Kruskal-Wallis
    # plot.type = "box",
    package = "ggsci",
    palette = "default_jco",
    pairwise.comparisons = TRUE,
    pairwise.display = "ns",
    ylab = "med(|Text F.I.|) - med(|Tabular F.I.|)",
    xlab = "Combination Method",
    title = "Do Models Assign More Importance to Text or Tabular Features?
Difference in Median Absolute Feature Importance, by Text Model",
    caption = paste("Dataset: ", ds),
    mean.plotting = FALSE,
  )
  print(p)
}

features_df <- read.csv("/home/james/CodingProjects/TextNTabularExplanations/notebooks/unranked_df_no_template.csv")

feature_df_list <- list()
for (value in unique_values) {
  df_subset <- features_df[features_df$ds_name == value, ]
  df_subset <- df_subset[, !names(df_subset) %in% "ds_name"] # Drop the ds_name column
  feature_df_list[[value]] <- df_subset
}



for (ds in unique_values_list) {
  p <- ggwithinstats(
    data = feature_df_list[[ds]],
    x = text_model,
    y = feature_importance,
    type = "nonparametric", # ANOVA or Kruskal-Wallis
    # plot.type = "box",
    package = "ggsci",
    palette = "default_jco",
    pairwise.comparisons = TRUE,
    pairwise.display = "ns",
    ylab = "med(|Text F.I.|) - med(|Tabular F.I.|)",
    xlab = "Combination Method",
    title = "Do Models Assign More Importance to Text or Tabular Features?
Difference in Median Absolute Feature Importance, by Combination Method",
    caption = paste("Dataset: ", ds),
    mean.plotting = FALSE,
  )
  print(p)
}

for (ds in unique_values_list) {
  p <- ggwithinstats(
    data = feature_df_list[[ds]],
    x = method,
    y = feature_importance,
    type = "nonparametric", # ANOVA or Kruskal-Wallis
    # plot.type = "box",
    package = "ggsci",
    palette = "default_jco",
    pairwise.comparisons = TRUE,
    pairwise.display = "ns",
    ylab = "med(|Text F.I.|) - med(|Tabular F.I.|)",
    xlab = "Combination Method",
    title = "Do Models Assign More Importance to Text or Tabular Features?
Difference in Median Absolute Feature Importance, by Combination Method",
    caption = paste("Dataset: ", ds),
    mean.plotting = FALSE,
  )
  print(p)
}

