library(ggstatsplot)
library(tidyverse)
library(dplyr)
library(irr)

# Data creation
#######################################
df <- read.csv("/home/james/CodingProjects/TextNTabularExplanations/models/shap_vals_instance_df.csv")
# Convert the method column to character type
df$method <- as.character(df$method)
# Rename the methods in your dataframe
df$method <- ifelse(df$method == "baseline", "all_text_baseline", df$method)
df$method <- ifelse(df$method == "all_text", "all_text_corrected", df$method)
df$method <- factor(df$method, levels = c(
  "ensemble_25", "ensemble_50", "ensemble_75", "all_text_baseline", "all_text_corrected", "stack"
))

df$text_model <- as.character(df$text_model)
df$text_model <- ifelse(df$text_model == "bert", "BERT", df$text_model)
df$text_model <- ifelse(df$text_model == "disbert", "DistilBERT", df$text_model)
df$text_model <- ifelse(df$text_model == "drob", "DistilRoBERTa", df$text_model)
df$text_model <- ifelse(df$text_model == "deberta", "DeBERTa", df$text_model)

# Filtering out the poorly performing models
df <- df %>%
  filter(!(ds_name == "channel" & (method == "all_text_corrected" | method == "all_text_baseline"))) %>%
  filter(!(ds_name %in% c("salary", "wine") & method == "stack")) %>%
  filter(!(ds_name == "prod_sent" & text_model == "DistilBERT" & method == "stack"))

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
caption_list <- c(
  "kick", "jigsaw",
  "wine *stack models excluded", "fake", "imdb_genre", "channel *all_text_baseline and all_text_corrected models excluded", "airbnb",
  "salary *stack models excluded",
  "prod *Distilbert-stack model excluded"
)

features_df <- read.csv("/home/james/CodingProjects/TextNTabularExplanations/notebooks/unranked_df_no_template.csv")
# Convert the method column to character type
features_df$method <- as.character(features_df$method)
# Rename the methods in your dataframe
features_df$method <- ifelse(features_df$method == "baseline", "all_text_baseline", features_df$method)
features_df$method <- ifelse(features_df$method == "all_text", "all_text_corrected", features_df$method)
features_df$method <- factor(features_df$method, levels = c(
  "ensemble_25", "ensemble_50", "ensemble_75", "all_text_baseline", "all_text_corrected", "stack"
))

features_df$text_model <- as.character(features_df$text_model)
features_df$text_model <- ifelse(features_df$text_model == "bert", "BERT", features_df$text_model)
features_df$text_model <- ifelse(features_df$text_model == "disbert", "DistilBERT", features_df$text_model)
features_df$text_model <- ifelse(features_df$text_model == "drob", "DistilRoBERTa", features_df$text_model)
features_df$text_model <- ifelse(features_df$text_model == "deberta", "DeBERTa", features_df$text_model)


# Filtering out the poorly performing models
features_df <- features_df %>%
  filter(!(ds_name == "channel" & (method == "all_text_corrected" | method == "all_text_baseline"))) %>%
  filter(!(ds_name %in% c("salary", "wine") & method == "stack")) %>%
  filter(!(ds_name == "prod_sent" & text_model == "DistilBERT" & method == "stack"))

feature_df_list <- list()
for (value in unique_values) {
  df_subset <- features_df[features_df$ds_name == value, ]
  df_subset <- df_subset[, !names(df_subset) %in% "ds_name"] # Drop the ds_name column
  feature_df_list[[value]] <- df_subset
}

unique_text_models <- unique(features_df$text_model)
unique_methods <- unique(features_df$method)

# Produce charts
####################################################


Map(function(ds, caption) {
  tryCatch(
    {
      p <- ggbetweenstats(
        data = df_list[[ds]],
        x = text_model,
        y = text.tab,
        type = "nonparametric", # ANOVA or Kruskal-Wallis
        # plot.type = "box",
        package = "ggsci",
        palette = "default_jco",
        pairwise.display = "ns",
        ylab = "med(|Text F.I.|) - med(|Tabular F.I.|)",
        xlab = "Text Model",
        title = "Are Text or Tabular Features Assigned More Importance?,
Difference in Median Absolute Feature Importance (SHAP), by Combination Method",
        caption = paste("Dataset: ", caption),
      )

      # Save the plot to a PDF file
      file_name <- paste("/home/james/CodingProjects/TextNTabularExplanations/notebooks/images/R_plots/pdfs/text_tab_mod_", ds, ".pdf", sep = "")
      ggsave(file = file_name, plot = p, width = 8, height = 5)

      # save as jpeg
      file_name <- paste("/home/james/CodingProjects/TextNTabularExplanations/notebooks/images/R_plots/jpegs/text_tab_mod_", ds, ".jpeg", sep = "")
      ggsave(file = file_name, plot = p, width = 8, height = 5)
    },
    error = function(e) {
      print(paste("Error occurred for Dataset:", ds))
      print(e) # Print the error message for debugging
    }
  )
}, unique_values_list, caption_list)

Map(function(ds, caption) {
  tryCatch(
    {
      p <- ggbetweenstats(
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
        xlab = "Text Model",
        title = "Are Text or Tabular Features Assigned More Importance?
Difference in Median Absolute Feature Importance (SHAP), by Text Model",
        caption = paste("Dataset: ", caption),
      )
      # Save the plot to a PDF file
      file_name <- paste("/home/james/CodingProjects/TextNTabularExplanations/notebooks/images/R_plots/pdfs/text_tab_comb_", ds, ".pdf", sep = "")
      ggsave(file = file_name, plot = p, width = 8, height = 5)
      # print(p)
      file_name <- paste("/home/james/CodingProjects/TextNTabularExplanations/notebooks/images/R_plots/jpegs/text_tab_comb_", ds, ".jpeg", sep = "")
      ggsave(file = file_name, plot = p, width = 8, height = 5)
    },
    error = function(e) {
      print(paste("Error occurred for Dataset:", ds))
      print(e) # Print the error message for debugging
    }
  )
}, unique_values_list, caption_list)

Map(function(ds, caption) {
  tryCatch(
    {
      p <- ggwithinstats(
        data = feature_df_list[[ds]],
        x = text_model,
        y = feature_importance,
        type = "nonparametric", # ANOVA or Kruskal-Wallis
        # plot.type = "box",
        package = "ggsci",
        palette = "default_jco",
        pairwise.display = "sig",
        ylab = "|Feature Importance (SHAP)|",
        xlab = "Text Model",
        title = "Are the Same Features Always the Most Important?
Absolute Feature Importance (SHAP) Compared, by Text Model",
        caption = paste("Dataset: ", caption),
      )
      # Save the plot to a PDF file
      file_name <- paste("/home/james/CodingProjects/TextNTabularExplanations/notebooks/images/R_plots/pdfs/ft_order_mod_", ds, ".pdf", sep = "")
      ggsave(file = file_name, plot = p, width = 8, height = 5)
      # print(p)
      file_name <- paste("/home/james/CodingProjects/TextNTabularExplanations/notebooks/images/R_plots/jpegs/ft_order_mod_", ds, ".jpeg", sep = "")
      ggsave(file = file_name, plot = p, width = 8, height = 5)
    },
    error = function(e) {
      print(paste("Error occurred for Dataset:", ds))
      print(e) # Print the error message for debugging
    }
  )
}, unique_values_list, caption_list)

Map(function(ds, caption) {
  tryCatch(
    {
      p <- ggwithinstats(
        data = feature_df_list[[ds]],
        x = method,
        y = feature_importance,
        type = "nonparametric", # ANOVA or Kruskal-Wallis
        # plot.type = "box",
        package = "ggsci",
        palette = "default_jco",
        pairwise.display = "sig",
        ylab = "|Feature Importance (SHAP)|",
        xlab = "Combination Method",
        title = "Are the Same Features Always the Most Important?
Absolute Feature Importance (SHAP) Compared, by Combination Method",
        caption = paste("Dataset: ", caption),
      )
      # Save the plot to a PDF file
      file_name <- paste("/home/james/CodingProjects/TextNTabularExplanations/notebooks/images/R_plots/pdfs/ft_order_comb_", ds, ".pdf", sep = "")
      ggsave(file = file_name, plot = p, width = 8, height = 5)
      # print(p)
      file_name <- paste("/home/james/CodingProjects/TextNTabularExplanations/notebooks/images/R_plots/jpegs/ft_order_comb_", ds, ".jpeg", sep = "")
      ggsave(file = file_name, plot = p, width = 8, height = 5)
    },
    error = function(e) {
      print(paste("Error occurred for Dataset:", ds))
      print(e) # Print the error message for debugging
    }
  )
}, unique_values_list, caption_list)



# Take each dataset, model combo as unique, save results to csv
#################################################
# Create an empty data frame to store the results
tm_result_df <- data.frame(ds = character(), tm = character(), kendall_value = numeric(), stringsAsFactors = FALSE)
m_result_df <- data.frame(ds = character(), m = character(), kendall_value = numeric(), stringsAsFactors = FALSE)


# Break it up by both text model and method
for (ds in unique_values_list) {
  for (tm in unique_text_models) {
    tryCatch(
      {
        df_pivot <- feature_df_list[[ds]] %>%
          filter(text_model == tm) %>%
          select(-text_model) %>%
          pivot_wider(names_from = method, values_from = feature_importance, names_prefix = "")
        kendall_value <- kendall(df_pivot)$value

        # Append the result to the tm_result_df
        tm_result_df[nrow(tm_result_df) + 1, ] <- c(ds, tm, kendall_value)
      },
      error = function(e) {
        # print(paste("Error occurred for Dataset:", ds, "Text Model:", tm))
        # print(e) # Print the error message for debugging
        # Append NA if there is an error
        # tm_result_df[nrow(tm_result_df) + 1, ] <- c(ds, tm, NA)
      }
    )
  }
  for (m in unique_methods) {
    tryCatch(
      {
        df_pivot <- feature_df_list[[ds]] %>%
          filter(method == m) %>%
          select(-method) %>%
          pivot_wider(names_from = text_model, values_from = feature_importance, names_prefix = "")
        kendall_value <- kendall(df_pivot)$value

        # Append the result to the m_result_df
        m_result_df[nrow(m_result_df) + 1, ] <- c(ds, m, kendall_value)
      },
      error = function(e) {
        # print(paste("Error occurred for Dataset:", ds, "Method:", m))
        # print(e) # Print the error message for debugging
        # Append NA if there is an error
        # m_result_df[nrow(m_result_df) + 1, ] <- c(ds, m, NA)
      }
    )
  }
}

# Write the result_df to a CSV file
write.csv(tm_result_df, "/home/james/CodingProjects/TextNTabularExplanations/notebooks/text_model_kendall.csv", row.names = FALSE)
write.csv(m_result_df, "/home/james/CodingProjects/TextNTabularExplanations/notebooks/method_kendall.csv", row.names = FALSE)


sal_df <- df_list[["kick"]]
grouped_ggbetweenstats(
  data = sal_df,
  x = method,
  y = text.tab,
  type = "nonparametric", # ANOVA or Kruskal-Wallis
  grouping.var = text_model,
  package = "ggsci",
  palette = "default_jco",
  pairwise.display = "ns",
  ylab = "med(|Text F.I.|) - med(|Tabular F.I.|)",
  xlab = "Text Model",
  # title = "Are Text or Tabular Features Assigned More Importance?",
  # caption = "Dataset: salary",
  # conf.level = FALSE,
  # pairwise.comparisons = FALSE,
  # centrality.plotting = FALSE
  centrality.point.args = list(size = 2, color = "darkred"),
  # centrality.label.args = list(data=n),
  centrality.path.args = list(linewidth = 1, color = "red", alpha = 0.5),
  centrality.label.args = list(size = 2.5, nudge_x = 0.4, segment.linetype = 4),
  # ggsignif.args = list(textsize = 2, tip_length = 0.01, na.rm = TRUE),
)
