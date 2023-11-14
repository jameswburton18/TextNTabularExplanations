library(ggstatsplot)
library(tidyverse)
library(dplyr)
library(irr)

# Data creation
# Original csv was made in top_tokens_&_length_anlys.ipynb
#######################################
df <- read.csv("/home/james/CodingProjects/TextNTabularExplanations/models/shap_vals_instance_df.csv")
# Convert the method column to character type
df$method <- as.character(df$method)
# Rename the methods in your dataframe
df$method <- ifelse(df$method == "baseline", "All-Text (Unimodal)", df$method)
df$method <- ifelse(df$method == "all_text", "All-Text", df$method)
df$method <- ifelse(df$method == "ensemble_25", "WE (w=.25)", df$method)
df$method <- ifelse(df$method == "ensemble_50", "WE (w=.50)", df$method)
df$method <- ifelse(df$method == "ensemble_75", "WE (w=.75)", df$method)
df$method <- ifelse(df$method == "stack", "Stack", df$method)
df$method <- factor(df$method, levels = c(
  "All-Text (Unimodal)", "All-Text", "WE (w=.25)", "WE (w=.50)", "WE (w=.75)", "Stack"
))

df$text_model <- as.character(df$text_model)
df$text_model <- ifelse(df$text_model == "bert", "BERT", df$text_model)
df$text_model <- ifelse(df$text_model == "disbert", "DistilBERT", df$text_model)
df$text_model <- ifelse(df$text_model == "drob", "DistilRoBERTa", df$text_model)
df$text_model <- ifelse(df$text_model == "deberta", "DeBERTa", df$text_model)

# Filtering out the poorly performing models
df <- df %>%
  filter(!(ds_name == "channel" & (method == "All-Text" | method == "All-Text (Unimodal)"))) %>%
  filter(!(ds_name %in% c("salary", "wine") & method == "Stack")) %>%
  filter(!(ds_name == "prod_sent" & text_model == "DistilBERT" & method == "Stack"))

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
unique_values_list <- c("kick", "jigsaw", "wine", "fake", "imdb_genre", "channel", "airbnb", "salary", "prod_sent")
caption_list <- c(
  "kick", "jigsaw",
  "wine *Stack models excluded", "fake", "imdb_genre", "channel *All-Text (Unimodal and Multimodal) models excluded", "airbnb",
  "salary *Stack models excluded",
  "prod *DistilBERT-Stack model excluded"
)

features_df <- read.csv("/home/james/CodingProjects/TextNTabularExplanations/notebooks/unranked_df_no_template.csv")
# Convert the method column to character type
features_df$method <- as.character(features_df$method)
# Rename the methods in your dataframe
features_df$method <- ifelse(features_df$method == "baseline", "All-Text (Unimodal)", features_df$method)
features_df$method <- ifelse(features_df$method == "all_text", "All-Text", features_df$method)
features_df$method <- ifelse(features_df$method == "ensemble_25", "WE (w=.25)", features_df$method)
features_df$method <- ifelse(features_df$method == "ensemble_50", "WE (w=.50)", features_df$method)
features_df$method <- ifelse(features_df$method == "ensemble_75", "WE (w=.75)", features_df$method)
features_df$method <- ifelse(features_df$method == "stack", "Stack", features_df$method)
features_df$method <- factor(features_df$method, levels = c(
  "All-Text (Unimodal)", "All-Text", "WE (w=.25)", "WE (w=.50)", "WE (w=.75)", "Stack"
))

features_df$text_model <- as.character(features_df$text_model)
features_df$text_model <- ifelse(features_df$text_model == "bert", "BERT", features_df$text_model)
features_df$text_model <- ifelse(features_df$text_model == "disbert", "DistilBERT", features_df$text_model)
features_df$text_model <- ifelse(features_df$text_model == "drob", "DistilRoBERTa", features_df$text_model)
features_df$text_model <- ifelse(features_df$text_model == "deberta", "DeBERTa", features_df$text_model)


# Filtering out the poorly performing models
features_df <- features_df %>%
  filter(!(ds_name == "channel" & (method == "All-Text" | method == "All-Text (Unimodal)"))) %>%
  filter(!(ds_name %in% c("salary", "wine") & method == "Stack")) %>%
  filter(!(ds_name == "prod_sent" & text_model == "DistilBERT" & method == "Stack"))

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
        ylab = "Median(Text FI) - Median(Tabular FI)",
        xlab = "Text Model",
        title = "Are Text or Tabular Features Assigned More Importance?,
Difference in Median Feature Importance (SHAP), by Text Model",
        caption = paste("Dataset: ", caption),
      )

      # Save the plot to a PDF file
      file_name <- paste("/home/james/CodingProjects/TextNTabularExplanations/notebooks/images/R_plots/pdfs/text_tab_mod_", ds, ".pdf", sep = "")
      ggsave(file = file_name, plot = p, width = 8.5, height = 5)

      # save as jpeg
      file_name <- paste("/home/james/CodingProjects/TextNTabularExplanations/notebooks/images/R_plots/jpegs/text_tab_mod_", ds, ".jpeg", sep = "")
      ggsave(file = file_name, plot = p, width = 8.5, height = 5)
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
        ylab = "Median(Text FI) - Median(Tabular FI)",
        xlab = "Combination Method",
        title = "Are Text or Tabular Features Assigned More Importance?
Difference in Median Feature Importance (SHAP), by Combination Method",
        caption = paste("Dataset: ", caption),
      )
      # Save the plot to a PDF file
      file_name <- paste("/home/james/CodingProjects/TextNTabularExplanations/notebooks/images/R_plots/pdfs/text_tab_comb_", ds, ".pdf", sep = "")
      ggsave(file = file_name, plot = p, width = 8.5, height = 5)
      # print(p)
      file_name <- paste("/home/james/CodingProjects/TextNTabularExplanations/notebooks/images/R_plots/jpegs/text_tab_comb_", ds, ".jpeg", sep = "")
      ggsave(file = file_name, plot = p, width = 8.5, height = 5)
    },
    error = function(e) {
      print(paste("Error occurred for Dataset:", ds))
      print(e) # Print the error message for debugging
    }
  )
}, unique_values_list, caption_list)
# Do one for each experiment
Map(function(ds, caption) {
  Map(function(cm) {
    tryCatch(
      {
        data <- df_list[[ds]][df_list[[ds]]$method == cm, ]
        p <- ggbetweenstats(
          data = data,
          x = text_model,
          y = text.tab,
          type = "nonparametric", # ANOVA or Kruskal-Wallis
          # plot.type = "box",
          package = "ggsci",
          palette = "default_jco",
          pairwise.display = "ns",
          ylab = "Median(Text FI) - Median(Tabular FI)",
          xlab = "Text Model",
          title = "Are Text or Tabular Features Assigned More Importance?,
Difference in Median Feature Importance (SHAP), by Text Model",
          caption = paste("Dataset: ", ds, "Combination Method: ", cm),
        )

        # Save the plot to a PDF file
        file_name <- paste("/home/james/CodingProjects/TextNTabularExplanations/notebooks/images/R_plots/individual/pdfs/text_tab_mod_", ds, "_", cm, ".pdf", sep = "")
        ggsave(file = file_name, plot = p, width = 8.5, height = 5)

        # save as jpeg
        file_name <- paste("/home/james/CodingProjects/TextNTabularExplanations/notebooks/images/R_plots/individual/jpegs/text_tab_mod_", ds, "_", cm, ".jpeg", sep = "")
        ggsave(file = file_name, plot = p, width = 8.5, height = 5)
      },
      error = function(e) {
        print(paste("Error occurred for Dataset:", ds, "Combination Method:", cm))
        # print(e) # Print the error message for debugging
      }
    )
  }, unique(df_list[[ds]]$method))
}, unique_values_list, caption_list)

Map(function(ds, caption) {
  Map(function(tm) {
    tryCatch(
      {
        data <- df_list[[ds]][df_list[[ds]]$text_model == tm, ]
        p <- ggbetweenstats(
          data = data,
          x = method,
          y = text.tab,
          type = "nonparametric", # ANOVA or Kruskal-Wallis
          # plot.type = "box",
          package = "ggsci",
          palette = "default_jco",
          pairwise.display = "ns",
          ylab = "Median(Text FI) - Median(Tabular FI)",
          xlab = "Combination Method",
          title = "Are Text or Tabular Features Assigned More Importance?,
Difference in Median Feature Importance (SHAP), by Combination Method",
          caption = paste("Dataset: ", ds, "Text Model: ", tm),
        )

        # Save the plot to a PDF file
        file_name <- paste("/home/james/CodingProjects/TextNTabularExplanations/notebooks/images/R_plots/individual/pdfs/text_tab_method_", ds, "_", tm, ".pdf", sep = "")
        ggsave(file = file_name, plot = p, width = 8.5, height = 5)

        # save as jpeg
        file_name <- paste("/home/james/CodingProjects/TextNTabularExplanations/notebooks/images/R_plots/individual/jpegs/indiv_text_tab_method_", ds, "_", tm, ".jpeg", sep = "")
        ggsave(file = file_name, plot = p, width = 8.5, height = 5)
      },
      error = function(e) {
        print(paste("Error occurred for Dataset:", ds, "Text Model:", tm))
        # print(e) # Print the error message for debugging
      }
    )
  }, unique(df_list[[ds]]$text_model))
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
        ylab = "Feature Importance",
        xlab = "Text Model",
        title = "Are the Same Features Always the Most Important?
Feature Importance (SHAP) Compared, by Text Model",
        caption = paste("Dataset: ", caption),
      )
      # Save the plot to a PDF file
      file_name <- paste("/home/james/CodingProjects/TextNTabularExplanations/notebooks/images/R_plots/pdfs/ft_order_mod_", ds, ".pdf", sep = "")
      ggsave(file = file_name, plot = p, width = 8.5, height = 5)
      # print(p)
      file_name <- paste("/home/james/CodingProjects/TextNTabularExplanations/notebooks/images/R_plots/jpegs/ft_order_mod_", ds, ".jpeg", sep = "")
      ggsave(file = file_name, plot = p, width = 8.5, height = 5)
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
        ylab = "Feature Importance",
        xlab = "Combination Method",
        title = "Are the Same Features Always the Most Important?
Feature Importance (SHAP) Compared, by Combination Method",
        caption = paste("Dataset: ", caption),
      )
      # Save the plot to a PDF file
      file_name <- paste("/home/james/CodingProjects/TextNTabularExplanations/notebooks/images/R_plots/pdfs/ft_order_comb_", ds, ".pdf", sep = "")
      ggsave(file = file_name, plot = p, width = 8.5, height = 5)
      # print(p)
      file_name <- paste("/home/james/CodingProjects/TextNTabularExplanations/notebooks/images/R_plots/jpegs/ft_order_comb_", ds, ".jpeg", sep = "")
      ggsave(file = file_name, plot = p, width = 8.5, height = 5)
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

# Calculate pairwise Kendall tau
#################################################

# Create an empty list to store the pairwise Kendall values
method_kendall_values <- list()
model_kendall_values <- list()

# Break it up by both text model and method
for (ds in unique_values_list) {
  for (tm in unique_text_models) {
    df_pivot <- feature_df_list[[ds]] %>%
      filter(text_model == tm) %>%
      select(-text_model) %>%
      pivot_wider(names_from = method, values_from = feature_importance, names_prefix = "")

    # Get the list of treatments from the column names of df_pivot
    treatments <- colnames(df_pivot)[-1] # Exclude the first column (blocks)
    # Calculate Kendall correlation for each pair of treatments
    for (i in 1:(length(treatments) - 1)) {
      for (j in (i + 1):length(treatments)) {
        tryCatch(
          {
            treatment_A <- treatments[i]
            treatment_B <- treatments[j]

            # Select the columns corresponding to the two treatments
            df_pair <- df_pivot %>%
              select(feature_name, !!treatment_A, !!treatment_B)

            # Calculate Kendall correlation for the pair and store the result in the list
            kendall_value <- cor.test(df_pair[[2]], df_pair[[3]], method = "kendall")
            pair_name <- paste(ds, tm, treatment_A, treatment_B, sep = ",")
            method_kendall_values[[pair_name]] <- unname(kendall_value$estimate)
          },
          error = function(e) {
            print(paste("Error occurred for Dataset:", ds, "Method:", m))
            print(e) # Print the error message for debugging
            # Append NA if there is an error
            pair_name <- paste(ds, tm, treatment_A, treatment_B, sep = ",")
            method_kendall_values[[pair_name]] <- NA
          }
        )
      }
    }
  }
  for (m in unique_methods) {
    df_pivot <- feature_df_list[[ds]] %>%
      filter(method == m) %>%
      select(-method) %>%
      pivot_wider(names_from = text_model, values_from = feature_importance, names_prefix = "")

    # Get the list of treatments from the column names of df_pivot
    treatments <- colnames(df_pivot)[-1] # Exclude the first column (blocks)
    # Calculate Kendall correlation for each pair of treatments
    for (i in 1:(length(treatments) - 1)) {
      for (j in (i + 1):length(treatments)) {
        tryCatch(
          {
            treatment_A <- treatments[i]
            treatment_B <- treatments[j]

            # Select the columns corresponding to the two treatments
            df_pair <- df_pivot %>%
              select(feature_name, !!treatment_A, !!treatment_B)

            # Calculate Kendall correlation for the pair and store the result in the list
            kendall_value <- cor.test(df_pair[[2]], df_pair[[3]], method = "kendall")
            pair_name <- paste(ds, m, treatment_A, treatment_B, sep = ",")
            model_kendall_values[[pair_name]] <- unname(kendall_value$estimate)
          },
          error = function(e) {
            print(paste("Error occurred for Dataset:", ds, "Method:", m))
            print(e) # Print the error message for debugging
            # Append NA if there is an error
            pair_name <- paste(ds, m, treatment_A, treatment_B, sep = ",")
            model_kendall_values[[pair_name]] <- NA
          }
        )
      }
    }
  }
}

write.csv(method_kendall_values, "/home/james/CodingProjects/TextNTabularExplanations/notebooks/method_kendall_values.csv", row.names = FALSE)
write.csv(model_kendall_values, "/home/james/CodingProjects/TextNTabularExplanations/notebooks/model_kendall_values.csv", row.names = FALSE)

# Testing
#################################################
# ds <- "kick"
# m <- "WE (w=25)"

# df_pivot <- feature_df_list[[ds]] %>%
#   filter(method == m) %>%
#   select(-method) %>%
#   pivot_wider(names_from = text_model, values_from = feature_importance, names_prefix = "")
# kendall_value <- kendall(df_pivot)$value

# # Get the list of treatments from the column names of df_pivot
# treatments <- colnames(df_pivot)[-1] # Exclude the first column (blocks)

# # Create an empty list to store the pairwise Kendall values
# kendall_values <- list()

# # Calculate Kendall correlation for each pair of treatments
# for (i in 1:(length(treatments) - 1)) {
#   for (j in (i + 1):length(treatments)) {
#     treatment_A <- treatments[i]
#     treatment_B <- treatments[j]

#     # Select the columns corresponding to the two treatments
#     df_pair <- df_pivot %>%
#       select(feature_name, !!treatment_A, !!treatment_B)

#     # Calculate Kendall correlation for the pair and store the result in the list
#     kendall_value <- cor.test(df_pair[[2]], df_pair[[3]], method = "kendall")
#     pair_name <- paste(ds, m, treatment_A, treatment_B, sep = ",")
#     kendall_values[[pair_name]] <- unname(kendall_value$estimate)
#   }
# }
# kendall(df_pair)$value
# k <- cor.test(df_pair[[2]], df_pair[[3]], method = "kendall")
# ke <- unname(k$statistic)


# sal_df <- df_list[["kick"]]
# grouped_ggbetweenstats(
#   data = sal_df,
#   x = method,
#   y = text.tab,
#   type = "nonparametric", # ANOVA or Kruskal-Wallis
#   grouping.var = text_model,
#   package = "ggsci",
#   palette = "default_jco",
#   pairwise.display = "ns",
#   ylab = "med(|Text F.I.|) - med(|Tabular F.I.|)",
#   xlab = "Text Model",
#   # title = "Are Text or Tabular Features Assigned More Importance?",
#   # caption = "Dataset: salary",
#   # conf.level = FALSE,
#   # pairwise.comparisons = FALSE,
#   # centrality.plotting = FALSE
#   centrality.point.args = list(size = 2, color = "darkred"),
#   # centrality.label.args = list(data=n),
#   centrality.path.args = list(linewidth = 1, color = "red", alpha = 0.5),
#   centrality.label.args = list(size = 2.5, nudge_x = 0.4, segment.linetype = 4),
#   # ggsignif.args = list(textsize = 2, tip_length = 0.01, na.rm = TRUE),
# )
ds <- "wine"
tm <- "DeBERTa"
data <- df_list[[ds]][df_list[[ds]]$text_model == tm, ]
# Get the median of the text.tab column, grouped by method
data %>%
  group_by(method) %>%
  summarise(median = median(text.tab)) %>%
  arrange(desc(median))

# Create a function to calculate the median for a given dataset
calculate_median <- function(dataset) {
  df <- df_list[[dataset]][df_list[[dataset]]$text_model == tm, ]
  median_value <- df %>%
    group_by(method) %>%
    summarise(median = median(text.tab)) %>%
    filter(!is.na(median)) # Remove rows with NA medians
  return(median_value)
}

# Apply the function to all datasets and store the results in a list
median_results <- lapply(sort(unique_values_list), calculate_median)

# Combine the results into a single data frame
combined_df <- bind_rows(median_results, .id = "Dataset")

# Pivot the data to get it in the desired format
wide_format <- combined_df %>%
  pivot_wider(names_from = "Dataset", values_from = "median")

# Reorder the columns in the desired way (optional)
# desired_column_order <- c("wine", "other_ds1", "other_ds2", ...)
# wide_format <- wide_format[, c("method", unique_values_list)]

# Print the resulting data frame
print(wide_format)
