#### Libraries and functions ####
library(stringi)
library(ggplot2)
library(reshape2)
library(dplyr)

# Function to modify the data frame of performances into form that the plots can be drawn.
prepare_df_for_figures <- function(df_performances){
  # Melt the data frame.
  df_performances <- melt(df_performances,
                          id.vars = c("data", "model"),
                          variable.name = "test_perf_measure",
                          value.name = "test_performance")
  
  # Specify the factor levels and labels.
  df_performances$data <- factor(df_performances$data,
                                 levels = c("davis", "metz", "kiba",
                                            "merget", "GPCR", "IC", "E"),
                                 labels = c("Davis", "Metz", "KiBA",
                                            "Merget", "GPCR", "IC", "E"))
  df_performances$test_perf_measure <- factor(df_performances$test_perf_measure,
                                              levels = c("IC_index", "C_d_index",
                                                         "C_t_index", "C_index"))
  
  # Split a string variable to separate variables for the settings and methods.
  setting_locations <- stri_locate(df_performances$model, regex = "\'[IO]D[IO]T\'")
  df_performances["setting"] <- substr(df_performances$model,
                                       setting_locations[,1]+1,
                                       setting_locations[,2]-1)
  # These are for having the abbreviations written out in the figures.
  df_performances <- df_performances %>%
    mutate(drug = case_when(setting %in% c("IDIT", "IDOT") ~ "In-training-set\ndrugs",
                            TRUE ~ "Off-training-set\ndrugs"),
           target = case_when(setting %in% c("IDIT", "ODIT") ~ "In-training-set\ntargets",
                              TRUE ~ "Off-training-set\ntargets"))
  
  algorithm_locations <- stri_locate(df_performances$model, regex = "\\.[:graph:]*\'>")
  df_performances["algorithm"] <- substr(df_performances$model,
                                         algorithm_locations[,1]+1,
                                         algorithm_locations[,2]-2)
  # There are still problems with how the model information is saved for deep learning methods.
  # Let's fix those rows manually..
  GDTA_rows <- grep("GINConvNet", df_performances$model)
  FF_rows <- grep("feedforward", df_performances$model)
  df_performances[GDTA_rows,]$algorithm <- "GDTA"
  df_performances[FF_rows,]$algorithm <- "FFNN"
  # Find additional specifications determining the used method.
  algorithm_specification_locations <- stri_locate(df_performances$model, regex = "\\[.*\\]") # "\\}.*\","
  df_performances["specification"] <- substr(df_performances$model,
                                             algorithm_specification_locations[,1]+1,
                                             algorithm_specification_locations[,2]-1)

  
  # Modify the names of the methods to clearer versions.
  df_performances$algorithm[df_performances$specification == "'gaussian', 'gaussian', 'pko_kronecker'"] <-
    "KR (Gaussian)"
  df_performances$algorithm[df_performances$specification == "'gaussian', 'gaussian', 'pko_linear'"] <-
    "LR (Gaussian)"
  df_performances$algorithm[df_performances$specification == "'linear', 'linear', 'pko_kronecker'"] <-
    "KR (linear)"
  df_performances$algorithm[df_performances$specification == "'linear', 'linear', 'pko_linear'"] <-
    "LR (linear)"
  df_performances$algorithm[df_performances$algorithm == "ltr_cls"] <- "PR"
  df_performances$algorithm[df_performances$algorithm == "ensemble._forest.RandomForestRegressor"] <- "RF"
  df_performances$algorithm[df_performances$algorithm == "neighbors._regression.KNeighborsRegressor"] <- "kNN"
  df_performances$algorithm[df_performances$algorithm == "sklearn.XGBRegressor"] <- "XGB"
  df_performances$algorithm[df_performances$algorithm == "DeepDTA"] <- "DDTA"
  
  return(df_performances)
}

#### Draw the figures ####
# Read the predictions. Might have to modify the path if the file is not in 
# the same folder with this file.
performances <- read.csv("performances.csv")

performances_IC_C_Cd_Ct_summary <-
  prepare_df_for_figures(performances)

data_set_names <- c("Davis", "Metz", "KiBA", "Merget", "GPCR", "IC", "E")
for (dsn in data_set_names) {
  performances_IC_C_Cd_Ct_summary %>%
    subset(data == dsn) %>%
    ggplot() +
    # Draw the bar plot with algorithm on x-axis, test set performance on y-axis,
    # and a group of bars contains the bars for different test performance measures.
    geom_bar(mapping = aes(x = algorithm, y = test_performance-0.4,
                           group = test_perf_measure,
                           fill = test_perf_measure),
             position = "dodge", stat = "identity", width = 0.6) +
    # Highligth 0.5 as it corresponds to random prediction.
    geom_hline(yintercept = 0.5-0.4, alpha = 0.2) +
    expand_limits(y = 0.6) +
    # Create the subplots for the different settings.
    facet_grid(drug~target, scales = "fixed") +
    # Modify the theme of the plot. Here it is possible to also select a font class that
    # is used in the plot texts.
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.3),
          text = element_text(size = 8),
          legend.position = "top",
          legend.direction = "horizontal",
          legend.justification = "left") +
    
    # Modify how the legends are presented.
    guides(fill = guide_legend(label.position = "right",
                               title.position = "top"),
           linetype = guide_legend(label.position = "right",
                                   title.position = "top"),
           col = guide_legend(label.position = "right",
                              title.position = "top")) +
    # Change the labels to match that the y-axis was scaled so that the
    # columns start from 0.4 instead of 0.
    scale_y_continuous(name = NULL,
                       breaks = c(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                       labels = c(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) +
    
    # Modify the presentation of the performance measure names in the figure.
    scale_color_discrete(breaks = c("IC_index", "C_d_index", "C_t_index", "C_index"),
                         labels = c("IC-index", bquote(C[d]-index), 
                                    bquote(C[t]-index), "C-index")) +
    scale_linetype_discrete(breaks = c("IC_index", "C_d_index", "C_t_index", "C_index"),
                            labels = c("IC-index", bquote(C[d]-index), 
                                       bquote(C[t]-index), "C-index")) +
    scale_fill_discrete(breaks = c("IC_index", "C_d_index", "C_t_index", "C_index"),
                        labels = c("IC-index", bquote(C[d]-index), 
                                   bquote(C[t]-index), "C-index")) +
    # Modify the variable titles.
    labs(
      fill = "Performance measure",
      group = "Performance measure",
      x = NULL)
  ggsave(paste0("barplot_",dsn,".pdf"), height = 10, width = 15, units = "cm")
}
