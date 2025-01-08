setwd("~/Ristiinvalidointiprojekti/AssignmentIndex")
#### Libraries ####
library(stringi)
library(ggplot2)
library(ggpattern)
library(reshape2)
library(patchwork)
library(dplyr)
library(extrafont)
# loadfonts(device = "win")

#### Imbalanced XOR simulation ####
# Function to modify the data frame of performances into form that the plots can be drawn.
prepare_df_for_figures_simulation <- function(df_performances){
  # Melt the data frame.
  df_performances <- melt(df_performances, 
                          id.vars = c("model"),
                          variable.name = "test_perf_measure",
                          value.name = "test_performance")
  
  df_performances$test_perf_measure <- factor(df_performances$test_perf_measure,
                                              levels = c("A_index", "C_d_index",
                                                         "C_t_index", "C_index", "accuracy"))
  
  
  # Split a string variable to separate variables for setting, method and performance measure.
  setting_locations <- stri_locate(df_performances$model, regex = "\'S[1-4]\'")
  df_performances["setting"] <- substr(df_performances$model, 
                                       setting_locations[,1]+1, 
                                       setting_locations[,2]-1)
  df_performances <- df_performances %>% 
    mutate(drug = case_when(setting %in% c("S1", "S2") ~ "in",
                            TRUE ~ "off"),
           target = case_when(setting %in% c("S1", "S3") ~ "in",
                              TRUE ~ "off"))
  
  algorithm_locations <- stri_locate(df_performances$model, regex = "\\, \'[a-zA-Z_]*\'")
  df_performances["algorithm"] <- substr(df_performances$model, 
                                         algorithm_locations[,1]+3, 
                                         algorithm_locations[,2]-1)
  return(df_performances)
}

# Read the data.
performances_XOR_imbalance_0.1_0.2_ACCheaviside <- 
  read.csv("~/Ristiinvalidointiprojekti/AssignmentIndex/performances_XOR_imbalance_0.1_0.2_ACCheaviside.csv")

# Repetitions saved in the file in several parts, 
# so remove the heading rows in the middle of the data frame.
# performances_XOR_imbalance_0.1_0.2_ACCheaviside <- 
#   performances_XOR_imbalance_0.1_0.2_ACCheaviside[-which(
#     performances_XOR_imbalance_0.1_0.2_ACCheaviside$random_seed == "random_seed"),]

# Calculate the mean values and credible intervals,
performances_XOR_imbalance_summary <- 
  prepare_df_for_figures_simulation(performances_XOR_imbalance_0.1_0.2_ACCheaviside[,-1]) %>% 
  mutate(algorithm = factor(algorithm, 
                            levels = c("GS", "DS", "TS", "SS", "PS", "PR"),
                            labels = c("Global sum", "Drugwise sum", 
                                       "Targetwise sum", "Sum of drugwise and targetwise sums", 
                                       "Product of drugwise and targetwise sums", "Polynomial regression"))) %>%
  group_by(algorithm, setting, drug, target, test_perf_measure) %>%
  summarise(average = mean(as.numeric(test_performance)), 
            lower = quantile(as.numeric(test_performance), probs = c(0.025)),
            upper = quantile(as.numeric(test_performance), probs = c(0.975)))

# Draw separate figures for the methods.
for(alg_name in unique(performances_XOR_imbalance_summary$algorithm)){
  assign(paste0("g_", alg_name), performances_XOR_imbalance_summary %>%
           subset(algorithm == alg_name) %>%
           mutate(drug = factor(drug, levels = c("in", "off"),
                                labels = c("In-training-set\ndrugs", 
                                           "Off-training-set\ndrugs")),
                  target = factor(target, levels = c("in", "off"),
                                  labels = c("In-training-set\ntargets", 
                                             "Off-training-set\ntargets"))) %>%
          
           ggplot() +
           
           # Draw a bar plot with algorithm on x-axis, test performance on y-axis,
           # and a group of bars containing the bars for different test performance measures.
           geom_col(mapping = aes(x = algorithm, y = average-0.4, 
                                  group = test_perf_measure, 
                                  fill = test_perf_measure),
                    position = "dodge", #stat = "identity", 
                    # width = 0.6,
           ) +
           # Add errorbars denoting the credible intervals. 
           geom_errorbar(aes(x = algorithm, 
                             ymin = lower-0.4, 
                             ymax = upper-0.4,
                             group = test_perf_measure),
                         position = "dodge", stat = "identity", 
                         # width = 0.6,
                         alpha = 0.7,
           ) +
           # Highlight 0.5 corresponding to random prediction.
           geom_hline(yintercept = 0.5-0.4, alpha = 0.2) +
           facet_grid(drug~target, drop = FALSE) +
           
           # Modify the theme of the plot.
           # Font changed to Times New Roman. 
           theme(
             text = element_text(size = 8, 
                                 # family = "Times New Roman",
                                 ),
             axis.ticks.x = element_blank(),
             axis.text.x = element_blank(),
             title = element_text(size = 8),
           ) +
           
           # Modify how the legends are presented.
           guides(fill = guide_legend(label.position = "right",
                                      title.position = "top"),
                  col = guide_legend(label.position = "right",
                                     title.position = "top")) +
           
           # Change the labels to match that the y-axis was scaled so that the
           # columns start from 0.4 instead of 0. 
           scale_y_continuous(breaks = c(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                              labels = c(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                              limits = c(0.0,0.6)) +
           
           # Determine the colors and how their labels are shown in the figure. 
           scale_color_discrete(breaks = c("A_index", "C_d_index", "C_t_index",
                                           "C_index", "accuracy"),
                                labels = c("A-index", bquote(C[d]-index), 
                                           bquote(C[t]-index),
                                           "C-index", "Accuracy"),
                                type = c("#F8766D", "#7CAE00", "#00BFC4", "#C77CFF",
                                         "#999999")) +
           scale_fill_discrete(breaks = c("A_index", "C_d_index", "C_t_index",
                                          "C_index", "accuracy"),
                               labels = c("A-index", bquote(C[d]-index), 
                                          bquote(C[t]-index),
                                          "C-index", "Accuracy"),
                               type = c("#F8766D", "#7CAE00", "#00BFC4", "#C77CFF",
                                        "#999999")) +
           # Modify the variable titles.
           labs(
             fill = "Performance measure",
             group = "Performance measure",
             y = NULL,
             x = NULL,
             title = alg_name)
  )
}

g_simulation <- guide_area()/(`g_Sum of drugwise and targetwise sums` + `g_Drugwise sum`)/
  (`g_Targetwise sum` + `g_Global sum`)/(`g_Product of drugwise and targetwise sums` + `g_Polynomial regression`) + 
  plot_layout(guides = "collect", heights = c(1,4,4,4)) +
  # plot_annotation(tag_levels = c("GS", "PR", "DS", "TS", "SS", "PS")) +
  NULL &
  theme(legend.position = "top",
        legend.justification = "left",
        plot.tag.position = c(0, 0.95),
        plot.tag = element_text(hjust = 0, vjust = -1),)
ggsave("barplots_simulation.pdf",
       g_simulation, width = 15, height = 17, units = "cm")


#### Drug-target data sets. See the other figures.R file!!!!! ####
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
                                              levels = c("A_index", "C_d_index",
                                                         "C_t_index", "C_index"))

  # Split a string variable to separate variables for setting, method and
  # performance measure.
  # The setting notations are not modified in the prediction files even though they
  # were changed to IDIT etc. if the predictions were calculated with the current codes.
  # The IDIT etc. alternative would be regex = "[IO]D[IO]T".
  setting_locations <- stri_locate(df_performances$model, regex = "\'S[1-4]\'")
  df_performances["setting"] <- substr(df_performances$model,
                                       setting_locations[,1]+1,
                                       setting_locations[,2]-1)
  # If the IDIT etc. abbreviations are used above, change them also here so that
  # c("S1", "S2") is replaced with c("IDIT", "IDOT") and 
  # c("S1", "S3") with c("IDIT", "ODIT").
  df_performances <- df_performances %>%
    mutate(drug = case_when(setting %in% c("S1", "S2") ~ "In-training-set\ndrugs",
                            TRUE ~ "Off-training-set\ndrugs"),
           target = case_when(setting %in% c("S1", "S3") ~ "In-training-set\ntargets",
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

  algorithm_specification_locations <- stri_locate(df_performances$model, regex = "\\}.*\",") # "\\[.*\\]"
  df_performances["specification"] <- substr(df_performances$model,
                                             algorithm_specification_locations[,1]+1,
                                             algorithm_specification_locations[,2]-2)

  perf_measure_locations <- stri_locate(df_performances$model, regex = "\", \'.*\'\\)$")
  df_performances$perf_measure <- substr(df_performances$model,
                                         perf_measure_locations[,1]+4, perf_measure_locations[,2]-2)
  # Again, fix manually the rows for GraphDTA and feedforward neural network.
  # These methods were calculated only with MSE at validation phase. 
  df_performances$perf_measure[c(GDTA_rows, FF_rows)] <- "MSE"
  
  # Ignore other than MSE as a performance measure at validation phase.
  df_performances <- subset(df_performances, perf_measure == "MSE")

  # Modify the names of the methods to clearer versions.
  df_performances$algorithm[df_performances$specification == ", ['gaussian', 'gaussian', 'pko_kronecker'])"] <-
    "KR (Gaussian)"
  df_performances$algorithm[df_performances$specification == ", ['gaussian', 'gaussian', 'pko_linear'])"] <-
    "LR (Gaussian)"
  df_performances$algorithm[df_performances$specification == ", ['linear', 'linear', 'pko_kronecker'])"] <-
    "KR (linear)"
  df_performances$algorithm[df_performances$specification == ", ['linear', 'linear', 'pko_linear'])"] <-
    "LR (linear)"
  df_performances$algorithm[df_performances$algorithm == "ltr_cls"] <- "PR"
  df_performances$algorithm[df_performances$algorithm == "ensemble._forest.RandomForestRegressor"] <- "RF"
  df_performances$algorithm[df_performances$algorithm == "neighbors._regression.KNeighborsRegressor"] <- "kNN"
  df_performances$algorithm[df_performances$algorithm == "sklearn.XGBRegressor"] <- "XGB"
  df_performances$algorithm[df_performances$algorithm == "DeepDTA"] <- "DDTA"

  return(df_performances)
}

performances_20122024 <-
  read.csv("~/Ristiinvalidointiprojekti/AssignmentIndex/performances_20122024.csv")

performances_A_C_Cd_Ct_summary <-
  prepare_df_for_figures(performances_20122024)

data_set_names <- c("Davis", "Metz", "KiBA", "Merget", "GPCR", "IC", "E")
for (dsn in data_set_names) {
  performances_A_C_Cd_Ct_summary %>%
    subset(data == dsn) %>%
    ggplot() +
    # Draw the bar plot with algorithm on x-axis, test performance on y-axis,
    # and a group of bars contains the bars for different test performance measures.
    geom_bar(mapping = aes(x = algorithm, y = test_performance-0.4,
                           group = test_perf_measure,
                           fill = test_perf_measure),
             position = "dodge", stat = "identity", width = 0.6) +
    # Highligth 0.5 as it corresponds to random prediction.
    geom_hline(yintercept = 0.5-0.4, alpha = 0.2) +
    expand_limits(y = 0.6) +
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
    scale_color_discrete(breaks = c("A_index", "C_d_index", "C_t_index", "C_index"),
                         labels = c("A-index", bquote(C[d]-index), 
                                    bquote(C[t]-index), "C-index")) +
    scale_linetype_discrete(breaks = c("A_index", "C_d_index", "C_t_index", "C_index"),
                            labels = c("A-index", bquote(C[d]-index), 
                                       bquote(C[t]-index), "C-index")) +
    scale_fill_discrete(breaks = c("A_index", "C_d_index", "C_t_index", "C_index"),
                        labels = c("A-index", bquote(C[d]-index), 
                                   bquote(C[t]-index), "C-index")) +
    # Modify the variable titles.
    labs(
      fill = "Performance measure",
      group = "Performance measure",
      x = NULL)
  ggsave(paste0("barplot_",dsn,".pdf"), height = 10, width = 15, units = "cm")
}
