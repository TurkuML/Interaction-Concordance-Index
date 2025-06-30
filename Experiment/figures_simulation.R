#### Libraries and functions ####
library(stringi)
library(ggplot2)
library(reshape2)
library(patchwork)
library(dplyr)

# Function to modify the data frame of performances into form that the plots can be drawn.
prepare_df_for_figures_simulation <- function(df_performances){
  # Melt the data frame.
  df_performances <- melt(df_performances, 
                          id.vars = c("model"),
                          variable.name = "test_perf_measure",
                          value.name = "test_performance")
  
  df_performances$test_perf_measure <- factor(df_performances$test_perf_measure,
                                              levels = c("IC_index", "C_index_d",
                                                         "C_index_t", "C_index", "accuracy"))
  
  
  # Split a string variable to separate variables for the settings and methods.
  setting_locations <- stri_locate(df_performances$model, regex = "\'[IO]D[IO]T\'")
  df_performances["setting"] <- substr(df_performances$model,
                                       setting_locations[,1]+1,
                                       setting_locations[,2]-1)
  # These are for having the abbreviations written out in the figures.
  df_performances <- df_performances %>%
    mutate(drug = as.factor(case_when(setting %in% c("IDIT", "IDOT") ~ "In-training-set\ndrugs",
                            TRUE ~ "Off-training-set\ndrugs")),
           target = as.factor(case_when(setting %in% c("IDIT", "ODIT") ~ "In-training-set\ntargets",
                              TRUE ~ "Off-training-set\ntargets")))
  
  algorithm_locations <- stri_locate(df_performances$model, regex = "\\, \'[a-zA-Z_]*\'")
  df_performances["algorithm"] <- substr(df_performances$model, 
                                         algorithm_locations[,1]+3, 
                                         algorithm_locations[,2]-1)
  return(df_performances)
}
#### Draw the figure ####
# Read the data.
performances_XOR_imbalance_0.1_0.2 <- read.csv("performances_XOR_imbalance_0.1_0.2.csv")

# Calculate the mean values and credible intervals,
performances_XOR_imbalance_summary <- 
  # The column of random seeds is not needed. Hence, do not include the first column.
  prepare_df_for_figures_simulation(performances_XOR_imbalance_0.1_0.2[,-c(1)]) %>%
  # This is for having the longer names in the figures instead of the abbreviations.
  mutate(algorithm = factor(algorithm, 
                            levels = c("GS", "DS", "TS", "SS", "PS", "PR"),
                            labels = c("Global sum", "Drugwise sum", 
                                       "Targetwise sum", "Sum of drugwise and targetwise sums", 
                                       "Product of drugwise and targetwise sums", "Polynomial regression"))) %>%
  # The averages and credible interval lower and upper bounds are calculated separately
  # for the different methods, settings and performance measures.
  group_by(algorithm, setting, drug, target, test_perf_measure) %>%
  summarise(average = mean(as.numeric(test_performance)), 
            lower = quantile(as.numeric(test_performance), probs = c(0.025)),
            upper = quantile(as.numeric(test_performance), probs = c(0.975)))

# Draw separate figures for the methods.
for(alg_name in unique(performances_XOR_imbalance_summary$algorithm)){
  assign(paste0("g_", alg_name), performances_XOR_imbalance_summary %>%
           subset(algorithm == alg_name) %>%
           ggplot() +
           
           # Draw a bar plot with algorithm on x-axis, test performance on y-axis,
           # and a group of bars containing the bars for different test performance measures.
           geom_col(mapping = aes(x = algorithm, y = average-0.4, 
                                  group = test_perf_measure, 
                                  fill = test_perf_measure),
                    position = "dodge",
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
           scale_color_discrete(breaks = c("IC_index", "C_index_d", "C_index_t",
                                           "C_index", "accuracy"),
                                labels = c("IC-index", bquote(C[d]-index), 
                                           bquote(C[t]-index),
                                           "C-index", "Accuracy"),
                                type = c("#F8766D", "#7CAE00", "#00BFC4", "#C77CFF",
                                         "#999999")) +
           scale_fill_discrete(breaks = c("IC_index", "C_index_d", "C_index_t",
                                          "C_index", "accuracy"),
                               labels = c("IC-index", bquote(C[d]-index), 
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
  NULL &
  theme(legend.position = "top",
        legend.justification = "left",
        plot.tag.position = c(0, 0.95),
        plot.tag = element_text(hjust = 0, vjust = -1),)
ggsave("barplots_simulation.pdf",
       g_simulation, width = 15, height = 17, units = "cm")
