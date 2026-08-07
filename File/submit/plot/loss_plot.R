#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(jsonlite)
    library(dplyr)
    library(ggplot2)
    library(ggExtra)
    library(viridis)
    library(tidyverse)
    library(patchwork)
})
setwd("E:/git/maize")

# Define a function to read and clean YOLO results
read_yolo_results <- function(file_path, device_type) {
  df <- read_csv(file_path, show_col_types = FALSE)
  # Clean names (remove whitespace and special characters)
  colnames(df) <- trimws(colnames(df))
  df$device <- device_type
  return(df)
}

# Read both results
results_dslr <- read_yolo_results("results_dslr.csv", "DSLR")[c(1:125),]
results_uav <- read_yolo_results("results_uav.csv", "UAV")[c(1:125),]

# Reshape data for plotting losses
loss_data <- results_dslr %>%
  select(epoch, device, starts_with("train/"), starts_with("val/")) %>%
  pivot_longer(cols = -c(epoch, device), names_to = "loss_type", values_to = "loss_value")

# Split loss_type into phase (train/val) and category (box, pose, etc.)
loss_data <- loss_data %>%
  separate(loss_type, into = c("phase", "category"), sep = "/")

# Plotting
p1 <- ggplot(loss_data, aes(x = epoch, y = loss_value, color = phase)) +
  geom_line(alpha = 0.8, size = 0.8) +
  facet_grid(category ~ device, scales = "free_y") +
  theme_bw() +
  labs(
       x = "Epoch",
       y = "Loss Value",
       color = "Phase") +
  theme(legend.position = "bottom",
        strip.text = element_text(face = "bold"),
        panel.grid.minor = element_blank())+ scale_color_manual(values = c("train" = "#FFD700", "val" = "#228B22"))
p1


# Reshape data for plotting losses
loss_data <- results_uav %>%
  select(epoch, device, starts_with("train/"), starts_with("val/")) %>%
  pivot_longer(cols = -c(epoch, device), names_to = "loss_type", values_to = "loss_value")

# Split loss_type into phase (train/val) and category (box, pose, etc.)
loss_data <- loss_data %>%
  separate(loss_type, into = c("phase", "category"), sep = "/")

# Plotting
p2 <- ggplot(loss_data, aes(x = epoch, y = loss_value, color = phase)) +
  geom_line(alpha = 0.8, size = 0.8) +
  facet_grid(category ~ device, scales = "free_y") +
  theme_bw() +
  labs(
       x = "Epoch",
       y = "Loss Value",
       color = "Phase") +
  theme(legend.position = "bottom",
        strip.text = element_text(face = "bold"),
        panel.grid.minor = element_blank())+ scale_color_manual(values = c("train" = "#FFD700", "val" = "#228B22"))
p2
p <- p1|p2;p
# Save the plot
ggsave("plot/yolo_loss_plot.pdf", p, width = 6, height = 5)

# --- Add Metrics Plotting ---

# Function to prepare metric data
prep_metrics <- function(df) {
  df %>%
    select(epoch, device, starts_with("metrics/")) %>%
    pivot_longer(cols = -c(epoch, device), names_to = "metric", values_to = "value") %>%
    mutate(metric = gsub("metrics/", "", metric))
}

metrics_dslr <- prep_metrics(results_dslr)
metrics_uav <- prep_metrics(results_uav)

plot_metrics <- function(data, title) {
  ggplot(data, aes(x = epoch, y = value, color = device)) +
    geom_line(size = 0.8) +
    facet_wrap(~metric, scales = "free_y", ncol = 2) +
    theme_bw() +
    labs(title = title, x = "Epoch", y = "Value") +
    theme(legend.position = "none", strip.text = element_text(size = 8))+scale_color_manual(values = c("DSLR" = "orange", "UAV" = "#228B22"))
}

m1 <- plot_metrics(metrics_dslr, "DSLR Metrics")
m2 <- plot_metrics(metrics_uav, "UAV Metrics")

m_plot <- m1 | m2;m_plot
ggsave("plot/yolo_metrics_plot.pdf", m_plot, width = 10, height = 8)
