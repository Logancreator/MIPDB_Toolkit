#!/usr/bin/env Rscript

## COCO keypoint dataset statistics in R
## Equivalent functionality to dataset_statistics.py using ggplot2

suppressPackageStartupMessages({
    library(jsonlite)
    library(dplyr)
    library(ggplot2)
    library(ggExtra)
    library(viridis)
})
setwd("E:/git/maize")
## 如果只在本机/RStudio 内使用，可以在这里直接写死默认 CSV 路径和输出目录
## 按需修改为你自己的实际路径
DEFAULT_BBOX_CSV <- "coco_bbox_stats.csv"
DEFAULT_KPT_CSV  <- "coco_keypoint_stats.csv"
DEFAULT_OUTPUT_DIR <- "./"

plot_coco_stats_impl <- function(args) {
    # Read precomputed CSV statistics (generated e.g. by the Python script)
    bbox_csv_path <- normalizePath(args$bbox_csv, mustWork = TRUE)
    kpt_csv_path  <- normalizePath(args$kpt_csv, mustWork = TRUE)

    output_dir <- args$output_dir
    if (is.null(output_dir) || output_dir == "") {
        output_dir <- dirname(bbox_csv_path)
    }
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

    cat("Loading bbox stats from:", bbox_csv_path, "\n")
    cat("Loading keypoint stats from:", kpt_csv_path, "\n")

    df_bbox <- read.csv(bbox_csv_path, stringsAsFactors = FALSE)
    df_kpt  <- read.csv(kpt_csv_path, stringsAsFactors = FALSE)

    if (nrow(df_bbox) == 0L) {
        stop("No valid bbox annotations for statistics (annotations may be empty or bbox has zero area).", call. = FALSE)
    }
    if (nrow(df_kpt) == 0L) {
        stop("No valid keypoint annotations for statistics (keypoints may be empty or all v==0).", call. = FALSE)
    }

    cat("Number of bboxes:", nrow(df_bbox), "\n")
    cat("Number of keypoints:", nrow(df_kpt), "\n")

    ## Optionally subsample for KDE plots to speed up visualization on large datasets
    max_n <- max(as.integer(args$max_samples_kde), 1L)

    set.seed(0L)
    if (nrow(df_bbox) > max_n) {
        df_bbox_plot <- df_bbox[sample.int(nrow(df_bbox), max_n), , drop = FALSE]
    } else {
        df_bbox_plot <- df_bbox
    }

    if (nrow(df_kpt) > max_n) {
        df_kpt_plot <- df_kpt[sample.int(nrow(df_kpt), max_n), , drop = FALSE]
    } else {
        df_kpt_plot <- df_kpt
    }

    # Ensure numeric columns for ggplot (avoid "Discrete value supplied to continuous scale")
    if (nrow(df_bbox_plot) > 0L) {
        df_bbox_plot <- df_bbox_plot %>%
            mutate(
                bbox_w = as.numeric(bbox_w),
                bbox_h = as.numeric(bbox_h),
                area = as.numeric(area),
                aspect_ratio = as.numeric(aspect_ratio)
            )
    }

    if (nrow(df_kpt_plot) > 0L) {
        df_kpt_plot <- df_kpt_plot %>%
            mutate(
                x = as.numeric(x),
                y = as.numeric(y),
                x_norm = as.numeric(x_norm),
                y_norm = as.numeric(y_norm)
            )
    }

    ## Global plotting style (similar to Python script)
    theme_set(theme_bw(base_size = 11))

    ## 2. Bounding box dimensions (width x height) - continuous heatmap style with marginal histograms
    if (!isTRUE(args$no_bbox_plot)) {
        p_bbox <- ggplot(df_bbox_plot, aes(x = bbox_w, y = bbox_h)) +
            # Invisible scatter layer so ggMarginal can detect a geom_point layer
            geom_point(alpha = 0) +
            stat_density_2d(
                aes(fill = after_stat(ndensity)),
                geom = "raster",
                contour = FALSE,
                n = 80
            ) +
            scale_fill_viridis_c(option = "viridis", name = "Density") +
            labs(
                title = "Bounding Box Dimensions",
                x = "Bounding box width (pixels)",
                y = "Bounding box height (pixels)"
            ) +
            theme(
                panel.grid.major = element_line(linetype = "dotted", colour = "grey80"),
                panel.grid.minor = element_blank(),
                plot.title = element_text(hjust = 0.5),
                panel.border = element_blank(),
                axis.line = element_line(colour = "black"),
                legend.position = "right"
            )

        ## Add marginal histograms similar to seaborn jointplot
        p_bbox_joint <- ggMarginal(
            p_bbox,
            type = "histogram",
            bins = 30,
            fill = "#3CB371",
            colour = "black"
        )

        ggsave(file.path(output_dir, "bbox_dimensions.png"), p_bbox_joint,
                     width = 6, height = 6, dpi = 300)
        ggsave(file.path(output_dir, "bbox_dimensions.pdf"), p_bbox_joint,
                     width = 6, height = 6)
    }

    ## 3. Keypoint spatial distribution (normalized coordinates) - continuous heatmap style with marginal histograms
    if (!isTRUE(args$no_kpt_plot)) {
        p_kpt <- ggplot(df_kpt_plot, aes(x = x_norm, y = y_norm)) +
            # Invisible scatter layer so ggMarginal can detect a geom_point layer
            geom_point(alpha = 0) +
            stat_density_2d(
                aes(fill = after_stat(ndensity)),
                geom = "raster",
                contour = FALSE,
                n = 80
            ) +
            scale_fill_viridis_c(option = "viridis", name = "Density") +
            scale_x_continuous(limits = c(0, 1)) +
            scale_y_reverse(limits = c(1, 0)) +
            labs(
                title = "Keypoint Spatial Distribution (normalized)",
                x = "Normalized X",
                y = "Normalized Y"
            ) +
            theme(
                panel.grid.major = element_line(linetype = "dotted", colour = "grey80"),
                panel.grid.minor = element_blank(),
                plot.title = element_text(hjust = 0.5),
                panel.border = element_blank(),
                axis.line = element_line(colour = "black"),
                legend.position = "right"
            )

        p_kpt_joint <- ggMarginal(
            p_kpt,
            type = "histogram",
            bins = 30,
            fill = "#3CB371",
            colour = "black"
        )

        ggsave(file.path(output_dir, "keypoint_spatial_distribution.png"), p_kpt_joint,
                     width = 6, height = 6, dpi = 300)
        ggsave(file.path(output_dir, "keypoint_spatial_distribution.pdf"), p_kpt_joint,
                     width = 6, height = 6)
    }

    ## 4. Per-image bbox/keypoint density (histograms)
    bbox_counts <- df_bbox %>%
        count(image_id, name = "n_bbox")

    p_bbox_hist <- ggplot(bbox_counts, aes(x = n_bbox)) +
        geom_histogram(bins = 30, fill = "#4C72B0", colour = "black", alpha = 0.8) +
        labs(
            x = "Number of bboxes per image",
            y = "Number of images",
            title = "Bounding box density per image"
        ) +
        theme(
            panel.grid.major.y = element_line(linetype = "dotted", colour = "grey80"),
            panel.grid.minor = element_blank(),
            panel.border = element_blank(),
            axis.line = element_line(colour = "black"),
            plot.title = element_text(hjust = 0.5)
        )

    ggsave(file.path(output_dir, "bbox_count_hist.png"), p_bbox_hist,
                 width = 6, height = 4, dpi = 300)
    ggsave(file.path(output_dir, "bbox_count_hist.pdf"), p_bbox_hist,
                 width = 6, height = 4)

    kpt_counts <- df_kpt %>%
        count(image_id, name = "n_kpt")

    p_kpt_hist <- ggplot(kpt_counts, aes(x = n_kpt)) +
        geom_histogram(bins = 30, fill = "#4C72B0", colour = "black", alpha = 0.8) +
        labs(
            x = "Number of keypoints per image",
            y = "Number of images",
            title = "Keypoint density per image"
        ) +
        theme(
            panel.grid.major.y = element_line(linetype = "dotted", colour = "grey80"),
            panel.grid.minor = element_blank(),
            panel.border = element_blank(),
            axis.line = element_line(colour = "black"),
            plot.title = element_text(hjust = 0.5)
        )

    ggsave(file.path(output_dir, "keypoint_count_hist.png"), p_kpt_hist,
                 width = 6, height = 4, dpi = 300)
    ggsave(file.path(output_dir, "keypoint_count_hist.pdf"), p_kpt_hist,
                 width = 6, height = 4)

    ## 5. Save statistics to CSV
    write.csv(df_bbox, file.path(output_dir, "coco_bbox_stats.csv"), row.names = FALSE)
    write.csv(df_kpt, file.path(output_dir, "coco_keypoint_stats.csv"), row.names = FALSE)

    cat("BBox statistics saved to", file.path(output_dir, "coco_bbox_stats.csv"), "\n")
    cat("Keypoint statistics saved to", file.path(output_dir, "coco_keypoint_stats.csv"), "\n")
}

plot_coco_stats_from_csv <- function(bbox_csv = DEFAULT_BBOX_CSV,
                                     kpt_csv  = DEFAULT_KPT_CSV,
                                     output_dir = DEFAULT_OUTPUT_DIR,
                                     max_samples_kde = 50000L,
                                     no_bbox_plot = FALSE,
                                     no_kpt_plot = FALSE) {
    args <- list(
        bbox_csv = bbox_csv,
        kpt_csv = kpt_csv,
        output_dir = output_dir,
        max_samples_kde = max_samples_kde,
        no_bbox_plot = no_bbox_plot,
        no_kpt_plot = no_kpt_plot
    )
    plot_coco_stats_impl(args)
}

## 直接运行脚本时，使用写死的默认路径自动生成图
if (sys.nframe() == 0) {
    plot_coco_stats_from_csv()
}
# End of dataset_statistics.R

