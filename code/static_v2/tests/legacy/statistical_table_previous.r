# ============================================================
# ROI GROUP COMPARISON
# Kruskal-Wallis tests with BH multiple-comparison correction
# Table contains mean ± SD, H statistic, and adjusted p-values
# ============================================================

# Install these packages once if necessary:
# install.packages(c("officer", "flextable", "gridExtra"))

library(officer)
library(flextable)
library(gridExtra)
library(grid)
library(FSA)

# ------------------------------------------------------------
# 1. Import data
# ------------------------------------------------------------

dat <- read.csv(
  "mice_NodePosStrength_PCorr.csv",
  header = TRUE,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

# First column = subject index
# Second column = group
names(dat)[1:2] <- c("index", "group")

# Remove underscores from group names
dat$group <- gsub("_", "", trimws(dat$group), fixed = TRUE)

# Set the required left-to-right group order
group_order <- c(
  "NTGY", "NTGM", "NTGA",
  "CRND8Y", "CRND8M",
  "5XFADY", "5XFADM"
)

# Check for unexpected group names
unexpected_groups <- setdiff(unique(dat$group), group_order)

if (length(unexpected_groups) > 0) {
  stop(
    "Unexpected group names found: ",
    paste(unexpected_groups, collapse = ", ")
  )
}

# Convert to an explicitly ordered factor
dat$group <- factor(
  dat$group,
  levels = group_order
)

# All remaining columns are ROIs
roi_names <- names(dat)[-(1:2)]

# Convert ROI columns to numeric
dat[roi_names] <- lapply(
  dat[roi_names],
  function(x) as.numeric(as.character(x))
)

# ------------------------------------------------------------
# 2. Formatting functions
# ------------------------------------------------------------

mean_sd <- function(x) {
  x <- x[is.finite(x)]

  if (length(x) == 0) {
    return("NA")
  }

  if (length(x) == 1) {
    return(sprintf("%.2f ± NA", mean(x)))
  }

  sprintf("%.2f ± %.2f", mean(x), sd(x))
}

format_p <- function(p) {
  if (is.na(p)) {
    return("NA")
  }

  stars <- if (p < 0.001) {
    "***"
  } else if (p < 0.01) {
    "**"
  } else if (p < 0.05) {
    "*"
  } else {
    ""
  }

  if (p < 0.001) {
    paste0("<0.001", stars)
  } else {
    paste0(sprintf("%.3f", p), stars)
  }
}

# ------------------------------------------------------------
# 3. Descriptive statistics: mean ± SD by group
# ------------------------------------------------------------

group_names <- levels(dat$group)

descriptive_results <- data.frame(
  ROI = roi_names,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

for (g in group_names) {
  descriptive_results[[g]] <- vapply(
    roi_names,
    function(roi) mean_sd(dat[dat$group == g, roi]),
    character(1)
  )
}

# ------------------------------------------------------------
# 4. Kruskal-Wallis test for every ROI
# ------------------------------------------------------------

test_results <- lapply(roi_names, function(roi) {

  test_data <- data.frame(
    value = dat[[roi]],
    group = dat$group
  )

  # Remove missing/non-finite observations
  test_data <- test_data[
    !is.na(test_data$group) & is.finite(test_data$value),
  ]

  # Remove empty group levels
  test_data$group <- droplevels(test_data$group)

  if (nlevels(test_data$group) < 2) {
    return(data.frame(
      ROI = roi,
      H = NA_real_,
      df = NA_integer_,
      p_raw = NA_real_
    ))
  }

  result <- kruskal.test(value ~ group, data = test_data)

  data.frame(
    ROI = roi,
    H = unname(result$statistic),
    df = unname(result$parameter),
    p_raw = result$p.value
  )
})

test_results <- do.call(rbind, test_results)

# BH/FDR correction across all tested ROIs
test_results$p_adjusted <- p.adjust(
  test_results$p_raw,
  method = "BH"
)

# ------------------------------------------------------------
# 5. Create publication table
# ------------------------------------------------------------

summary_table <- merge(
  descriptive_results,
  test_results,
  by = "ROI",
  sort = FALSE
)

# Preserve original ROI order
summary_table <- summary_table[
  match(roi_names, summary_table$ROI),
]

summary_table$`Kruskal-Wallis H (df)` <- ifelse(
  is.na(summary_table$H),
  "NA",
  sprintf("%.2f (%d)", summary_table$H, summary_table$df)
)

summary_table$`Adjusted p-value` <- vapply(
  summary_table$p_adjusted,
  format_p,
  character(1)
)

# Keep only columns intended for the paper
summary_table <- summary_table[
  c(
    "ROI",
    group_names,
    "Kruskal-Wallis H (df)",
    "Adjusted p-value"
  )
]

# ============================================================
# OPTIONAL POST-HOC TABLE: DUNN'S TEST WITH HOLM CORRECTION
# Correction is across group contrasts separately within each ROI
# ============================================================

dunn_results <- do.call(
  rbind,
  lapply(roi_names, function(roi) {

    test_data <- data.frame(
      value = dat[[roi]],
      group = dat$group
    )

    # Remove missing and non-finite values
    test_data <- test_data[
      !is.na(test_data$group) & is.finite(test_data$value),
      ,
      drop = FALSE
    ]

    test_data$group <- droplevels(test_data$group)

    if (nlevels(test_data$group) < 2) {
      return(NULL)
    }

    # Dunn's test; Holm adjustment occurs among contrasts
    # for this ROI only because each ROI is analyzed separately
    result <- FSA::dunnTest(
      value ~ group,
      data = test_data,
      method = "holm"
    )$res

    data.frame(
      ROI = roi,
      Contrast = gsub(" - ", " vs ", result$Comparison, fixed = TRUE),
      Z = result$Z,
      p_raw = result$P.unadj,
      p_adjusted = result$P.adj,
      stringsAsFactors = FALSE
    )
  })
)

# Format Dunn Z statistics
dunn_results$`Dunn Z` <- ifelse(
  is.na(dunn_results$Z),
  "NA",
  sprintf("%.2f", dunn_results$Z)
)

# Format Holm-adjusted p-values and add significance symbols
dunn_results$`Holm-adjusted p-value` <- vapply(
  dunn_results$p_adjusted,
  format_p,
  character(1)
)

# Final publication table
dunn_table <- dunn_results[
  c(
    "ROI",
    "Contrast",
    "Dunn Z",
    "Holm-adjusted p-value"
  )
]

# Preserve original ROI order
dunn_table$ROI <- factor(
  dunn_table$ROI,
  levels = roi_names
)

dunn_table <- dunn_table[
  order(dunn_table$ROI),
  ,
  drop = FALSE
]

dunn_table$ROI <- as.character(dunn_table$ROI)

# Export CSV
write.csv(
  dunn_table,
  "Strength_Dunn_posthoc_table.csv",
  row.names = FALSE,
  na = ""
)

# ------------------------------------------------------------
# Export Dunn table to Word
# ------------------------------------------------------------

dunn_ft <- flextable(dunn_table)
dunn_ft <- theme_booktabs(dunn_ft)
dunn_ft <- bold(dunn_ft, part = "header")
dunn_ft <- align(dunn_ft, align = "center", part = "all")
dunn_ft <- align(
  dunn_ft,
  j = c("ROI", "Contrast"),
  align = "left",
  part = "all"
)
dunn_ft <- fontsize(dunn_ft, size = 9, part = "all")
dunn_ft <- padding(dunn_ft, padding = 3, part = "all")
dunn_ft <- autofit(dunn_ft)
dunn_ft <- set_caption(
  dunn_ft,
  "Dunn's post-hoc tests with Holm correction by ROI"
)

dunn_word <- read_docx()

dunn_word <- body_add_par(
  dunn_word,
  "ROI Dunn Post-hoc Comparisons",
  style = "heading 1"
)

dunn_word <- body_add_flextable(
  dunn_word,
  value = dunn_ft
)

dunn_word <- body_add_par(
  dunn_word,
  paste(
    "Dunn's post-hoc tests follow Kruskal-Wallis tests.",
    "P-values were Holm-adjusted across pairwise group contrasts",
    "separately within each ROI.",
    "* p < 0.05; ** p < 0.01; *** p < 0.001."
  )
)

dunn_word <- body_end_section_landscape(dunn_word)

print(
  dunn_word,
  target = "Strength_Dunn_posthoc_table.docx"
)

# ------------------------------------------------------------
# Export Dunn table to multipage PDF
# ------------------------------------------------------------

dunn_rows_per_page <- 25
dunn_pages <- ceiling(nrow(dunn_table) / dunn_rows_per_page)

pdf(
  "Strength_Dunn_posthoc_table.pdf",
  width = 11,
  height = 8.5,
  onefile = TRUE
)

for (page in seq_len(dunn_pages)) {

  first_row <- ((page - 1) * dunn_rows_per_page) + 1
  last_row <- min(page * dunn_rows_per_page, nrow(dunn_table))

  page_table <- dunn_table[
    first_row:last_row,
    ,
    drop = FALSE
  ]

  table_graphic <- tableGrob(
    page_table,
    rows = NULL,
    theme = ttheme_minimal(
      base_size = 8,
      core = list(
        fg_params = list(hjust = 0.5, x = 0.5)
      ),
      colhead = list(
        fg_params = list(
          fontface = "bold",
          hjust = 0.5,
          x = 0.5
        )
      )
    )
  )

  grid.newpage()

  grid.draw(
    textGrob(
      "ROI Dunn Post-hoc Comparisons",
      x = 0.5,
      y = 0.97,
      gp = gpar(fontsize = 14, fontface = "bold")
    )
  )

  pushViewport(
    viewport(
      x = 0.5,
      y = 0.52,
      width = 0.95,
      height = 0.85
    )
  )

  grid.draw(table_graphic)
  popViewport()

  grid.draw(
    textGrob(
      paste(
        "Dunn's tests; Holm correction across contrasts",
        "separately within each ROI.",
        "* p < 0.05; ** p < 0.01; *** p < 0.001."
      ),
      x = 0.5,
      y = 0.025,
      gp = gpar(fontsize = 7)
    )
  )
}

dev.off()

message(
  "Completed: Strength_Dunn_posthoc_table.csv, ",
  "Strength_Dunn_posthoc_table.docx, and ",
  "Strength_Dunn_posthoc_table.pdf"
)

# ------------------------------------------------------------
# 6. Export results as CSV
# ------------------------------------------------------------

write.csv(
  summary_table,
  "Strength_summary_table.csv",
  row.names = FALSE,
  na = ""
)

# ------------------------------------------------------------
# 7. Export results as a Word DOCX file
# ------------------------------------------------------------

ft <- flextable(summary_table)

ft <- theme_booktabs(ft)
ft <- bold(ft, part = "header")
ft <- align(ft, align = "center", part = "all")
ft <- align(ft, j = "ROI", align = "left", part = "all")
ft <- fontsize(ft, size = 9, part = "all")
ft <- padding(ft, padding = 3, part = "all")
ft <- autofit(ft)
ft <- set_caption(
  ft,
  "ROI values by group and Kruskal-Wallis test results"
)

word_document <- read_docx()

word_document <- body_add_par(
  word_document,
  "ROI Group Comparison",
  style = "heading 1"
)

word_document <- body_add_flextable(
  word_document,
  value = ft
)

word_document <- body_add_par(
  word_document,
  paste(
    "Values are mean ± SD.",
    "P-values were adjusted across ROIs using the",
    "Benjamini-Hochberg false-discovery-rate procedure.",
    "* p < 0.05; ** p < 0.01; *** p < 0.001."
  )
)

word_document <- body_end_section_landscape(word_document)

print(
  word_document,
  target = "Strength_summary_table.docx"
)

# ------------------------------------------------------------
# 8. Export results as a multipage landscape PDF
# ------------------------------------------------------------

rows_per_page <- 20
number_of_pages <- ceiling(nrow(summary_table) / rows_per_page)

pdf(
  "Strength_summary_table.pdf",
  width = 14,
  height = 8.5,
  onefile = TRUE
)

for (page in seq_len(number_of_pages)) {

  first_row <- ((page - 1) * rows_per_page) + 1
  last_row <- min(page * rows_per_page, nrow(summary_table))

  page_table <- summary_table[first_row:last_row, , drop = FALSE]

  table_graphic <- tableGrob(
    page_table,
    rows = NULL,
    theme = ttheme_minimal(
      base_size = 7,
      core = list(
        fg_params = list(hjust = 0.5, x = 0.5)
      ),
      colhead = list(
        fg_params = list(fontface = "bold", hjust = 0.5, x = 0.5)
      )
    )
  )

  grid.newpage()

  grid.draw(
    textGrob(
      "ROI Group Comparison",
      x = 0.5,
      y = 0.97,
      gp = gpar(fontsize = 14, fontface = "bold")
    )
  )

  pushViewport(
    viewport(
      x = 0.5,
      y = 0.52,
      width = 0.98,
      height = 0.85
    )
  )

  grid.draw(table_graphic)
  popViewport()

  grid.draw(
    textGrob(
      paste(
        "Mean ± SD; Kruskal-Wallis H (df);",
        "BH-adjusted p-values.",
        "* p < 0.05; ** p < 0.01; *** p < 0.001."
      ),
      x = 0.5,
      y = 0.025,
      gp = gpar(fontsize = 7)
    )
  )
}

dev.off()

message(
  "Completed: ROI_summary_table.csv, ",
  "ROI_summary_table.docx, and ROI_summary_table.pdf"
)