# Run IDSL.IPA on one benchmark run with the template parameter spreadsheet.
# Only run-specific fields are changed: input/output paths, file list, thread count,
# and the RT-correction reference samples (the template names the authors' own files).
# Every detection parameter keeps its template value.
# usage: Rscript run_idslipa.R TEMPLATE.xlsx DATA_DIR OUT_DIR THREADS
suppressMessages({
  lib <- Sys.getenv("R_LIBS_USER")
  library(readxl, lib.loc = lib)
  library(writexl, lib.loc = lib)
  library(IDSL.IPA, lib.loc = lib)
})
a <- commandArgs(trailingOnly = TRUE)
template <- a[1]; data_dir <- normalizePath(a[2]); out_dir <- a[3]; threads <- a[4]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_dir <- normalizePath(out_dir)

sheets <- excel_sheets(template)
tabs <- lapply(sheets, function(s) read_xlsx(template, sheet = s, col_types = "text"))
names(tabs) <- sheets
p <- tabs[["parameters"]]
setp <- function(id, value) {
  i <- which(p[[2]] == id)
  stopifnot(length(i) == 1)
  p[i, 4] <<- as.character(value)
}

files <- sort(list.files(data_dir, pattern = "\\.(mzML|mzXML)$", ignore.case = TRUE))
setp("PARAM0006", threads)
setp("PARAM0007", data_dir)
setp("PARAM0008", "All")
setp("PARAM0010", out_dir)
# parallelize over peaks instead of samples when there are fewer files than threads
# (a scheduling choice only; the template's "Sample Mode" would leave cores idle)
if (length(files) < as.numeric(threads)) setp("PARAM_PAR", "Peak Mode")
if (length(files) >= 2) {
  # RT-correction reference samples: the template's 003/004 do not exist here;
  # use the first two files of the run (alphabetical)
  setp("PARAM0030", paste(head(files, 2), collapse = ";"))
} else {
  # a single file has nothing to align, gap-fill, or RT-correct against
  setp("PARAM0002", "NO"); setp("PARAM0003", "NO"); setp("PARAM0029", "NO")
}
tabs[["parameters"]] <- p
xlsx <- file.path(out_dir, "IPA_parameters_used.xlsx")
write_xlsx(tabs, xlsx)
cat("IDSL.IPA", as.character(packageVersion("IDSL.IPA", lib.loc = lib)),
    "on", length(files), "files\n")
IPA_workflow(xlsx)
