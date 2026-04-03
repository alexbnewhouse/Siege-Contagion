#!/usr/bin/env Rscript
# Export all Iron March .rda tables to CSV for Python ingestion

out_dir <- "data/raw/csv"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

data_dir <- "data/raw/ironmarch/data"

export_list <- function(obj, prefix) {
  if (is.data.frame(obj)) {
    # Flatten list columns to character so write.csv doesn't fail
    for (cn in colnames(obj)) {
      if (is.list(obj[[cn]])) {
        obj[[cn]] <- vapply(obj[[cn]], function(x) {
          if (is.null(x) || length(x) == 0) NA_character_
          else paste(x, collapse = ";")
        }, character(1))
      }
    }
    path <- file.path(out_dir, paste0(prefix, ".csv"))
    cat("  Writing", path, ":", nrow(obj), "rows x", ncol(obj), "cols\n")
    write.csv(obj, path, row.names = FALSE)
  } else if (is.list(obj)) {
    for (name in names(obj)) {
      export_list(obj[[name]], paste0(prefix, "__", name))
    }
  } else {
    cat("  Skipping", prefix, "- type:", class(obj), "\n")
  }
}

rda_files <- list.files(data_dir, pattern = "\\.rda$", full.names = TRUE)

for (f in rda_files) {
  cat("\n=== Loading", basename(f), "===\n")
  env <- new.env()
  tryCatch({
    load(f, envir = env)
    for (name in ls(env)) {
      obj <- get(name, envir = env)
      cat("  Object:", name, "- class:", paste(class(obj), collapse=", "), "\n")
      export_list(obj, name)
    }
  }, error = function(e) {
    cat("  ERROR:", conditionMessage(e), "\n")
  })
}

cat("\nDone! Files in", out_dir, ":\n")
cat(paste(list.files(out_dir), collapse="\n"), "\n")
