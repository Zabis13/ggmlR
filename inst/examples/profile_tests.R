#!/usr/bin/env Rscript
# Per-file timing profile of the testthat suite: where does the test time go?
# Run: Rscript inst/examples/profile_tests.R
#
# Runs the whole suite in a separate R process (clean Vulkan context, and a
# crash cannot take down the calling session), then reports per-file CPU time,
# wall time, and their ratio. A ratio well above 1 means the file parallelises
# across threads; a ratio near 1 means it is serial or GPU/IO-bound.
#
# Sorted by wall time, so the files worth optimising (or moving to the `heavy`
# list in tests/testthat.R) are at the top.

library(callr)
library(dplyr)

PKG_DIR <- "/mnt/Data2/DS_projects/ggmlR"

# ---- run the suite ----------------------------------------------------------
#
# NOTE: test_dir(), NOT test_local(). test_local() calls pkgload::load_all(),
# which reloads the package from source and builds a FRESH namespace with a new
# (empty) .ag_device_state -- while an already-registered mlr3 learner
# (.register_mlr3) stays bound to the OLD namespace. That mismatch produces
# spurious failures in test-mlr3-autograd.R (accuracy 0.55 instead of 0.97) that
# appear neither under R CMD check nor on CRAN. test_dir() uses the installed
# package, with no reload, and reports the truth.
#
# stop_on_failure = FALSE so one failing file still leaves a complete table.

results <- callr::r(
  function() {
    library(testthat)
    library(ggmlR)
    reporter <- ListReporter$new()
    test_dir("tests/testthat", reporter = reporter, stop_on_failure = FALSE)
    reporter$get_results()
  },
  show = TRUE,
  wd = PKG_DIR
)

# ---- shape the raw results into a table -------------------------------------

stats_list <- lapply(results, function(res) {
  data.frame(
    file    = res$file,
    cpu     = res$user + res$system,
    elapsed = res$real
  )
})
df_raw <- do.call(rbind, stats_list)

# ---- aggregate per file, compute the parallelism ratio ----------------------

df_grouped <- df_raw %>%
  group_by(file) %>%
  summarise(total_cpu     = sum(cpu),
            total_elapsed = sum(elapsed),
            .groups = "drop") %>%
  mutate(ratio = ifelse(total_elapsed > 0, total_cpu / total_elapsed, 0)) %>%
  arrange(desc(total_elapsed))

print(df_grouped, n = Inf)

cat(sprintf("\nTotal wall time: %.1f s across %d files\n",
            sum(df_grouped$total_elapsed), nrow(df_grouped)))
