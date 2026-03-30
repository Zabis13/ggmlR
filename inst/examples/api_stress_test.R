#!/usr/bin/env Rscript
# ============================================================================
# API Stress Test Client
# ============================================================================
# Usage:
#   1. Start server:  Rscript api_server.R mnist 8080
#   2. Run client:    Rscript api_stress_test.R [host:port] [total_requests] [n_workers]
#
#   Rscript api_stress_test.R localhost:8080 1000 4
#
# Requires: httr2, jsonlite (not ggmlR dependencies, install separately)
# ============================================================================

library(httr2)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
base_url       <- if (length(args) >= 1) args[1] else "localhost:8080"
total_requests <- if (length(args) >= 2) as.integer(args[2]) else 1000L
n_workers      <- if (length(args) >= 3) as.integer(args[3]) else 1L

if (!grepl("^https?://", base_url)) base_url <- paste0("http://", base_url)

cat("==============================================================\n")
cat("  API Stress Test\n")
cat("==============================================================\n\n")
cat(sprintf("  Server:    %s\n", base_url))
cat(sprintf("  Requests:  %s\n", format(total_requests, big.mark = ",")))
cat(sprintf("  Workers:   %d\n\n", n_workers))

# --- Check server health ---
cat("Checking server... ")
health <- tryCatch({
  resp <- request(paste0(base_url, "/health")) |> req_perform()
  resp_body_json(resp)
}, error = function(e) {
  stop(sprintf("Server not reachable at %s: %s", base_url, e$message))
})
cat(sprintf("OK (model=%s, device=%s)\n", health$model, health$device))

# --- Get model info ---
info <- tryCatch({
  resp <- request(paste0(base_url, "/info")) |> req_perform()
  resp_body_json(resp)
}, error = function(e) list(input_size = 784))

input_size <- as.integer(info$input_size)
cat(sprintf("Input size: %d values\n\n", input_size))

# --- Single worker function ---
run_worker <- function(n_requests, worker_id) {
  set.seed(42 + worker_id)
  input_data <- runif(input_size)
  body_json <- toJSON(list(data = input_data), auto_unbox = TRUE)

  times     <- numeric(n_requests)
  errors    <- 0L
  http_codes <- integer(n_requests)

  for (i in seq_len(n_requests)) {
    t0 <- proc.time()
    result <- tryCatch({
      resp <- request(paste0(base_url, "/predict")) |>
        req_body_raw(body_json, type = "application/json") |>
        req_perform()
      http_codes[i] <- resp_status(resp)
      resp_body_json(resp)
    }, error = function(e) {
      errors <<- errors + 1L
      http_codes[i] <<- 0L
      NULL
    })
    times[i] <- (proc.time() - t0)[3]
  }

  list(
    worker_id  = worker_id,
    n_requests = n_requests,
    times      = times,
    errors     = errors,
    http_codes = http_codes
  )
}

# --- Run stress test ---
cat("Running stress test...\n")
requests_per_worker <- ceiling(total_requests / n_workers)

t_total <- proc.time()

if (n_workers == 1L) {
  results <- list(run_worker(total_requests, 1L))
} else {
  results <- parallel::mclapply(
    seq_len(n_workers),
    function(w) run_worker(requests_per_worker, w),
    mc.cores = n_workers
  )
}

total_sec <- (proc.time() - t_total)[3]

# --- Aggregate results ---
all_times  <- unlist(lapply(results, function(r) r$times))
all_errors <- sum(sapply(results, function(r) r$errors))
actual_total <- length(all_times)

# Convert to ms
all_times_ms <- all_times * 1000

cat("\n==============================================================\n")
cat("  Results\n")
cat("==============================================================\n\n")

cat(sprintf("  Total requests:  %s\n", format(actual_total, big.mark = ",")))
cat(sprintf("  Errors:          %d (%.1f%%)\n", all_errors,
            100 * all_errors / actual_total))
cat(sprintf("  Wall time:       %.1f sec\n", total_sec))
cat(sprintf("  Throughput:      %.0f req/sec\n\n", actual_total / total_sec))

cat("  Latency (ms):\n")
cat(sprintf("    Mean:    %8.2f\n", mean(all_times_ms)))
cat(sprintf("    Median:  %8.2f\n", median(all_times_ms)))
cat(sprintf("    p90:     %8.2f\n", quantile(all_times_ms, 0.90)))
cat(sprintf("    p95:     %8.2f\n", quantile(all_times_ms, 0.95)))
cat(sprintf("    p99:     %8.2f\n", quantile(all_times_ms, 0.99)))
cat(sprintf("    p99.9:   %8.2f\n", quantile(all_times_ms, 0.999)))
cat(sprintf("    Min:     %8.2f\n", min(all_times_ms)))
cat(sprintf("    Max:     %8.2f\n", max(all_times_ms)))
cat(sprintf("    Stdev:   %8.2f\n\n", sd(all_times_ms)))

# Latency stability: first 10% vs last 10%
n10 <- max(1L, as.integer(actual_total * 0.1))
lat_first <- mean(all_times_ms[seq_len(n10)])
lat_last  <- mean(all_times_ms[seq(actual_total - n10 + 1, actual_total)])
drift_pct <- (lat_last - lat_first) / lat_first * 100
cat(sprintf("  Stability:\n"))
cat(sprintf("    First 10%%: %.2f ms\n", lat_first))
cat(sprintf("    Last 10%%:  %.2f ms\n", lat_last))
cat(sprintf("    Drift:     %+.1f%%\n\n", drift_pct))

# Per-worker breakdown
if (n_workers > 1) {
  cat("  Per-worker breakdown:\n")
  cat(sprintf("  %-8s %10s %10s %10s %8s\n",
              "Worker", "Requests", "Mean(ms)", "P99(ms)", "Errors"))
  cat(paste(rep("-", 52), collapse = ""), "\n")
  for (r in results) {
    w_ms <- r$times * 1000
    cat(sprintf("  %-8d %10d %10.2f %10.2f %8d\n",
                r$worker_id, r$n_requests,
                mean(w_ms), quantile(w_ms, 0.99), r$errors))
  }
  cat("\n")
}

# --- CSV output ---
csv_file <- sprintf("api_stress_%s_%s.csv", health$model,
                     format(Sys.time(), "%Y%m%d_%H%M%S"))
csv_df <- data.frame(
  request_id = seq_len(actual_total),
  latency_ms = round(all_times_ms, 3)
)
write.csv(csv_df, csv_file, row.names = FALSE)
cat(sprintf("  Per-request latencies saved to: %s\n", csv_file))

cat("\n==============================================================\n")
cat("  Stress test complete\n")
cat("==============================================================\n")
