#!/usr/bin/env bash
# A/B: GGML_SCHED_MAX_SPLIT_INPUTS = 30 (upstream) vs 128 (ours), TODO 9.2a.
#
# Uses the INSTALLED ggmlR (build it first). The limit is read by
# ggml_backend_sched_new() from the environment variable of the same name,
# so both variants run from one build, each in a fresh process, interleaved
# (30 128 30 128 ...) to spread drift evenly.
#
# Per run: MaskRCNN-12-int8 load, first run, median of N timed runs,
# detections, VmPeak/VmSize (address space), VmHWM (peak RSS), device memory.
# At the end: per-variant medians over rounds and the 30-vs-128 difference.
#
# Usage (from anywhere):
#   bash inst/scripts/ab_sched_split_inputs.sh [device=vulkan] [rounds=3] [runs=10]
set -euo pipefail

DEV="${1:-vulkan}"
ROUNDS="${2:-3}"
RUNS="${3:-10}"
VARIANTS=(30 128)
OUT="${AB_OUT:-/tmp/ab_split_inputs_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

# ---- measure (one process = one variant) ------------------------------------
cat > "$OUT/measure.R" <<'EOF'
args <- commandArgs(trailingOnly = TRUE)
dev <- args[1]; runs <- as.integer(args[2])
suppressMessages(library(ggmlR))

path  <- file.path(Sys.getenv("ONNX_DIR", "/mnt/Data2/DS_projects/ONNX models-main"),
                   "MaskRCNN-12-int8.onnx")
shape <- c(3L, 224L, 224L)

vm_kb <- function(key) {
  l <- grep(paste0("^", key, ":"), readLines("/proc/self/status"), value = TRUE)
  if (!length(l)) NA_real_ else as.numeric(gsub("[^0-9]", "", l))
}
t_ms <- function(f) { t0 <- Sys.time(); f(); as.numeric(Sys.time() - t0, units = "secs") * 1000 }

vk   <- dev == "vulkan" && ggml_vulkan_available()
mem0 <- if (vk) ggml_vulkan_device_memory(0)$free else NA
model <- NULL
t_load <- t_ms(function() model <<- onnx_load(path, device = dev, input_shapes = list(image = shape)))
set.seed(42)
inp <- list(image = runif(prod(shape)))
out <- NULL
t_first <- t_ms(function() out <<- onnx_run(model, inp))
det_first <- length(as.numeric(out[[1]])) %/% 4L
ts <- vapply(seq_len(runs), function(i) t_ms(function() out <<- onnx_run(model, inp)), numeric(1))
det_last <- length(as.numeric(out[[1]])) %/% 4L
mem1 <- if (vk) ggml_vulkan_device_memory(0)$free else NA

cat(sprintf("%s,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%d,%d,%.1f,%.1f,%.1f,%.1f\n",
    Sys.getenv("GGML_SCHED_MAX_SPLIT_INPUTS"), dev, t_load, t_first,
    median(ts), min(ts), max(ts), det_first, det_last,
    vm_kb("VmPeak") / 1024, vm_kb("VmSize") / 1024, vm_kb("VmHWM") / 1024,
    if (vk) (mem0 - mem1) / 2^20 else NA_real_))
EOF

# ---- confirm the variable is honoured (split debug line prints "limit N") ---
echo "=== check: limit seen by the scheduler"
for N in "${VARIANTS[@]}"; do
  GGML_SCHED_MAX_SPLIT_INPUTS=$N GGML_SCHED_DEBUG_SPLITS=1 \
    Rscript "$OUT/measure.R" "$DEV" 1 > /dev/null 2> "$OUT/splits_$N.err" || true
  printf 'N=%-4s ' "$N"; grep -m1 "^\[sched\]" "$OUT/splits_$N.err" || echo "(no [sched] line, see $OUT/splits_$N.err)"
done

# ---- runs --------------------------------------------------------------------
CSV="$OUT/results.csv"
echo "N,device,load_ms,first_ms,median_ms,min_ms,max_ms,det_first,det_last,VmPeak_MB,VmSize_MB,VmHWM_MB,dev_mem_MB" > "$CSV"
for r in $(seq 1 "$ROUNDS"); do
  for N in "${VARIANTS[@]}"; do
    echo "=== round $r/$ROUNDS, N=$N"
    GGML_SCHED_MAX_SPLIT_INPUTS=$N Rscript "$OUT/measure.R" "$DEV" "$RUNS" >> "$CSV" \
      2> "$OUT/run_${N}_r$r.err" || echo "N=$N round $r FAILED, see $OUT/run_${N}_r$r.err"
  done
done

# ---- summary -----------------------------------------------------------------
echo
echo "=== all runs"
column -s, -t < "$CSV"
echo
echo "=== summary (median over rounds; diff = 30 - 128)"
Rscript -e '
d <- read.csv(commandArgs(TRUE)[1])
cols <- c("load_ms","first_ms","median_ms","min_ms","max_ms","VmPeak_MB","VmSize_MB","VmHWM_MB","dev_mem_MB")
s <- sapply(split(d[cols], d$N), function(x) sapply(x, median, na.rm = TRUE))
r <- data.frame(metric = cols, N30 = s[, "30"], N128 = s[, "128"])
r$diff <- r$N30 - r$N128
r$pct  <- 100 * r$diff / r$N128
print(format(r, digits = 4, nsmall = 1), row.names = FALSE)
cat("\nrounds per variant:", paste(names(table(d$N)), table(d$N), sep = "=", collapse = " "), "\n")
det <- unique(d[c("N","det_first","det_last")])
cat("detections (first/last):\n"); print(det, row.names = FALSE)
if (nrow(unique(d[c("det_first","det_last")])) > 1) cat("WARNING: detections differ between runs/variants\n")
' "$CSV"
echo
echo "results: $CSV"
