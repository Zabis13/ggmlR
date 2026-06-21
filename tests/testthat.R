library(testthat)
library(ggmlR)

if (requireNamespace("mlr3", quietly = TRUE)) library(mlr3)
if (requireNamespace("parsnip", quietly = TRUE)) library(parsnip)

heavy <- c(
  "normalization",
  "sequence-ops",
  "binary-operations",
  "glu",
  "nn-sequential",
  "matrix-ops",
  "callbacks",
  "onnx-quant",
  "cpu-backend",
  "onnx-elementwise",
  "onnx-norm",
  "onnx-misc",
  "onnx-logic",
  "onnx-gemm",
  "onnx-resize",
  "onnx-nn",
  "nn-functional",
  "activations-extended",
  "onnx-ops",
  "transformer-ops",
  "onnx-shape",
  "math-operations",
  "activations",
  "backend-extended",
  "helpers",
  "onnx-reduce",
  "chain-sequential-batchnorm",
  "chain-patterns",
  "onnx-chain-quant",
  "onnx-chain-qlinearconv",
  "onnx-chain-uncovered-ops",
  "onnx-chain-transformer",
  "onnx-chain-broadcast-strict",
  "onnx-chain-maskrcnn-broadcast",
  "onnx-chain-erf-gelu",
  "onnx-chain-resize-broadcast",
  "onnx-chain-constantofshape-int64",
  "onnx-chain-roberta-attn",
  "onnx-chain-batched-matmul",
  "onnx-chain-classify",
  "onnx-chain-xcit-dynamic",
  "onnx-chain-fpn",
  "onnx-chain-pooling",
  "onnx-chain-unet",
  "onnx-chain-audio",
  "onnx-chain-superres",
  "onnx-chain-convtranspose",
  "onnx-chain-eyelike",
  "onnx-chain-decoder",
  "onnx-chain-detect",
  "onnx-chain-powernorm",
  "onnx-chain-bert-mlp",
  "onnx-chain-position",
  "onnx-chain-attn-mask",
  "onnx-chain-postprocess",
  "onnx-chain-cast",
  "onnx-chain-preprocess",
  "model-ops",
  "onnx-boundary",
  "onnx-edge",
  "parsnip",
  "parsnip-tidymodels",
  "parsnip-broom",
  "seed",
  "upscale",
  "rope-multi",
  "mlr3-learner",
  "keras-api",
  "quants-iq-degenerate",
  "getrows-offload-vulkan"
)

# Under the valgrind memtest, running the full light suite (~67 files) blows
# the memory limit: valgrind's shadow memory accumulates across every
# ggml_init / backend buffer and never returns to the OS, so the process grows
# to several GiB and gets SIGKILLed regardless of any single test's size.
# valgrind's purpose here is to catch C-level memory bugs (use-after-free,
# invalid reads/writes) in the ggml bindings — a small core suite that
# exercises context/tensor lifecycle, basic ops and the backend is enough for
# that. So under valgrind we run an explicit whitelist instead of the full set.
valgrind_core <- c(
  "tensors",       # tensor create/free lifecycle
  "context",       # ggml_init / ggml_free
  "ggml",          # core library smoke
  "types",         # type helpers
  "operations-extended",
  "tensor-utils",
  "backend",       # backend init/free + buffers
  "memory",
  "quants"         # quant/dequant round-trips
)

# Detect valgrind via several independent signals (any one is enough):
#  1. vgpreload_*.so injected into the process (visible in /proc/self/maps)
#  2. the RUNNING_ON_VALGRIND env var some setups export
#  3. R's own valgrind marker in the session (R_CHECK_*/loaded vg libs)
detect_valgrind <- function() {
  if (nzchar(Sys.getenv("RUNNING_ON_VALGRIND"))) return(TRUE)
  # /proc/self/maps holds vgpreload_core / vgpreload_memcheck under valgrind
  m <- tryCatch(readLines("/proc/self/maps", warn = FALSE),
                error = function(e) character(0))
  if (any(grepl("valgrind|vgpreload", m, ignore.case = TRUE))) return(TRUE)
  # /proc/self/cmdline of the wrapping process: valgrind launches R
  s <- tryCatch(readLines("/proc/self/status", warn = FALSE),
                error = function(e) character(0))
  ppid <- sub("^PPid:\\s*", "", grep("^PPid:", s, value = TRUE))
  if (length(ppid) == 1L) {
    pcmd <- tryCatch(
      readChar(file.path("/proc", ppid, "cmdline"), 1e4, useBytes = TRUE),
      error = function(e) "")
    if (grepl("valgrind", pcmd, ignore.case = TRUE)) return(TRUE)
  }
  FALSE
}
under_valgrind <- isTRUE(detect_valgrind())
message("valgrind detected: ", under_valgrind)

# Disable OpenMP thread pool under valgrind: GOMP_parallel → pthread_create
# allocates 352b TLS per worker thread which valgrind flags as "possibly lost".
# Under valgrind we test correctness, not throughput — 1 thread is sufficient.
if (under_valgrind) {
  Sys.setenv(OMP_NUM_THREADS = "1")
}

on_cran <- !identical(Sys.getenv("NOT_CRAN"), "true")

test_dir <- if (dir.exists("testthat")) "testthat" else "tests/testthat"

if (on_cran) {
  message("--- RUNNING LIGHT TESTS ONLY ---")

  all_tests <- list.files(test_dir, pattern = "^test-.*\\.R$")
  all_names <- sub("^test-(.*)\\.R$", "\\1", all_tests)

  if (under_valgrind) {
    message("--- VALGRIND DETECTED: running small core suite only ---")
    # Whitelist intersected with what actually exists in the package.
    light_tests <- intersect(valgrind_core, all_names)
  } else {
    light_tests <- setdiff(all_names, heavy)
  }
  message("Tests to run: ", paste(light_tests, collapse = ", "))

  if (length(light_tests) == 0) {
    # fallback
    test_check("ggmlR")
  } else {
    # testthat applies `filter` as grepl(filter, <test name>) with no anchors,
    # so anchor each name with ^...$ to avoid e.g. "backend" also matching
    # "backend-buffers"/"backend-extended" (which would re-bloat the set).
    filter_regex <- paste0("^(", paste(light_tests, collapse = "|"), ")$")
    test_check("ggmlR", filter = filter_regex)
  }
} else {
  test_check("ggmlR")
}
