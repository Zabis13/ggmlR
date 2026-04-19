#!/usr/bin/env Rscript
# Test 5D tensor operations: add, mul, sub, div, concat (all axes).
# Each op is run on CPU and Vulkan, outputs are compared numerically.
# Run: Rscript inst/examples/test_5d_ops.R

suppressPackageStartupMessages(library(ggmlR))

ABS_TOL <- 1e-4
REL_TOL <- 1e-3

# ---- helpers -----------------------------------------------------------------

make_backend <- function(device) {
  if (device == "cpu") {
    b <- ggml_backend_cpu_init()
    ggml_backend_cpu_set_n_threads(b, 2L)
    b
  } else {
    ggml_vulkan_init(0L)
  }
}

# Run a single op defined by build_fn(ctx) -> output_tensor.
# Sets input tensor data via set_inputs(ctx, buf) after allocation.
# Returns numeric vector of output data, or an error string.
run_op <- function(device, build_fn, set_inputs, n_out) {
  tryCatch({
    ctx     <- ggml_init(mem_size = 64L * 1024L * 1024L, no_alloc = TRUE)
    out     <- build_fn(ctx)
    backend <- make_backend(device)
    buf     <- ggml_backend_alloc_ctx_tensors(ctx, backend)
    set_inputs(ctx, buf)
    graph   <- ggml_build_forward_expand(ctx, out)
    ggml_backend_graph_compute(backend, graph)
    result  <- ggml_backend_tensor_get_data(out, n_elements = n_out)
    ggml_backend_buffer_free(buf)
    ggml_backend_free(backend)
    ggml_free(ctx)
    result
  }, error = function(e) conditionMessage(e))
}

compare <- function(a, b, label) {
  if (is.character(a)) return(list(ok = FALSE, msg = paste("CPU error:", a)))
  if (is.character(b)) return(list(ok = FALSE, msg = paste("GPU error:", b)))
  if (length(a) != length(b))
    return(list(ok = FALSE, msg = sprintf("length mismatch: %d vs %d", length(a), length(b))))
  if (any(!is.finite(a)) || any(!is.finite(b)))
    return(list(ok = FALSE, msg = "NaN/Inf in output"))
  diff    <- abs(a - b)
  max_abs <- max(diff)
  max_rel <- max_abs / max(max(abs(a), abs(b)), 1e-8)
  ok      <- max_abs < ABS_TOL || max_rel < REL_TOL
  list(ok = ok, max_abs = max_abs, max_rel = max_rel, msg = NULL)
}

results <- list()

record <- function(name, cmp) {
  status <- if (isTRUE(cmp$ok)) "PASS" else "FAIL"
  results[[length(results) + 1]] <<- list(
    name   = name,
    status = status,
    msg    = cmp$msg,
    max_abs = cmp$max_abs,
    max_rel = cmp$max_rel
  )
}

# ---- test cases --------------------------------------------------------------

# Shape: ne0=4, ne1=3, ne2=2, ne3=5, ne4=2  (5D, ~240 elements)
NE <- c(4L, 3L, 2L, 5L, 2L)
N  <- prod(NE)
set.seed(7)
A_DATA <- runif(N, 0.1, 1.0)
B_DATA <- runif(N, 0.1, 1.0)

make_ab <- function(ctx) {
  a <- ggml_new_tensor(ctx, GGML_TYPE_F32, 5L, NE)
  b <- ggml_new_tensor(ctx, GGML_TYPE_F32, 5L, NE)
  list(a = a, b = b)
}

# --- binary ops ---

make_binary_test <- function(opname) {
  env <- new.env(parent = emptyenv())
  env$tensors <- list()

  build <- function(ctx) {
    ab <- make_ab(ctx)
    env$tensors <- ab
    switch(opname,
      add = ggml_add(ctx, ab$a, ab$b),
      mul = ggml_mul(ctx, ab$a, ab$b),
      sub = ggml_sub(ctx, ab$a, ab$b),
      div = ggml_div(ctx, ab$a, ab$b)
    )
  }

  setter <- function(ctx, buf) {
    ggml_backend_tensor_set_data(env$tensors$a, A_DATA)
    ggml_backend_tensor_set_data(env$tensors$b, B_DATA)
  }

  list(build = build, setter = setter)
}

for (opname in c("add", "mul", "sub", "div")) {
  t <- make_binary_test(opname)
  cpu <- run_op("cpu",    t$build, t$setter, N)
  gpu <- run_op("vulkan", t$build, t$setter, N)
  record(sprintf("5D %s", opname), compare(cpu, gpu))
}

# --- concat on each axis ---

make_concat_test <- function(axis) {
  env <- new.env(parent = emptyenv())
  env$tensors <- list()

  NE_out           <- NE
  NE_out[axis + 1L] <- NE_out[axis + 1L] * 2L
  N_out            <- prod(NE_out)

  build <- function(ctx) {
    ab <- make_ab(ctx)
    env$tensors <- ab
    ggml_concat(ctx, ab$a, ab$b, dim = axis)
  }

  setter <- function(ctx, buf) {
    ggml_backend_tensor_set_data(env$tensors$a, A_DATA)
    ggml_backend_tensor_set_data(env$tensors$b, B_DATA)
  }

  list(build = build, setter = setter, n_out = N_out)
}

for (axis in 0:4) {
  t <- make_concat_test(axis)
  cpu <- run_op("cpu",    t$build, t$setter, t$n_out)
  gpu <- run_op("vulkan", t$build, t$setter, t$n_out)
  record(sprintf("5D concat axis=%d", axis), compare(cpu, gpu))
}

# --- rms_norm on 5D ---

make_rms_test <- function() {
  env <- new.env(parent = emptyenv())
  env$a <- NULL

  build <- function(ctx) {
    a <- ggml_new_tensor(ctx, GGML_TYPE_F32, 5L, NE)
    env$a <- a
    ggml_rms_norm(ctx, a)
  }

  setter <- function(ctx, buf) {
    ggml_backend_tensor_set_data(env$a, A_DATA)
  }

  list(build = build, setter = setter)
}

t_rms <- make_rms_test()
cpu_rms <- run_op("cpu",    t_rms$build, t_rms$setter, N)
gpu_rms <- run_op("vulkan", t_rms$build, t_rms$setter, N)
record("5D rms_norm", compare(cpu_rms, gpu_rms))

# ---- report ------------------------------------------------------------------

if (!ggml_vulkan_available()) {
  cat("Vulkan not available — skipping GPU tests\n")
  quit(status = 0)
}

cat("=============================================================\n")
cat(sprintf("  5D ops test — %s\n", ggml_vulkan_device_description(0)))
cat("=============================================================\n\n")
cat(sprintf("%-22s %-6s %10s %10s  %s\n",
            "Test", "Status", "max_abs", "max_rel", "Note"))
cat(strrep("-", 62), "\n")

for (r in results) {
  cat(sprintf("%-22s %-6s %10s %10s  %s\n",
    r$name, r$status,
    if (is.null(r$max_abs)) "—" else sprintf("%.2e", r$max_abs),
    if (is.null(r$max_rel)) "—" else sprintf("%.2e", r$max_rel),
    if (!is.null(r$msg)) r$msg else ""))
}

n_pass <- sum(sapply(results, function(r) r$status == "PASS"))
n_fail <- sum(sapply(results, function(r) r$status == "FAIL"))
cat(sprintf("\n%d PASS   %d FAIL\n", n_pass, n_fail))

if (n_fail > 0) quit(status = 1)
