# Compute through the meta backend (test-meta-backend.R only round-trips data).
#
# The meta device wraps two references to the CPU device, so each "device" gets
# its own CPU backend and its own share of every split tensor -- the multi-device
# path without a second GPU. The same graph runs through the meta backend and
# through a plain CPU backend; the results must agree.
#
# The graph also carries two ggmlR-only ops (GELU_BACK, CAST_NUMERIC): the meta
# backend's split-state switch did not know them and hit
# GGML_ABORT("ggml op not implemented") as soon as such a graph ran on it.

K <- 8L; M <- 4L; N <- 3L
set.seed(7)
# Small integers: the matmul is exact in f32, so CAST_NUMERIC -> I32 is exact too.
XW  <- as.numeric(sample(-3:3, K * M, replace = TRUE))
XX  <- as.numeric(sample(-3:3, K * N, replace = TRUE))
XDY <- runif(M * N, -1, 1)
# mm = w^T x, ggml ne = [M, N]; crossprod gives the same column-major order.
REF_MM <- as.vector(crossprod(matrix(XW, K), matrix(XX, K)))

cpu_dev_or_skip <- function() {
  dev <- ggml_backend_dev_by_type(ggml_backend_device_type_cpu())
  skip_if(is.null(dev), "No CPU device available")
  dev
}

# split: named list tensor name -> list(axis, ne); anything not listed is MIRRORED.
meta_split_fn <- function(split) {
  function(tensor_info, n_devs) {
    s <- split[[tensor_info$name]]
    if (is.null(s)) return(list(axis = 10L, ne = rep(0L, n_devs), n_segments = 1L))
    list(axis = s$axis, ne = s$ne, n_segments = 1L)
  }
}

# Builds w, x, dy in a buffer of `wbackend`, runs
#   mm = mul_mat(w, x); sc = 0.5 * mm; gb = gelu_back(dy, sc); ci = cast(mm, I32)
# through a scheduler over `backends`, returns every output.
run_graph <- function(wbackend, backends, read_mm = TRUE) {
  ctx_w <- ggml_init(1024 * 1024, no_alloc = TRUE)
  w  <- ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, K, M); ggml_set_name(w,  "w")
  x  <- ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, K, N); ggml_set_name(x,  "x")
  dy <- ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, M, N); ggml_set_name(dy, "dy")
  buf <- ggml_backend_alloc_ctx_tensors(ctx_w, wbackend)
  ggml_backend_tensor_set_data(w,  XW)
  ggml_backend_tensor_set_data(x,  XX)
  ggml_backend_tensor_set_data(dy, XDY)

  ctx_g <- ggml_init(1024 * 1024, no_alloc = TRUE)
  mm <- ggml_mul_mat(ctx_g, w, x);                ggml_set_name(mm, "mm")
  sc <- ggml_scale(ctx_g, mm, 0.5);               ggml_set_name(sc, "sc")
  gb <- ggml_gelu_back(ctx_g, dy, sc);            ggml_set_name(gb, "gb")
  ci <- ggml_cast_numeric(ctx_g, mm, GGML_TYPE_I32); ggml_set_name(ci, "ci")
  for (t in list(mm, sc, gb, ci)) ggml_set_output(t)
  gf <- ggml_build_forward_expand(ctx_g, gb)
  ggml_graph_expand(gf, ci)

  sched <- ggml_backend_sched_new(backends, parallel = FALSE)
  on.exit({
    ggml_backend_sched_free(sched)
    ggml_free(ctx_g)
    ggml_backend_buffer_free(buf)
    ggml_free(ctx_w)
  })
  expect_true(ggml_backend_sched_alloc_graph(sched, gf))
  expect_identical(ggml_backend_sched_graph_compute(sched, gf), 0L)

  # read_mm = FALSE: in the row-parallel layouts mm holds per-device partial
  # sums; the meta backend reduces them for its consumers but cannot read them.
  list(mm = if (read_mm) ggml_backend_tensor_get_data(mm),
       sc = ggml_backend_tensor_get_data(sc),
       gb = ggml_backend_tensor_get_data(gb),
       ci = as.numeric(ggml_backend_tensor_get_data(ci)))
}

reference <- local({
  ref <- NULL
  function() {
    if (is.null(ref)) {
      cpu <- ggml_backend_cpu_init()
      on.exit(ggml_backend_free(cpu))
      ref <<- run_graph(cpu, list(cpu))
    }
    ref
  }
})

run_meta <- function(split, read_mm = TRUE) {
  cpu_dev <- cpu_dev_or_skip()
  meta_dev <- ggml_backend_meta_device(list(cpu_dev, cpu_dev), meta_split_fn(split))
  meta <- ggml_backend_dev_init(meta_dev)
  skip_if(is.null(meta), "Could not init meta backend")
  cpu <- ggml_backend_cpu_init()
  on.exit({ ggml_backend_free(cpu); ggml_backend_free(meta) })
  run_graph(meta, list(meta, cpu), read_mm)
}

expect_same_as_cpu <- function(out, tol = 1e-6) {
  ref <- reference()
  expect_equal(ref$mm, REF_MM, tolerance = 1e-6)
  if (!is.null(out$mm)) expect_equal(out$mm, ref$mm, tolerance = tol)
  expect_equal(out$sc, ref$sc, tolerance = tol)
  expect_equal(out$gb, ref$gb, tolerance = tol)
  expect_identical(out$ci, ref$ci)
  expect_identical(out$ci, REF_MM)
}

test_that("meta compute, everything MIRRORED, matches plain CPU", {
  expect_same_as_cpu(run_meta(list()))
})

test_that("meta compute, w split along axis 1 (column-parallel), matches plain CPU", {
  # mm comes out split along axis 0; scale / gelu_back / cast follow that split.
  # gelu_back is element-wise over (dy, sc): both srcs must share one split
  # state, so dy is split the same way.
  half <- c(M / 2L, M / 2L)
  expect_same_as_cpu(run_meta(list(w  = list(axis = 1L, ne = half),
                                    dy = list(axis = 0L, ne = half))))
})

test_that("meta compute, w and x split along K (row-parallel, reduced), matches plain CPU", {
  # both split along axis 0: each device holds a partial sum, the meta backend
  # reduces them before mm is used
  s <- list(axis = 0L, ne = c(K / 2L, K / 2L))
  expect_same_as_cpu(run_meta(list(w = s, x = s), read_mm = FALSE))
})

# ---- ggml_meta_backend(): named rules instead of the raw axis-code callback ----

run_meta_backend <- function(devices, split, read_mm = TRUE) {
  meta <- ggml_meta_backend(devices, split = split)
  cpu <- ggml_backend_cpu_init()
  on.exit({ ggml_backend_free(cpu); ggml_backend_free(meta) })
  # the rules run from C, out of reach of expect_warning(); read the log instead
  attr(meta, "meta_split_log")$emit <- FALSE
  out <- run_graph(meta, list(meta, cpu), read_mm)
  out$warnings <- attr(meta, "meta_split_log")$warnings
  out$applied <- attr(meta, "meta_split_log")$applied
  out
}

# w [K, M] "rows" = ne[1] = M (output features): mm [M, N] comes out split along
# its axis 0, so dy [M, N] must be split along ITS axis 0 = "cols".
COL_PAR <- list(w = "rows", dy = "cols")
ROW_PAR <- list(w = "cols", x = "cols")    # both along K: partial sums, reduced

test_that("ggml_meta_backend on 2x CPU: mirror / column / row parallel match plain CPU", {
  cpu_dev_or_skip()
  expect_same_as_cpu(run_meta_backend(c("CPU", "CPU"), list()))
  out <- run_meta_backend(c("CPU", "CPU"), COL_PAR)
  expect_same_as_cpu(out)
  expect_identical(out$applied$w, "rows")
  expect_identical(out$applied$x, "mirror")
  expect_same_as_cpu(run_meta_backend(c("CPU", "CPU"), ROW_PAR, read_mm = FALSE))
})

test_that("ggml_meta_backend accepts uneven weights and a split function", {
  cpu_dev_or_skip()
  expect_same_as_cpu(run_meta_backend(c("CPU", "CPU"),
    list(w = list(axis = "cols", weights = c(3, 1)),
         x = list(axis = "cols", weights = c(3, 1))), read_mm = FALSE))
  f <- function(name, ne) if (name %in% c("w", "x")) "cols" else NULL
  expect_same_as_cpu(run_meta_backend(c("CPU", "CPU"), f, read_mm = FALSE))
})

test_that("ggml_meta_backend: a rule that cannot apply mirrors that tensor with a warning", {
  cpu_dev_or_skip()
  # w is 2-D: ne[2] = 1 cannot be split over 2 devices
  out <- run_meta_backend(c("CPU", "CPU"), list(w = "axis2"))
  expect_gte(length(out$warnings), 1L)   # the rule is asked more than once per tensor
  expect_match(out$warnings, "'w'.*cannot split")
  expect_same_as_cpu(out)
  expect_identical(out$applied$w, "mirror")
  # a split function that errors affects only its own tensor
  f <- function(name, ne) if (name == "w") stop("boom") else NULL
  out <- run_meta_backend(c("CPU", "CPU"), f)
  expect_match(out$warnings, "boom")
  expect_identical(out$applied$w, "mirror")
  expect_identical(out$applied$x, "mirror")
  expect_same_as_cpu(out)
})

test_that("ggml_meta_backend validates its arguments up front", {
  expect_error(ggml_meta_backend(c("CPU", "CPU"), list(w = "diagonal")), "unknown split")
  expect_error(ggml_meta_backend(c("CPU", "CPU"), list("rows")), "named list")
  expect_error(ggml_meta_backend(c("CPU", "CPU"), default = "nope"), "unknown split")
  expect_error(ggml_meta_backend("NoSuchDevice0"), "no device named")
  expect_error(ggml_meta_backend(-1), "non-negative")
})

test_that("ggml_meta_backend on 2x the same Vulkan device matches plain CPU", {
  skip_on_cran()
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "Vulkan not available")
  expect_same_as_cpu(run_meta_backend(c(0L, 0L), list()), tol = 1e-4)
  expect_same_as_cpu(run_meta_backend(c(0L, 0L), COL_PAR), tol = 1e-4)
  expect_same_as_cpu(run_meta_backend(c(0L, 0L), ROW_PAR, read_mm = FALSE), tol = 1e-4)
})

# ---- ggml_tp_forward(): whole-graph TP from R arrays ----

set.seed(11)
N_IN <- 16L; N_HID <- 32L; N_OUT <- 8L; BATCH <- 4L
W1 <- array(rnorm(N_IN * N_HID), c(N_IN, N_HID))
W2 <- array(rnorm(N_HID * N_OUT), c(N_HID, N_OUT))
X  <- array(rnorm(N_IN * BATCH), c(N_IN, BATCH))
MLP_REF <- t(W2) %*% pmax(t(W1) %*% X, 0)
mlp <- function(ctx, input, w)
  ggml_mul_mat(ctx, w$w2, ggml_relu(ctx, ggml_mul_mat(ctx, w$w1, input)))
MEGATRON <- list(w1 = "rows", w2 = "cols")

tp_mlp <- function(devices, split, ...) {
  ggml_tp_forward(weights = list(w1 = W1, w2 = W2), forward = mlp, x = X,
                  devices = devices, split = split, ...)
}

test_that("ggml_tp_forward MLP on 2x CPU: mirror and Megatron split match R", {
  cpu_dev_or_skip()
  for (s in list(list(), MEGATRON, list(w1 = "cols", input = "cols"))) {
    y <- tp_mlp(c("CPU", "CPU"), s)
    expect_identical(dim(y), c(N_OUT, BATCH))
    expect_equal(y, MLP_REF, tolerance = 1e-5, ignore_attr = TRUE)
  }
})

test_that("ggml_tp_forward with F16 weights stays close to R", {
  cpu_dev_or_skip()
  y <- tp_mlp(c("CPU", "CPU"), MEGATRON, type = GGML_TYPE_F16)
  expect_equal(y, MLP_REF, tolerance = 1e-2, ignore_attr = TRUE)
})

test_that("ggml_tp_forward returns a vector for a 1-D output", {
  cpu_dev_or_skip()
  x1 <- X[, 1]
  y <- ggml_tp_forward(list(w1 = W1, w2 = W2), mlp, x = x1,
                       devices = c("CPU", "CPU"), split = MEGATRON)
  expect_null(dim(y))
  expect_equal(y, as.vector(MLP_REF[, 1]), tolerance = 1e-5)
})

test_that("ggml_tp_forward signals a split fallback as a catchable warning", {
  cpu_dev_or_skip()
  expect_warning(y <- tp_mlp(c("CPU", "CPU"), list(w1 = "axis2")), "'w1'.*cannot split")
  expect_equal(y, MLP_REF, tolerance = 1e-5, ignore_attr = TRUE)
})

test_that("ggml_tp_forward validates its arguments", {
  f <- function(...) ggml_tp_forward(..., devices = c("CPU", "CPU"))
  expect_error(f(list(W1), mlp, X), "unique names")
  expect_error(f(list(input = W1), mlp, X), "reserved")
  expect_error(f(list(w1 = W1), "mlp", X), "forward must be")
  expect_error(f(list(w1 = W1), mlp, X, type = GGML_TYPE_Q4_0), "F32 or GGML_TYPE_F16")
  expect_error(f(list(w1 = array(0, c(1, 1, 1, 1, 2))), mlp, X), "at most 4")
  expect_error(f(list(w1 = W1, w2 = W2), function(ctx, input, w) 42, X), "ggml tensor")
})

test_that("ggml_tp_forward MLP on 2x the same Vulkan device matches R", {
  skip_on_cran()
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "Vulkan not available")
  for (s in list(list(), MEGATRON)) {
    y <- tp_mlp(c(0L, 0L), s)
    # f32 matmul on Vulkan may run at f16 precision without GGML_PREC_F32
    expect_equal(y, MLP_REF, tolerance = 1e-2, ignore_attr = TRUE)
  }
})

test_that("ggml_tp_forward MLP on 2 real Vulkan devices matches R", {
  skip_on_cran()
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 2L,
              "need >= 2 Vulkan devices")
  y <- tp_mlp(c(0L, 1L), MEGATRON)
  expect_equal(y, MLP_REF, tolerance = 1e-2, ignore_attr = TRUE)
})
