# Extracted from test-ag-flash-batch.R:148

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "ggmlR", path = "..")
attach(test_env, warn.conflicts = FALSE)

# prequel ----------------------------------------------------------------------
ag_flash_attention <- ggmlR:::ag_flash_attention
with_cpu_backend <- function(code) {
  st <- ggmlR:::.ag_device_state
  prev_dev <- st$device
  prev_bk  <- st$backend
  on.exit({
    st$device  <- prev_dev
    st$backend <- prev_bk
  }, add = TRUE)
  st$device  <- "cpu"
  st$backend <- ggml_backend_cpu_init()
  ggml_backend_cpu_set_n_threads(st$backend, ggml_get_n_threads())
  eval(substitute(code), envir = parent.frame())
}
cols_of <- function(b, s) ((b - 1L) * s + 1L):(b * s)
grads_of <- function(Q, K, V, G, h, batch = 1L, causal = FALSE) {
  qq <- ag_param(Q); kk <- ag_param(K); vv <- ag_param(V)
  with_grad_tape({
    out  <- ag_flash_attention(qq, kk, vv, n_heads = h, batch = batch,
                               causal = causal)
    loss <- ag_sum(ag_mul(out, ag_tensor(G)))
  })
  backward(loss)
  list(out = ggmlR:::.ag_as_matrix(ggmlR:::.ag_data(out)),
       q = ggmlR:::.ag_as_matrix(qq$grad),
       k = ggmlR:::.ag_as_matrix(kk$grad),
       v = ggmlR:::.ag_as_matrix(vv$grad))
}
grads_per_sample <- function(Q, K, V, G, h, B, s, causal = FALSE) {
  r <- list(out = Q * 0, q = Q * 0, k = K * 0, v = V * 0)
  for (b in seq_len(B)) {
    cc <- cols_of(b, s)
    g1 <- grads_of(Q[, cc, drop = FALSE], K[, cc, drop = FALSE],
                   V[, cc, drop = FALSE], G[, cc, drop = FALSE], h,
                   causal = causal)
    for (nm in names(r)) r[[nm]][, cc] <- g1[[nm]]
  }
  r
}
mk <- function(d, n) matrix(runif(d * n, -1, 1), d, n)
gpu_graph_grads <- function(Q, K, V, G, h, B) {
  old <- ggmlR:::ag_backward_graph(TRUE)
  withr::defer(ggmlR:::ag_backward_graph(old))
  r <- grads_of(Q, K, V, G, h, batch = B)
  r$path <- ggmlR:::ag_backward_path()
  r
}

# test -------------------------------------------------------------------------
skip_if_not(ggml_vulkan_available(), "Vulkan not available")
skip_if_not(ggml_vulkan_device_count() > 0, "No Vulkan devices")
skip_on_cran()
local_cpu_device()
d <- 8L
h <- 2L
s <- 4L
B <- 8200L
set.seed(9L)
Q <- mk(d, B * s)
K <- mk(d, B * s)
V <- mk(d, B * s)
G <- mk(d, B * s)
ref <- with_cpu_backend(grads_of(Q, K, V, G, h, batch = B))
ag_device("gpu")
ag_dtype("f32")
got <- gpu_graph_grads(Q, K, V, G, h, B)
expect_identical(got$path, "graph")
for (nm in c("out", "q", "k", "v"))
    expect_equal(got[[nm]], ref[[nm]], tolerance = 1e-4, info = nm)
