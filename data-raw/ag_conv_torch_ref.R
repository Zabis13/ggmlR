# Reference values for tests/testthat/test-ag-conv.R from libtorch (R torch),
# in float64. Run from the package root:
#   Rscript data-raw/ag_conv_torch_ref.R
# Writes tests/testthat/fixtures/ag_conv_torch.rds. Everything is stored in
# torch's own layout (x [B, C, H, W], weight [OC, C, KH, KW], output and its
# gradient weights G [B, OC, OH, OW]); the test converts to ggmlR's layout
# itself, so the conversion is what the test checks.
suppressMessages(library(torch))
tt <- function(a, grad = FALSE) torch_tensor(a, dtype = torch_float64(), requires_grad = grad)

configs <- list(
  list(name = "3x3, batch 2",                 B = 2L, C = 2L, H = 7L,  W = 6L,  OC = 3L, k = c(3L, 3L), s = c(1L, 1L), p = c(0L, 0L), d = c(1L, 1L)),
  list(name = "non-square kernel and image",  B = 3L, C = 3L, H = 9L,  W = 11L, OC = 2L, k = c(2L, 4L), s = c(2L, 1L), p = c(1L, 2L), d = c(1L, 1L)),
  list(name = "padding and dilation",         B = 2L, C = 2L, H = 10L, W = 9L,  OC = 2L, k = c(3L, 3L), s = c(1L, 1L), p = c(2L, 2L), d = c(2L, 2L)),
  list(name = "uncovered tail",               B = 2L, C = 1L, H = 11L, W = 10L, OC = 2L, k = c(4L, 3L), s = c(3L, 4L), p = c(0L, 0L), d = c(1L, 1L)),
  list(name = "Nature-CNN conv1, stride 4",   B = 2L, C = 4L, H = 20L, W = 20L, OC = 3L, k = c(8L, 8L), s = c(4L, 4L), p = c(0L, 0L), d = c(1L, 1L)))

set.seed(20261120)
ref <- lapply(configs, function(cf) {
  x <- array(runif(cf$B * cf$C * cf$H * cf$W, -1, 1), c(cf$B, cf$C, cf$H, cf$W))
  w <- array(rnorm(cf$OC * cf$C * prod(cf$k), sd = 0.3), c(cf$OC, cf$C, cf$k))
  b <- rnorm(cf$OC)
  xt <- tt(x, TRUE); wt <- tt(w, TRUE); bt <- tt(b, TRUE)
  y <- nnf_conv2d(xt, wt, bt, stride = cf$s, padding = cf$p, dilation = cf$d)
  G <- array(rnorm(prod(y$shape)), y$shape)
  (y * tt(G))$sum()$backward()
  c(cf, list(x = x, w = w, b = b, G = G, y = as.array(y$detach()),
             dx = as.array(xt$grad), dw = as.array(wt$grad), db = as.array(bt$grad)))
})
attr(ref, "torch_version") <- as.character(packageVersion("torch"))
saveRDS(ref, "tests/testthat/fixtures/ag_conv_torch.rds", version = 2)
cat("wrote", length(ref), "configurations\n")
