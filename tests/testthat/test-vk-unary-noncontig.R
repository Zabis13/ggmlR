# Vulkan unary ops on contiguous and non-contiguous inputs (unary.comp, 9.9).
#
# Until the switch to ggml 0.27's unary.comp our unary shaders indexed flat, so
# a view was read at the wrong elements (and supports_op had to refuse it).
# unary.comp addresses every element through the tensor's strides, so the
# Vulkan backend now takes views itself. The graphs run on the GPU backend
# directly and are checked against references written in R. Every layout keeps
# its rows contiguous: ggml_unary_impl asserts that (as in 0.27).

# es: bytes per element of the input type
make_layouts <- function(es) list(
  # [6, 5, 3] as is
  contiguous = list(shape = c(6, 5, 3), view = function(ctx, x) x,
                    pick = function(a) a),
  # 4 of every 6 floats per row: rows contiguous, padded after each
  padded     = list(shape = c(6, 5, 3),
                    view = function(ctx, x) ggml_view_3d(ctx, x, 4, 5, 3, 6 * es, 6 * 5 * es, 0),
                    pick = function(a) a[1:4, , , drop = FALSE]),
  # dims 1 and 2 swapped: rows stay contiguous (GGML_OP_UNARY requires it,
  # ggml_is_contiguous_rows), the planes do not follow the shape any more
  permuted   = list(shape = c(6, 5, 3), view = function(ctx, x) ggml_permute(ctx, x, 0, 2, 1, 3),
                    pick = function(a) aperm(a, c(1, 3, 2)))
)

softplus <- function(x) ifelse(x > 20, x, log1p(exp(x)))
XI <- list(an = 0.8, ap = 0.6, beta = 0.5, eps = -1e-6)

ops <- list(
  abs         = list(f = ggml_abs,         ref = abs),
  neg         = list(f = ggml_neg,         ref = function(x) -x),
  sqr         = list(f = ggml_sqr,         ref = function(x) x^2),
  sqrt        = list(f = ggml_sqrt,        ref = sqrt, positive = TRUE),
  log         = list(f = ggml_log,         ref = log,  positive = TRUE),
  sin         = list(f = ggml_sin,         ref = sin),
  cos         = list(f = ggml_cos,         ref = cos),
  exp         = list(f = ggml_exp,         ref = exp),
  expm1       = list(f = ggml_expm1,       ref = expm1),
  relu        = list(f = ggml_relu,        ref = function(x) pmax(x, 0)),
  leaky_relu  = list(f = function(ctx, a) ggml_leaky_relu(ctx, a, negative_slope = 0.1),
                     ref = function(x) ifelse(x > 0, x, 0.1 * x)),
  elu         = list(f = ggml_elu,         ref = function(x) ifelse(x < 0, expm1(x), x)),
  sigmoid     = list(f = ggml_sigmoid,     ref = plogis),
  tanh        = list(f = ggml_tanh,        ref = tanh),
  silu        = list(f = ggml_silu,        ref = function(x) x * plogis(x)),
  gelu        = list(f = ggml_gelu,
                     ref = function(x) 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))),
  gelu_quick  = list(f = ggml_gelu_quick,  ref = function(x) x * plogis(1.702 * x)),
  gelu_erf    = list(f = ggml_gelu_erf,    ref = function(x) x * pnorm(x)),
  softplus    = list(f = ggml_softplus,    ref = softplus),
  hardswish   = list(f = ggml_hardswish,   ref = function(x) x * pmin(1, pmax(0, (x + 3) / 6))),
  hardsigmoid = list(f = ggml_hardsigmoid, ref = function(x) pmin(1, pmax(0, (x + 3) / 6))),
  step        = list(f = ggml_step,        ref = function(x) as.numeric(x > 0)),
  sgn         = list(f = ggml_sgn,         ref = sign),
  round       = list(f = ggml_round,       ref = function(x) sign(x) * floor(abs(x) + 0.5)),
  round_even  = list(f = ggml_round_even,  ref = round),        # R's round() is half-to-even
  ceil        = list(f = ggml_ceil,        ref = ceiling),
  floor       = list(f = ggml_floor,       ref = floor),
  trunc       = list(f = ggml_trunc,       ref = trunc),
  # ggml_clamp works in place (its result is a view of the input, with the
  # input's strides): ggml_cont gathers it into a dense tensor to read back
  clamp       = list(f = function(ctx, a) ggml_cont(ctx, ggml_clamp(ctx, a, -0.5, 0.75)),
                     ref = function(x) pmin(pmax(x, -0.5), 0.75)),
  xielu       = list(f = function(ctx, a) ggml_xielu(ctx, a, XI$an, XI$ap, XI$beta, XI$eps),
                     ref = function(x) {
                       an <- XI$beta + softplus(XI$an); ap <- softplus(XI$ap)
                       ifelse(x > 0, ap * x^2 + XI$beta * x,
                              an * (exp(pmin(x, XI$eps)) - 1 - x) + XI$beta * x)
                     })
)

run_vk <- function(gpu, type, shape, data, build) {
  ctx <- ggml_init(4 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  x <- ggml_new_tensor_3d(ctx, type, shape[1], shape[2], shape[3])
  out <- build(ctx, x)
  gf <- ggml_build_forward_expand(ctx, out)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, gpu)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE)
  ggml_backend_tensor_set_data(x, data)
  ggml_backend_graph_compute(gpu, gf)
  ggml_backend_tensor_get_data(out)
}

# Odd multiples of 1/256 in (-3, 3): exact in f16 as well as f32, and never 0
# or a rounding tie, so one R reference serves both types.
make_input <- function(n, positive) {
  set.seed(n)
  v <- (2 * sample(-384:383, n, replace = TRUE) + 1) / 256
  if (positive) abs(v) else v
}

for (type_name in c("f32", "f16")) {
  local({
    type <- if (type_name == "f32") GGML_TYPE_F32 else GGML_TYPE_F16
    tol  <- if (type_name == "f32") 1e-4 else 5e-3
    layouts <- make_layouts(if (type_name == "f32") 4 else 2)
    test_that(sprintf("Vulkan unary ops match R on contiguous, padded and permuted %s input", type_name), {
      skip_if_not(ggml_vulkan_available(), "no Vulkan device")
      gpu <- ggml_vulkan_init(0L)
      on.exit(ggml_backend_free(gpu))
      for (nm in names(ops)) for (ln in names(layouts)) {
        op <- ops[[nm]]; lay <- layouts[[ln]]
        n <- prod(lay$shape)
        x <- make_input(n, isTRUE(op$positive))
        got <- run_vk(gpu, type, lay$shape, x, function(ctx, t) op$f(ctx, lay$view(ctx, t)))
        want <- as.vector(op$ref(lay$pick(array(x, lay$shape))))
        expect_equal(got, want, tolerance = tol, label = sprintf("%s %s %s", nm, ln, type_name))
      }
    })
  })
}

test_that("Vulkan supports_op accepts unary ops on views", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan device")
  gpu <- ggml_vulkan_init(0L)
  on.exit(ggml_backend_free(gpu), add = TRUE)
  dev <- ggml_backend_get_device(gpu)
  ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  layouts <- make_layouts(4)
  on.exit(ggml_free(ctx), add = TRUE)
  x <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 6, 5, 3)
  for (nm in names(ops)) for (ln in names(layouts)) {
    expect_true(ggml_backend_dev_supports_op(dev, ops[[nm]]$f(ctx, layouts[[ln]]$view(ctx, x))),
                label = paste(nm, ln))
  }
})

test_that("Vulkan rounding ties and step at zero follow the CPU rules", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan device")
  gpu <- ggml_vulkan_init(0L)
  on.exit(ggml_backend_free(gpu))
  x <- c(-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5, 3.5)
  one <- function(f) run_vk(gpu, GGML_TYPE_F32, c(8, 1, 1), x, f)
  expect_equal(one(ggml_round),      c(-3, -2, -1, 0, 1, 2, 3, 4))   # ties away from zero
  expect_equal(one(ggml_round_even), c(-2, -2,  0, 0, 0, 2, 2, 4))   # ties to even
  expect_equal(one(ggml_step),       c( 0,  0,  0, 0, 1, 1, 1, 1))   # strictly > 0
})
