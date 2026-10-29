# Vulkan UNARY+MUL fusion (gelu/sigmoid/silu/softplus followed by mul): the fused
# shader against the CPU, which runs the two ops separately. The fusion counter
# tells the fused path ran, not only that the numbers agree. Covers f32/f16, same
# shape (norepeat), a broadcast multiplier on either side of the mul, the unary on
# the mul's src1 tiling into src0 (op_on_b), and a 5D tensor.

unary_mul_count <- function() .Call("R_ggml_vk_unary_mul_fusion_count", PACKAGE = "ggmlR")

unary_mul_ops <- list(gelu = ggml_gelu, sigmoid = ggml_sigmoid, silu = ggml_silu, softplus = ggml_softplus)

# The CPU's f32 gelu looks values up in an f16 table (GGML_GELU_FP16), ~1e-3 off;
# Vulkan evaluates the formula. For gelu f32 the CPU gets this exact gelu (tanh
# form, as ggml) applied in R and an identity in place of the unary, so it still
# does the mul and its broadcast.
gelu_exact <- function(x) 0.5 * x * (1 + tanh(sqrt(2 / pi) * x * (1 + 0.044715 * x^2)))

# inputs: list of list(shape, data); all tensors of one type
run_unary_mul <- function(backend, type, inputs, build) {
  ctx <- ggml_init(16 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  tensors <- lapply(inputs, function(x) ggml_new_tensor(ctx, type, length(x$shape), x$shape))
  out <- do.call(build, c(list(ctx), tensors))
  gf <- ggml_build_forward_expand(ctx, out)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  for (i in seq_along(inputs)) ggml_backend_tensor_set_data(tensors[[i]], inputs[[i]]$data)
  ggml_backend_graph_compute(backend, gf)
  ggml_backend_tensor_get_data(out)
}

unary_mul_cases <- function(f) list(
  same_shape   = list(shapes = list(c(64, 6, 3), c(64, 6, 3)),
                      build  = function(ctx, a, b) ggml_mul(ctx, f(ctx, a), b)),
  bcast_other  = list(shapes = list(c(64, 6, 3), c(64, 1, 1)),
                      build  = function(ctx, a, b) ggml_mul(ctx, f(ctx, a), b)),
  unary_as_b   = list(shapes = list(c(64, 6, 3), c(64, 6, 3)),
                      build  = function(ctx, a, b) ggml_mul(ctx, b, f(ctx, a))),
  op_on_b      = list(shapes = list(c(64, 1, 3), c(64, 6, 3)),
                      build  = function(ctx, a, b) ggml_mul(ctx, b, f(ctx, a))),
  five_d       = list(shapes = list(c(16, 3, 2, 2, 3), c(16, 1, 2, 1, 3)),
                      build  = function(ctx, a, b) ggml_mul(ctx, f(ctx, a), b))
)

for (type_name in c("f32", "f16")) {
  local({
    type <- if (type_name == "f32") GGML_TYPE_F32 else GGML_TYPE_F16
    tol  <- if (type_name == "f32") 1e-5 else 5e-3
    test_that(sprintf("Vulkan UNARY+MUL fuses and matches the CPU (%s)", type_name), {
      skip_if_not(ggml_vulkan_available(), "no Vulkan device")
      gpu <- ggml_vulkan_init(0L)
      on.exit(ggml_backend_free(gpu))
      cpu <- ggml_backend_cpu_init()
      on.exit(ggml_backend_free(cpu), add = TRUE)
      ggml_backend_cpu_set_n_threads(cpu, 2L)
      set.seed(41)
      for (nm in names(unary_mul_ops)) {
        cases <- unary_mul_cases(unary_mul_ops[[nm]])
        exact_gelu <- nm == "gelu" && type_name == "f32"
        ref_cases <- if (exact_gelu) unary_mul_cases(function(ctx, a) a) else cases
        for (cn in names(cases)) {
          cc <- cases[[cn]]
          inputs <- lapply(cc$shapes, function(s) list(shape = s, data = runif(prod(s), -3, 3)))
          ref_inputs <- inputs
          if (exact_gelu) ref_inputs[[1]]$data <- gelu_exact(inputs[[1]]$data)
          want <- run_unary_mul(cpu, type, ref_inputs, ref_cases[[cn]]$build)
          c0 <- unary_mul_count()
          got <- run_unary_mul(gpu, type, inputs, cc$build)
          lab <- sprintf("%s %s %s", nm, cn, type_name)
          expect_equal(got, want, tolerance = tol, label = lab)
          expect_equal(unary_mul_count() - c0, 1, label = paste(lab, "fusions"))
        }
      }
    })
  })
}

test_that("Vulkan does not fuse UNARY+MUL when the unary result is used again", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan device")
  gpu <- ggml_vulkan_init(0L)
  on.exit(ggml_backend_free(gpu))
  cpu <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(cpu), add = TRUE)
  ggml_backend_cpu_set_n_threads(cpu, 2L)
  set.seed(42)
  inputs <- list(list(shape = c(64, 6), data = runif(384, -3, 3)),
                 list(shape = c(64, 6), data = runif(384, -3, 3)))
  # silu(a) feeds both the mul and the add: fusing would drop the add's input
  build <- function(ctx, a, b) {
    s <- ggml_silu(ctx, a)
    ggml_add(ctx, ggml_mul(ctx, s, b), s)
  }
  want <- run_unary_mul(cpu, GGML_TYPE_F32, inputs, build)
  c0 <- unary_mul_count()
  got <- run_unary_mul(gpu, GGML_TYPE_F32, inputs, build)
  expect_equal(got, want, tolerance = 1e-5)
  expect_equal(unary_mul_count() - c0, 0)
})
