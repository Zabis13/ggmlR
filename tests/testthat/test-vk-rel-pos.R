# Vulkan kernel rel_pos.comp for GET_REL_POS / ADD_REL_POS (TODO 9.5-gpu-gaps).
# Each case runs on the CPU and on Vulkan; Vulkan must agree bit for bit (GET
# only moves f16 values, ADD performs the two adds in the CPU kernel's order),
# both are checked against a reference written in R, and the Vulkan device must
# actually offer the op: without the supports_op check a missing kernel would
# fall back to the CPU and pass.

skip_if_no_vk <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "Vulkan not available")
}

# inputs: list of list(type, shape, data). build(ctx, tensors) -> output.
# Returns list(out, supported).
run_rel_pos <- function(build, inputs, backend = c("cpu", "vulkan")) {
  backend <- match.arg(backend)
  ctx <- ggml_init(64 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)
  b <- if (backend == "vulkan") ggml_vulkan_init(0L) else {
    cb <- ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(cb, 2L); cb
  }
  on.exit(ggml_backend_free(b), add = TRUE, after = FALSE)

  ts <- lapply(inputs, function(i) {
    t <- ggml_new_tensor(ctx, i$type, length(i$shape), i$shape)
    ggml_set_input(t)
    t
  })
  out <- build(ctx, ts)
  ggml_set_output(out)
  supported <- ggml_backend_dev_supports_op(ggml_backend_get_device(b), out)

  buf <- ggml_backend_alloc_ctx_tensors(ctx, b)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  for (k in seq_along(ts)) ggml_backend_tensor_set_data(ts[[k]], inputs[[k]]$data)
  ggml_backend_graph_compute(b, ggml_build_forward_expand(ctx, out))
  list(out = ggml_backend_tensor_get_data(out), supported = supported)
}

both_rel_pos <- function(build, inputs) {
  cpu <- run_rel_pos(build, inputs, "cpu")
  vk  <- run_rel_pos(build, inputs, "vulkan")
  expect_true(vk$supported)
  expect_identical(vk$out, cpu$out)
  cpu$out
}

# ---- GET_REL_POS ------------------------------------------------------------

# a: [C, 2w-1] -> [C, w, w]; 1-based: out[i0, i1, i2] = a[i0, w - i1 + i2]
get_rel_pos_ref <- function(x, C, w) {
  a <- array(x, c(C, 2 * w - 1))
  out <- array(0, c(C, w, w))
  for (i2 in seq_len(w)) for (i1 in seq_len(w)) out[, i1, i2] <- a[, w - i1 + i2]
  as.vector(out)
}

test_that("Vulkan GET_REL_POS matches the CPU and the definition", {
  skip_if_no_vk()
  set.seed(2)
  for (case in list(list(C = 1L,  w = 1L),
                    list(C = 3L,  w = 4L),
                    list(C = 16L, w = 14L),
                    list(C = 7L,  w = 33L))) {
    n <- case$C * (2 * case$w - 1)
    x <- round(runif(n, -4, 4) * 8) / 8   # exact in f16
    got <- both_rel_pos(function(ctx, t) ggml_get_rel_pos(ctx, t[[1]], case$w, case$w),
                        list(list(type = GGML_TYPE_F16, shape = c(case$C, 2 * case$w - 1), data = x)))
    expect_identical(got, get_rel_pos_ref(x, case$C, case$w))
  }
})

test_that("Vulkan GET_REL_POS above 512*65535 elements (NVIDIA x-limit)", {
  skip_if_no_vk()
  C <- 64L; w <- 725L                      # dst 64*725*725 = 33 640 000 > 33 553 920
  x <- round(runif(C * (2 * w - 1), -4, 4) * 8) / 8
  both_rel_pos(function(ctx, t) ggml_get_rel_pos(ctx, t[[1]], w, w),
               list(list(type = GGML_TYPE_F16, shape = c(C, 2 * w - 1), data = x)))
})

# ---- ADD_REL_POS ------------------------------------------------------------

# a: [K*K, n1*n2, n3]; pw, ph: [K, n1, n2, n3]; r = row of pw/ph (i11, i12, i13).
# out[(b, a), r] = a + ph[a, r] + pw[b, r]   (b fastest, 1-based)
add_rel_pos_ref <- function(x, pw, ph, K, rows) {
  a  <- array(x,  c(K, K, rows))
  hw <- array(pw, c(K, rows))
  hh <- array(ph, c(K, rows))
  for (r in seq_len(rows)) a[, , r] <- a[, , r] + outer(hw[, r], hh[, r], "+")
  as.vector(a)
}

add_case <- function(K, n1, n2, n3) {
  rows <- n1 * n2 * n3
  list(K = K, rows = rows,
       inputs = list(
         list(type = GGML_TYPE_F32, shape = c(K * K, n1 * n2, n3), data = runif(K * K * rows, -1, 1)),
         list(type = GGML_TYPE_F32, shape = c(K, n1, n2, n3),      data = runif(K * rows, -1, 1)),
         list(type = GGML_TYPE_F32, shape = c(K, n1, n2, n3),      data = runif(K * rows, -1, 1))))
}

add_build <- function(ctx, t) ggml_add_rel_pos(ctx, t[[1]], t[[2]], t[[3]])

test_that("Vulkan ADD_REL_POS matches the CPU bit for bit and the definition", {
  skip_if_no_vk()
  set.seed(3)
  for (cs in list(c(1, 1, 1, 1), c(3, 2, 2, 1), c(14, 14, 14, 2), c(5, 3, 7, 4))) {
    cc <- do.call(add_case, as.list(cs))
    got <- both_rel_pos(add_build, cc$inputs)
    expect_equal(got,
                 add_rel_pos_ref(cc$inputs[[1]]$data, cc$inputs[[2]]$data,
                                 cc$inputs[[3]]$data, cc$K, cc$rows),
                 tolerance = 1e-6)   # f32 vs double
  }
})

test_that("Vulkan ADD_REL_POS above 512*65535 elements (NVIDIA x-limit)", {
  skip_if_no_vk()
  set.seed(4)
  cc <- add_case(64, 64, 64, 2)            # 64*64 * 64*64 * 2 = 33 554 432 > 33 553 920
  both_rel_pos(add_build, cc$inputs)
})
