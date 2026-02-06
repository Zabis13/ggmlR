# Test CPU Feature Detection and Tensor Layout Functions

test_that("CPU feature detection functions work", {
  # All functions should return logical or integer values without error
  expect_type(ggml_cpu_has_sse3(), "logical")
  expect_type(ggml_cpu_has_ssse3(), "logical")
  expect_type(ggml_cpu_has_avx(), "logical")
  expect_type(ggml_cpu_has_avx_vnni(), "logical")
  expect_type(ggml_cpu_has_avx2(), "logical")
  expect_type(ggml_cpu_has_bmi2(), "logical")
  expect_type(ggml_cpu_has_f16c(), "logical")
  expect_type(ggml_cpu_has_fma(), "logical")
  expect_type(ggml_cpu_has_avx512(), "logical")
  expect_type(ggml_cpu_has_avx512_vbmi(), "logical")
  expect_type(ggml_cpu_has_avx512_vnni(), "logical")
  expect_type(ggml_cpu_has_avx512_bf16(), "logical")
  expect_type(ggml_cpu_has_amx_int8(), "logical")
})

test_that("ARM feature detection functions work", {
  expect_type(ggml_cpu_has_neon(), "logical")
  expect_type(ggml_cpu_has_arm_fma(), "logical")
  expect_type(ggml_cpu_has_fp16_va(), "logical")
  expect_type(ggml_cpu_has_dotprod(), "logical")
  expect_type(ggml_cpu_has_matmul_int8(), "logical")
  expect_type(ggml_cpu_has_sve(), "logical")
  expect_type(ggml_cpu_get_sve_cnt(), "integer")
  expect_type(ggml_cpu_has_sme(), "logical")
})

test_that("Other architecture feature detection works", {
  expect_type(ggml_cpu_has_riscv_v(), "logical")
  expect_type(ggml_cpu_get_rvv_vlen(), "integer")
  expect_type(ggml_cpu_has_vsx(), "logical")
  expect_type(ggml_cpu_has_vxe(), "logical")
  expect_type(ggml_cpu_has_wasm_simd(), "logical")
  expect_type(ggml_cpu_has_llamafile(), "logical")
})

test_that("ggml_cpu_features returns all features", {
  features <- ggml_cpu_features()
  expect_type(features, "list")
  expect_true("sse3" %in% names(features))
  expect_true("avx2" %in% names(features))
  expect_true("neon" %in% names(features))
  expect_true("riscv_v" %in% names(features))
  expect_true("llamafile" %in% names(features))
  # Should have 28 features total
  expect_gte(length(features), 25)
})

test_that("Tensor contiguity functions work", {
  ctx <- ggml_init(1024 * 1024)

  # Create a contiguous tensor
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8)
  ggml_set_f32(a, rnorm(32))

  # Contiguous tensor tests
  expect_true(ggml_is_contiguous_0(a))
  expect_true(ggml_is_contiguous_1(a))
  expect_true(ggml_is_contiguous_2(a))
  expect_true(ggml_is_contiguously_allocated(a))
  expect_true(ggml_is_contiguous_rows(a))

  # Create a transposed tensor (view, not contiguous at dim 0)
  b <- ggml_transpose(ctx, a)
  expect_false(ggml_is_contiguous_0(b))

  ggml_free(ctx)
})

test_that("Tensor stride comparison works", {
  ctx <- ggml_init(1024 * 1024)

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8)
  c <- ggml_transpose(ctx, a)

  # Same shape tensors have same stride
  expect_true(ggml_are_same_stride(a, b))

  # Transposed tensor has different stride
  expect_false(ggml_are_same_stride(a, c))

  ggml_free(ctx)
})

test_that("ggml_can_repeat works", {
  ctx <- ggml_init(1024 * 1024)

  # Small tensor can be repeated to match larger tensor
  small <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  large <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8)

  expect_true(ggml_can_repeat(small, large))

  # Cannot repeat if dimensions don't align
  wrong <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  expect_false(ggml_can_repeat(wrong, large))

  ggml_free(ctx)
})

test_that("ggml_count_equal creates graph operation", {
  ctx <- ggml_init(1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 10)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 10)

  # Should create a tensor for the count_equal operation
  result <- ggml_count_equal(ctx, a, b)
  expect_false(is.null(result))

  ggml_free(ctx)
})
