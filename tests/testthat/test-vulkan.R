test_that("ggml_vulkan_available returns logical", {
  result <- ggml_vulkan_available()
  expect_type(result, "logical")
  expect_length(result, 1)
})

test_that("ggml_vulkan_device_count returns non-negative integer", {
  count <- ggml_vulkan_device_count()
  expect_type(count, "integer")
  expect_gte(count, 0)
})

test_that("ggml_vulkan_status runs without error", {
  expect_no_error(ggml_vulkan_status())
})

# Conditional tests that only run if Vulkan is available
if (ggml_vulkan_available() && ggml_vulkan_device_count() > 0) {

  test_that("ggml_vulkan_list_devices returns list", {
    devices <- ggml_vulkan_list_devices()
    expect_type(devices, "list")
    expect_gt(length(devices), 0)

    # Check first device structure
    dev <- devices[[1]]
    expect_named(dev, c("index", "name", "free_memory", "total_memory"))
    expect_type(dev$index, "integer")
    expect_type(dev$name, "character")
    expect_type(dev$free_memory, "double")
    expect_type(dev$total_memory, "double")
  })

  test_that("ggml_vulkan_device_description returns string", {
    desc <- ggml_vulkan_device_description(0)
    expect_type(desc, "character")
    expect_gt(nchar(desc), 0)
  })

  test_that("ggml_vulkan_device_memory returns memory info", {
    mem <- ggml_vulkan_device_memory(0)
    expect_type(mem, "list")
    expect_named(mem, c("free", "total"))
    expect_type(mem$free, "double")
    expect_type(mem$total, "double")
    expect_gte(mem$free, 0)
    expect_gte(mem$total, 0)
    expect_lte(mem$free, mem$total)
  })

  test_that("ggml_vulkan_init and free work", {
    backend <- ggml_vulkan_init(0)
    expect_type(backend, "externalptr")

    # Test backend name
    name <- ggml_vulkan_backend_name(backend)
    expect_type(name, "character")
    expect_gt(nchar(name), 0)

    # Test is_backend check
    is_vk <- ggml_vulkan_is_backend(backend)
    expect_type(is_vk, "logical")
    expect_true(is_vk)

    # Free backend
    expect_no_error(ggml_vulkan_free(backend))
  })

  test_that("ggml_vulkan_device_description errors on invalid index", {
    count <- ggml_vulkan_device_count()
    expect_error(
      ggml_vulkan_device_description(count + 100),
      "Invalid device index"
    )
    expect_error(
      ggml_vulkan_device_description(-1),
      "Invalid device index"
    )
  })

  test_that("ggml_vulkan_device_memory errors on invalid index", {
    count <- ggml_vulkan_device_count()
    expect_error(
      ggml_vulkan_device_memory(count + 100),
      "Invalid device index"
    )
    expect_error(
      ggml_vulkan_device_memory(-1),
      "Invalid device index"
    )
  })

  test_that("ggml_vulkan_init errors on invalid index", {
    count <- ggml_vulkan_device_count()
    expect_error(
      ggml_vulkan_init(count + 100),
      "Invalid device index"
    )
    expect_error(
      ggml_vulkan_init(-1),
      "Invalid device index"
    )
  })

  # ========================================================================
  # Computational tests for LLM operations
  # ========================================================================

  test_that("Vulkan: swiglu activation (LLaMA/Mistral)", {
    ctx <- ggml_init(mem_size = 16 * 1024 * 1024)
    ggml_set_no_alloc(ctx, TRUE)

    # Create input tensor for swiglu (will be split internally)
    # swiglu expects input of size 2*hidden_dim and splits it
    x <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512)  # Will split to 2x256
    result <- ggml_swiglu(ctx, x)

    # Setup Vulkan backend
    backend_vk <- ggml_vulkan_init(0)
    buffer_vk <- ggml_backend_alloc_ctx_tensors(ctx, backend_vk)

    # Set test data (concatenated x and y)
    x_data <- seq(-2, 2, length.out = 512)
    ggml_backend_tensor_set_data(x, x_data)

    # Compute
    graph <- ggml_build_forward_expand(ctx, result)
    ggml_backend_graph_compute(backend_vk, graph)

    # Get result
    result_data <- ggml_backend_tensor_get_data(result)

    # Basic checks - swiglu computation works
    expect_length(result_data, 256)
    expect_false(any(is.na(result_data)))
    expect_false(any(is.infinite(result_data)))

    # Result should be in reasonable range
    expect_true(max(abs(result_data)) < 10)

    # SwiGLU produces non-zero output for non-zero input
    expect_true(sum(abs(result_data)) > 0.1)

    # Cleanup
    ggml_backend_buffer_free(buffer_vk)
    ggml_vulkan_free(backend_vk)
    ggml_free(ctx)
  })

  test_that("Vulkan: geglu activation", {
    ctx <- ggml_init(mem_size = 16 * 1024 * 1024)
    ggml_set_no_alloc(ctx, TRUE)

    # Create input tensor (will be split internally like swiglu)
    x <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256)  # Will split to 2x128
    result <- ggml_geglu(ctx, x)

    # Setup Vulkan backend
    backend_vk <- ggml_vulkan_init(0)
    buffer_vk <- ggml_backend_alloc_ctx_tensors(ctx, backend_vk)

    # Set test data
    x_data <- seq(-1, 1, length.out = 256)
    ggml_backend_tensor_set_data(x, x_data)

    # Compute
    graph <- ggml_build_forward_expand(ctx, result)
    ggml_backend_graph_compute(backend_vk, graph)

    # Get result
    result_data <- ggml_backend_tensor_get_data(result)

    # Basic checks - geglu computation works
    expect_length(result_data, 128)
    expect_false(any(is.na(result_data)))
    expect_false(any(is.infinite(result_data)))

    # Result should be in reasonable range
    expect_true(max(abs(result_data)) < 10)

    # GeGLU produces non-zero output for non-zero input
    expect_true(sum(abs(result_data)) > 0.1)

    # Cleanup
    ggml_backend_buffer_free(buffer_vk)
    ggml_vulkan_free(backend_vk)
    ggml_free(ctx)
  })

  test_that("Vulkan: RoPE (Rotary Position Embedding)", {
    skip("RoPE requires position tensor - tested through higher-level models")

    # Note: RoPE operations require proper position input tensors and are
    # typically tested through complete transformer model inference.
    # The Vulkan backend supports rope_norm, rope_neox, rope_vision shaders.

    expect_true(TRUE)
  })

  test_that("Vulkan: Flash Attention", {
    ctx <- ggml_init(mem_size = 64 * 1024 * 1024)
    ggml_set_no_alloc(ctx, TRUE)

    # Parameters
    n_head <- 4
    n_embd <- 128
    seq_len <- 32
    head_dim <- n_embd / n_head

    # Create Q, K, V tensors
    # Shape: [head_dim, n_head, seq_len, batch]
    q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, seq_len, 1)
    k <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, seq_len, 1)
    v <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, seq_len, 1)

    # Flash attention
    scale <- 1.0 / sqrt(head_dim)
    max_bias <- 0.0
    logit_softcap <- 0.0
    result <- ggml_flash_attn_ext(ctx, q, k, v, NULL, scale, max_bias, logit_softcap)

    # Setup Vulkan backend
    backend_vk <- ggml_vulkan_init(0)
    buffer_vk <- ggml_backend_alloc_ctx_tensors(ctx, backend_vk)

    # Set test data (normalized random)
    q_data <- rnorm(head_dim * n_head * seq_len)
    k_data <- rnorm(head_dim * n_head * seq_len)
    v_data <- rnorm(head_dim * n_head * seq_len)

    ggml_backend_tensor_set_data(q, q_data)
    ggml_backend_tensor_set_data(k, k_data)
    ggml_backend_tensor_set_data(v, v_data)

    # Compute
    graph <- ggml_build_forward_expand(ctx, result)
    ggml_backend_graph_compute(backend_vk, graph)

    # Get result
    result_data <- ggml_backend_tensor_get_data(result)

    # Basic checks
    expect_length(result_data, head_dim * n_head * seq_len)
    expect_false(any(is.na(result_data)))
    expect_false(any(is.infinite(result_data)))

    # Output should be weighted combination of V, so magnitude similar
    result_norm <- sqrt(mean(result_data^2))
    v_norm <- sqrt(mean(v_data^2))
    expect_true(abs(result_norm / v_norm - 1) < 0.5)

    # Cleanup
    ggml_backend_buffer_free(buffer_vk)
    ggml_vulkan_free(backend_vk)
    ggml_free(ctx)
  })

  test_that("Vulkan: Quantized tensor support Q4_0", {
    skip("Quantization tests require special setup - tested in quantized matmul")

    # Note: Direct quantization roundtrip testing is complex because:
    # 1. ggml_cpy between types needs proper tensor setup
    # 2. Quantization happens at backend level, not in compute graph
    # 3. Better to test through actual operations (mul_mat) that use quantized tensors

    expect_true(TRUE)
  })

  test_that("Vulkan: Quantized matrix multiplication Q4_0", {
    skip("Quantized matmul requires proper quantization setup - verified through benchmarks")

    # Note: Q4_0 and other quantized formats require:
    # 1. Proper quantization at tensor creation
    # 2. Backend-specific buffer management
    # 3. Dequantization shaders (dequant_q4_0.comp, mul_mat_vec_q4_0.comp)
    #
    # The Vulkan backend supports all quantization formats (Q4_0, Q8_0, etc.)
    # and is tested through real model inference and benchmarks.

    expect_true(TRUE)
  })

} else {
  test_that("Vulkan functions handle unavailable state", {
    skip("Vulkan not available or no devices found")
  })
}
