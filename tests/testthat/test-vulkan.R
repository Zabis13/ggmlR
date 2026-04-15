library(ggmlR)

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

  # -----------------------------------------------------------------------
  # Subgroup-shuffle mmq pipeline: Q4_K / Q5_K / Q6_K
  # Tests that the new USE_SUBGROUP_NO_SHMEM path (selected automatically
  # on wavefront-64 devices) computes without crash and returns finite output.
  # No CPU cross-check — just smoke: correct shape, no NaN/Inf.
  # -----------------------------------------------------------------------

  for (qspec in list(c(GGML_TYPE_Q4_K, "Q4_K"),
                     c(GGML_TYPE_Q5_K, "Q5_K"),
                     c(GGML_TYPE_Q6_K, "Q6_K"))) {
    local({
      qt    <- as.integer(qspec[1])
      qname <- qspec[2]

      test_that(paste("Vulkan: quantized matmul", qname, "(mmq shuffle path)"), {
        skip_if_not(ggml_vulkan_available(),        "Vulkan not available")
        skip_if_not(ggml_vulkan_device_count() > 0, "No Vulkan devices")

        caps <- ggml_vulkan_device_caps(0L)
        skip_if_not(caps$integer_dot_product, "integer_dot_product not supported")

        # Dimensions: K must be multiple of block size (256 for k-quants)
        M <- 32L; N <- 32L; K <- 256L

        ctx <- ggml_init(mem_size = 32L * 1024L * 1024L)
        ggml_set_no_alloc(ctx, TRUE)

        w   <- ggml_new_tensor_2d(ctx, qt, K, M)
        x   <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N)
        out <- ggml_mul_mat(ctx, w, x)

        backend_vk <- ggml_vulkan_init(0L)
        buffer_vk  <- ggml_backend_alloc_ctx_tensors(ctx, backend_vk)

        ggml_backend_tensor_set_data(x, rnorm(K * N))
        # Quantize F32 → raw bytes, pass directly to tensor
        w_raw <- switch(qname,
          Q4_K = quantize_row_q4_K_ref(rnorm(K * M), K * M),
          Q5_K = quantize_row_q5_K_ref(rnorm(K * M), K * M),
          Q6_K = quantize_row_q6_K_ref(rnorm(K * M), K * M)
        )
        ggml_backend_tensor_set_data(w, w_raw)

        graph <- ggml_build_forward_expand(ctx, out)
        expect_no_error(ggml_backend_graph_compute(backend_vk, graph))

        result <- ggml_backend_tensor_get_data(out)

        expect_length(result, M * N)
        expect_false(any(is.nan(result)),      label = paste(qname, "no NaN"))
        expect_false(any(is.infinite(result)), label = paste(qname, "no Inf"))

        ggml_backend_buffer_free(buffer_vk)
        ggml_vulkan_free(backend_vk)
        ggml_free(ctx)
      })
    })
  }

} else {
  test_that("Vulkan functions handle unavailable state", {
    skip("Vulkan not available or no devices found")
  })
}
