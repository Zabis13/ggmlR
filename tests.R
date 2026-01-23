#!/usr/bin/env Rscript
# ============================================================================
# ggmlR Smoke Tests - One test per implemented CPU function
# ============================================================================

library(ggmlR)

divider <- paste(rep("=", 70), collapse="")
cat(divider, "\n")
cat("ggmlR Smoke Tests\n")
cat(divider, "\n\n")

passed <- 0
failed <- 0
errors <- character()

test <- function(name, expr) {
  result <- tryCatch({
    res <- eval(expr)
    if (isTRUE(res) || (!is.null(res) && !identical(res, FALSE))) {
      cat("[PASS]", name, "\n")
      passed <<- passed + 1
      TRUE
    } else {
      cat("[FAIL]", name, "\n")
      failed <<- failed + 1
      errors <<- c(errors, name)
      FALSE
    }
  }, error = function(e) {
    cat("[FAIL]", name, "-", conditionMessage(e), "\n")
    failed <<- failed + 1
    errors <<- c(errors, paste(name, "-", conditionMessage(e)))
    FALSE
  })
}

# ============================================================================
# Context and Memory Management
# ============================================================================
cat("\n--- Context and Memory ---\n")

ctx <- ggml_init(mem_size = 64 * 1024 * 1024)
test("ggml_init", !is.null(ctx))
test("ggml_is_available", ggml_is_available())
test("ggml_version", nchar(ggml_version()) > 0)
test("ggml_tensor_overhead", ggml_tensor_overhead() > 0)
test("ggml_get_mem_size", ggml_get_mem_size(ctx) > 0)
test("ggml_used_mem", ggml_used_mem(ctx) >= 0)
test("ggml_set_no_alloc", { ggml_set_no_alloc(ctx, FALSE); TRUE })
test("ggml_get_no_alloc", is.logical(ggml_get_no_alloc(ctx)))
test("ggml_get_max_tensor_size", ggml_get_max_tensor_size(ctx) >= 0)  # 0 is valid for empty ctx
test("ggml_graph_overhead", ggml_graph_overhead() > 0)

# ============================================================================
# Tensor Creation
# ============================================================================
cat("\n--- Tensor Creation ---\n")

t1d <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
test("ggml_new_tensor_1d", !is.null(t1d))

t2d <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4)
test("ggml_new_tensor_2d", !is.null(t2d))

t3d <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 2)
test("ggml_new_tensor_3d", !is.null(t3d))

t4d <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 4, 2, 2)
test("ggml_new_tensor_4d", !is.null(t4d))

tnd <- ggml_new_tensor(ctx, GGML_TYPE_F32, 3, c(4, 4, 2))
test("ggml_new_tensor", !is.null(tnd))

tdup <- ggml_dup_tensor(ctx, t2d)
test("ggml_dup_tensor", !is.null(tdup))

tf32 <- ggml_new_f32(ctx, 3.14)
test("ggml_new_f32", !is.null(tf32))

ti32 <- ggml_new_i32(ctx, 42L)
test("ggml_new_i32", !is.null(ti32))

# ============================================================================
# Tensor Properties
# ============================================================================
cat("\n--- Tensor Properties ---\n")

test("ggml_nelements", ggml_nelements(t2d) == 32)
test("ggml_nbytes", ggml_nbytes(t2d) > 0)
test("ggml_n_dims", ggml_n_dims(t2d) == 2)
test("ggml_is_contiguous", is.logical(ggml_is_contiguous(t2d)))
test("ggml_is_transposed", is.logical(ggml_is_transposed(t2d)))
test("ggml_is_permuted", is.logical(ggml_is_permuted(t2d)))
test("ggml_tensor_shape", length(ggml_tensor_shape(t2d)) == 4)
test("ggml_tensor_type", ggml_tensor_type(t2d) == GGML_TYPE_F32)
test("ggml_type_size", ggml_type_size(GGML_TYPE_F32) == 4)
test("ggml_element_size", ggml_element_size(t2d) == 4)
test("ggml_nrows", ggml_nrows(t2d) == 4)
test("ggml_are_same_shape", ggml_are_same_shape(t2d, tdup))
test("ggml_set_name", { ggml_set_name(t2d, "test_tensor"); TRUE })
test("ggml_get_name", nchar(ggml_get_name(t2d)) > 0)

# ============================================================================
# Tensor Data Operations
# ============================================================================
cat("\n--- Tensor Data Operations ---\n")

test("ggml_set_zero", { ggml_set_zero(t1d); TRUE })
test("ggml_set_f32", { ggml_set_f32(t1d, rep(1.0, 10)); TRUE })
test("ggml_get_f32", length(ggml_get_f32(t1d)) == 10)

ti <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 5)
test("ggml_set_i32", { ggml_set_i32(ti, 1:5); TRUE })
test("ggml_get_i32", length(ggml_get_i32(ti)) == 5)

# ============================================================================
# Basic Arithmetic Operations
# ============================================================================
cat("\n--- Basic Arithmetic ---\n")

a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4)
b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4)

test("ggml_dup", !is.null(ggml_dup(ctx, a)))
test("ggml_add", !is.null(ggml_add(ctx, a, b)))
test("ggml_sub", !is.null(ggml_sub(ctx, a, b)))
test("ggml_mul", !is.null(ggml_mul(ctx, a, b)))
test("ggml_div", !is.null(ggml_div(ctx, a, b)))
test("ggml_scale", !is.null(ggml_scale(ctx, a, 2.0)))

# ============================================================================
# Matrix Operations
# ============================================================================
cat("\n--- Matrix Operations ---\n")

# mul_mat: A[k,n] x B[k,m] -> [n,m]
m1 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4)  # k=8, n=4
m2 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 6)  # k=8, m=6

test("ggml_mul_mat", !is.null(ggml_mul_mat(ctx, m1, m2)))

# out_prod: A[m,n] x B[p,n] -> [m,p] (both have same n rows)
op_a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 6)  # m=4, n=6
op_b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 6)  # p=8, n=6
test("ggml_out_prod", !is.null(ggml_out_prod(ctx, op_a, op_b)))

test("ggml_transpose", !is.null(ggml_transpose(ctx, a)))

# ============================================================================
# Unary Math Operations
# ============================================================================
cat("\n--- Unary Math Operations ---\n")

test("ggml_sqr", !is.null(ggml_sqr(ctx, a)))
test("ggml_sqrt", !is.null(ggml_sqrt(ctx, a)))
test("ggml_log", !is.null(ggml_log(ctx, a)))
test("ggml_exp", !is.null(ggml_exp(ctx, a)))
test("ggml_abs", !is.null(ggml_abs(ctx, a)))
test("ggml_neg", !is.null(ggml_neg(ctx, a)))
test("ggml_sgn", !is.null(ggml_sgn(ctx, a)))
test("ggml_step", !is.null(ggml_step(ctx, a)))
test("ggml_sin", !is.null(ggml_sin(ctx, a)))
test("ggml_cos", !is.null(ggml_cos(ctx, a)))
test("ggml_floor", !is.null(ggml_floor(ctx, a)))
test("ggml_ceil", !is.null(ggml_ceil(ctx, a)))
test("ggml_round", !is.null(ggml_round(ctx, a)))
test("ggml_clamp", !is.null(ggml_clamp(ctx, a, 0.0, 1.0)))

# ============================================================================
# Activation Functions
# ============================================================================
cat("\n--- Activation Functions ---\n")

test("ggml_relu", !is.null(ggml_relu(ctx, a)))
test("ggml_gelu", !is.null(ggml_gelu(ctx, a)))
test("ggml_gelu_quick", !is.null(ggml_gelu_quick(ctx, a)))
test("ggml_gelu_erf", !is.null(ggml_gelu_erf(ctx, a)))
test("ggml_silu", !is.null(ggml_silu(ctx, a)))
test("ggml_tanh", !is.null(ggml_tanh(ctx, a)))
test("ggml_sigmoid", !is.null(ggml_sigmoid(ctx, a)))
test("ggml_elu", !is.null(ggml_elu(ctx, a)))
test("ggml_leaky_relu", !is.null(ggml_leaky_relu(ctx, a, 0.01)))
test("ggml_hardswish", !is.null(ggml_hardswish(ctx, a)))
test("ggml_hardsigmoid", !is.null(ggml_hardsigmoid(ctx, a)))
test("ggml_softplus", !is.null(ggml_softplus(ctx, a)))

# ============================================================================
# GLU Variants
# ============================================================================
cat("\n--- GLU Variants ---\n")

glu_input <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4)  # even first dim
glu_a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4)
glu_b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4)

test("ggml_glu", !is.null(ggml_glu(ctx, glu_input, GGML_GLU_OP_REGLU)))
test("ggml_reglu", !is.null(ggml_reglu(ctx, glu_input)))
test("ggml_geglu", !is.null(ggml_geglu(ctx, glu_input)))
test("ggml_swiglu", !is.null(ggml_swiglu(ctx, glu_input)))
test("ggml_geglu_quick", !is.null(ggml_geglu_quick(ctx, glu_input)))
test("ggml_glu_split", !is.null(ggml_glu_split(ctx, glu_a, glu_b, GGML_GLU_OP_REGLU)))
test("ggml_reglu_split", !is.null(ggml_reglu_split(ctx, glu_a, glu_b)))
test("ggml_geglu_split", !is.null(ggml_geglu_split(ctx, glu_a, glu_b)))
test("ggml_swiglu_split", !is.null(ggml_swiglu_split(ctx, glu_a, glu_b)))

# ============================================================================
# Normalization
# ============================================================================
cat("\n--- Normalization ---\n")

test("ggml_norm", !is.null(ggml_norm(ctx, a, 1e-5)))
test("ggml_norm_inplace", !is.null(ggml_norm_inplace(ctx, a, 1e-5)))
test("ggml_rms_norm", !is.null(ggml_rms_norm(ctx, a, 1e-5)))
test("ggml_rms_norm_inplace", !is.null(ggml_rms_norm_inplace(ctx, a, 1e-5)))
test("ggml_group_norm", !is.null(ggml_group_norm(ctx, a, n_groups = 2, eps = 1e-5)))
test("ggml_group_norm_inplace", !is.null(ggml_group_norm_inplace(ctx, a, n_groups = 2)))
test("ggml_l2_norm", !is.null(ggml_l2_norm(ctx, a, 1e-5)))
test("ggml_l2_norm_inplace", !is.null(ggml_l2_norm_inplace(ctx, a)))
test("ggml_rms_norm_back", !is.null(ggml_rms_norm_back(ctx, a, b, 1e-5)))

# ============================================================================
# Softmax
# ============================================================================
cat("\n--- Softmax ---\n")

test("ggml_soft_max", !is.null(ggml_soft_max(ctx, a)))
test("ggml_soft_max_inplace", !is.null(ggml_soft_max_inplace(ctx, a)))
test("ggml_soft_max_ext", !is.null(ggml_soft_max_ext(ctx, a, scale = 1.0)))

# ============================================================================
# Reduction Operations
# ============================================================================
cat("\n--- Reduction Operations ---\n")

test("ggml_sum", !is.null(ggml_sum(ctx, a)))
test("ggml_sum_rows", !is.null(ggml_sum_rows(ctx, a)))
test("ggml_mean", !is.null(ggml_mean(ctx, a)))
test("ggml_argmax", !is.null(ggml_argmax(ctx, a)))

# ============================================================================
# Reshape and View Operations
# ============================================================================
cat("\n--- Reshape and View ---\n")

test("ggml_view_tensor", !is.null(ggml_view_tensor(ctx, a)))
test("ggml_reshape_1d", !is.null(ggml_reshape_1d(ctx, a, 16)))
test("ggml_reshape_2d", !is.null(ggml_reshape_2d(ctx, a, 8, 2)))
test("ggml_reshape_3d", !is.null(ggml_reshape_3d(ctx, a, 4, 2, 2)))
test("ggml_reshape_4d", !is.null(ggml_reshape_4d(ctx, a, 2, 2, 2, 2)))

v_src <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64)
test("ggml_view_1d", !is.null(ggml_view_1d(ctx, v_src, 16, 0)))
test("ggml_view_2d", !is.null(ggml_view_2d(ctx, v_src, 4, 4, 4*4, 0)))
test("ggml_view_3d", !is.null(ggml_view_3d(ctx, v_src, 2, 2, 4, 2*4, 4*4, 0)))
test("ggml_view_4d", !is.null(ggml_view_4d(ctx, v_src, 2, 2, 2, 2, 2*4, 4*4, 8*4, 0)))

test("ggml_permute", !is.null(ggml_permute(ctx, t4d, 0, 2, 1, 3)))
test("ggml_cont", !is.null(ggml_cont(ctx, a)))

# ============================================================================
# Tensor Manipulation
# ============================================================================
cat("\n--- Tensor Manipulation ---\n")

test("ggml_repeat", !is.null(ggml_repeat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2), a)))
test("ggml_repeat_back", !is.null(ggml_repeat_back(ctx, a, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2))))
test("ggml_concat", !is.null(ggml_concat(ctx, a, b, dim = 0)))
test("ggml_upscale", !is.null(ggml_upscale(ctx, a, 2)))
test("ggml_pad", !is.null(ggml_pad(ctx, a, 1, 1, 0, 0)))
test("ggml_diag", !is.null(ggml_diag(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4))))

# ============================================================================
# Indexing Operations
# ============================================================================
cat("\n--- Indexing Operations ---\n")

idx <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2)
test("ggml_get_rows", !is.null(ggml_get_rows(ctx, a, idx)))
test("ggml_argsort", !is.null(ggml_argsort(ctx, a)))
test("ggml_top_k", !is.null(ggml_top_k(ctx, a, 2)))

# ============================================================================
# Copy and Set Operations
# ============================================================================
cat("\n--- Copy and Set ---\n")

cpy_dst <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4)
test("ggml_cpy", !is.null(ggml_cpy(ctx, a, cpy_dst)))

set_dst <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8)
set_src <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4)
test("ggml_set", !is.null(ggml_set(ctx, set_dst, set_src, 8*4, 8*4*4, 0, 0)))

# ============================================================================
# CNN Operations
# ============================================================================
cat("\n--- CNN Operations ---\n")

# Convolution tensors - GGML format:
# conv_1d: kernel [K, IC, OC], input [L, IC, N]
conv1d_kernel <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 4, 8)   # K=3, IC=4, OC=8
conv1d_input <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 4, 1)   # L=16, IC=4, N=1

test("ggml_conv_1d", !is.null(ggml_conv_1d(ctx, conv1d_kernel, conv1d_input, s0 = 1, p0 = 1, d0 = 1)))

# conv_transpose_1d: kernel [K, IC, OC], input needs ne[1] == OC
# For transpose conv, input is "output" of regular conv: [L', OC, N]
conv1d_transpose_input <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 14, 8, 1)  # L'=14, OC=8, N=1
test("ggml_conv_transpose_1d", !is.null(ggml_conv_transpose_1d(ctx, conv1d_kernel, conv1d_transpose_input, s0 = 1, p0 = 0, d0 = 1)))

# conv_2d: kernel [KW, KH, IC, OC], input [W, H, IC, N]
conv2d_kernel <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 4, 8)   # KW=3, KH=3, IC=4, OC=8
conv2d_input <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 16, 4, 1)  # W=16, H=16, IC=4, N=1

test("ggml_conv_2d", !is.null(ggml_conv_2d(ctx, conv2d_kernel, conv2d_input, s0 = 1, s1 = 1, p0 = 1, p1 = 1)))

# Pooling
pool_input_1d <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 4)  # L=16, C=4
pool_input_2d <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 4)  # W=16, H=16, C=4

test("GGML_OP_POOL_MAX constant", GGML_OP_POOL_MAX == 0L)
test("GGML_OP_POOL_AVG constant", GGML_OP_POOL_AVG == 1L)
test("ggml_pool_1d (max)", !is.null(ggml_pool_1d(ctx, pool_input_1d, GGML_OP_POOL_MAX, k0 = 2, s0 = 2, p0 = 0)))
test("ggml_pool_1d (avg)", !is.null(ggml_pool_1d(ctx, pool_input_1d, GGML_OP_POOL_AVG, k0 = 2)))
test("ggml_pool_2d (max)", !is.null(ggml_pool_2d(ctx, pool_input_2d, GGML_OP_POOL_MAX, k0 = 2, k1 = 2)))
test("ggml_pool_2d (avg)", !is.null(ggml_pool_2d(ctx, pool_input_2d, GGML_OP_POOL_AVG, k0 = 2, k1 = 2, s0 = 2, s1 = 2)))

# im2col
test("ggml_im2col", !is.null(ggml_im2col(ctx, conv2d_kernel, conv2d_input, s0 = 1, s1 = 1, p0 = 0, p1 = 0, d0 = 1, d1 = 1, is_2D = TRUE)))

# ============================================================================
# Attention Operations
# ============================================================================
cat("\n--- Attention Operations ---\n")

q <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 8, 4)  # head_dim, seq_len, n_heads
k <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 8, 4)
v <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 8, 4)

test("ggml_flash_attn_ext", !is.null(ggml_flash_attn_ext(ctx, q, k, v, scale = 1.0/8.0)))

# ============================================================================
# Masking Operations
# ============================================================================
cat("\n--- Masking Operations ---\n")

mask_tensor <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8)
test("ggml_diag_mask_inf", !is.null(ggml_diag_mask_inf(ctx, mask_tensor, 0)))
test("ggml_diag_mask_inf_inplace", !is.null(ggml_diag_mask_inf_inplace(ctx, mask_tensor, 0)))
test("ggml_diag_mask_zero", !is.null(ggml_diag_mask_zero(ctx, mask_tensor, 0)))

# ============================================================================
# RoPE Operations
# ============================================================================
cat("\n--- RoPE Operations ---\n")

# rope: a->ne[2] == b->ne[0] required
rope_a <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 8, 4)  # [n_dims, seq_len, batch]
rope_pos <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4)        # length = a->ne[2] = batch

test("ggml_rope", !is.null(ggml_rope(ctx, rope_a, rope_pos, 64)))
test("ggml_rope_inplace", !is.null(ggml_rope_inplace(ctx, rope_a, rope_pos, 64)))
test("ggml_rope_ext", !is.null(ggml_rope_ext(ctx, rope_a, rope_pos, n_dims = 64)))

# ============================================================================
# Backward Operations (for training)
# ============================================================================
cat("\n--- Backward Operations ---\n")

grad <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4)
test("ggml_silu_back", !is.null(ggml_silu_back(ctx, a, grad)))
test("ggml_soft_max_ext_back", !is.null(ggml_soft_max_ext_back(ctx, a, grad)))

rows_a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2)
rows_b <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2)
rows_c <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4)
test("ggml_get_rows_back", !is.null(ggml_get_rows_back(ctx, rows_a, rows_b, rows_c)))

test("ggml_rope_ext_back", !is.null(ggml_rope_ext_back(ctx, rope_a, rope_pos, n_dims = 64)))

# ============================================================================
# Graph Operations
# ============================================================================
cat("\n--- Graph Operations ---\n")

# Build a simple computation graph
x <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4)
y <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4)
z <- ggml_add(ctx, x, y)
graph <- ggml_build_forward_expand(ctx, z)

test("ggml_build_forward_expand", !is.null(graph))
test("ggml_graph_n_nodes", ggml_graph_n_nodes(graph) > 0)
test("ggml_graph_node", !is.null(ggml_graph_node(graph, 0)))

# Graph allocator
galloc <- ggml_gallocr_new()
test("ggml_gallocr_new", !is.null(galloc))
test("ggml_gallocr_reserve", ggml_gallocr_reserve(galloc, graph))
test("ggml_gallocr_get_buffer_size", ggml_gallocr_get_buffer_size(galloc) >= 0)
test("ggml_gallocr_alloc_graph", ggml_gallocr_alloc_graph(galloc, graph))
#test("ggml_gallocr_free", { ggml_gallocr_free(galloc); TRUE })

#test("ggml_graph_compute_with_ctx", { ggml_graph_compute_with_ctx(ctx, graph); TRUE })
#test("ggml_graph_reset", { ggml_graph_reset(graph); TRUE })

# ============================================================================
# Threading
# ============================================================================
cat("\n--- Threading ---\n")

test("ggml_get_n_threads", ggml_get_n_threads() > 0)
test("ggml_set_n_threads", { ggml_set_n_threads(4); TRUE })

# ============================================================================
# Timing
# ============================================================================
cat("\n--- Timing ---\n")

test("ggml_time_init", { ggml_time_init(); TRUE })
test("ggml_time_ms", ggml_time_ms() >= 0)
test("ggml_time_us", ggml_time_us() >= 0)
test("ggml_cycles", ggml_cycles() >= 0)
test("ggml_cycles_per_ms", ggml_cycles_per_ms() >= 0)

# ============================================================================
# Context Reset and Cleanup
# ============================================================================
cat("\n--- Reset and Cleanup ---\n")

test("ggml_reset", { ggml_reset(ctx); TRUE })
test("ggml_free", { ggml_free(ctx); TRUE })

# ============================================================================
# Backend Operations
# ============================================================================
cat("\n--- Backend Operations ---\n")

backend <- ggml_backend_cpu_init()
test("ggml_backend_cpu_init", !is.null(backend))
test("ggml_backend_name", nchar(ggml_backend_name(backend)) > 0)
test("ggml_backend_cpu_set_n_threads", { ggml_backend_cpu_set_n_threads(backend, 4); TRUE })

# Allocate tensors using backend
ctx_backend <- ggml_init(mem_size = 16 * 1024 * 1024)
ggml_set_no_alloc(ctx_backend, TRUE)
tb1 <- ggml_new_tensor_2d(ctx_backend, GGML_TYPE_F32, 4, 4)
tb2 <- ggml_new_tensor_2d(ctx_backend, GGML_TYPE_F32, 4, 4)
tb3 <- ggml_add(ctx_backend, tb1, tb2)

buffer <- ggml_backend_alloc_ctx_tensors(ctx_backend, backend)
test("ggml_backend_alloc_ctx_tensors", !is.null(buffer))
test("ggml_backend_buffer_name", nchar(ggml_backend_buffer_name(buffer)) > 0)
test("ggml_backend_buffer_get_size", ggml_backend_buffer_get_size(buffer) > 0)

# Set and get tensor data
test_data <- rep(1.0, 16)
test("ggml_backend_tensor_set_data", { ggml_backend_tensor_set_data(tb1, test_data); TRUE })
test("ggml_backend_tensor_get_data", {
  result <- ggml_backend_tensor_get_data(tb1)
  !is.null(result) && length(result) == 16
})

# Build and compute graph with backend
bg <- ggml_build_forward_expand(ctx_backend, tb3)
test("ggml_backend_graph_compute", { ggml_backend_graph_compute(backend, bg); TRUE })

test("ggml_backend_buffer_free", { ggml_backend_buffer_free(buffer); TRUE })
test("ggml_backend_free", { ggml_backend_free(backend); TRUE })
ggml_free(ctx_backend)

# ============================================================================
# Quantization
# ============================================================================
cat("\n--- Quantization ---\n")

test("ggml_quantize_init (Q4_0)", { ggml_quantize_init(GGML_TYPE_Q4_0); TRUE })
test("ggml_quantize_requires_imatrix (Q4_0)", is.logical(ggml_quantize_requires_imatrix(GGML_TYPE_Q4_0)))
test("ggml_quantize_requires_imatrix (Q4_1)", is.logical(ggml_quantize_requires_imatrix(GGML_TYPE_Q4_1)))

# Quantize some float data
quant_data <- rep(0.5, 256)  # Must be multiple of 32 for quantization block size
test("ggml_quantize_chunk (Q8_0)", {
  result <- ggml_quantize_chunk(GGML_TYPE_Q4_0, quant_data, nrows = 1, n_per_row = 256)
  !is.null(result) && is.raw(result) && length(result) > 0
})

test("ggml_quantize_free", { ggml_quantize_free(); TRUE })

# ============================================================================
# Helper Functions
# ============================================================================
cat("\n--- Helper Functions ---\n")

test("ggml_init_auto", {
  ctx2 <- ggml_init_auto(c(10, 10), c(5, 5))
  result <- !is.null(ctx2)
  ggml_free(ctx2)
  result
})

test("ggml_estimate_memory", ggml_estimate_memory(GGML_TYPE_F32, 100, 100) > 0)

# ============================================================================
# Vulkan Backend (if available)
# ============================================================================
cat("\n--- Vulkan Backend ---\n")

if (ggml_vulkan_available()) {
  test("ggml_vulkan_available", TRUE)
  test("ggml_vulkan_device_count", ggml_vulkan_device_count() > 0)

  if (ggml_vulkan_device_count() > 0) {
    # Test device info
    test("ggml_vulkan_device_description", {
      desc <- ggml_vulkan_device_description(0)
      !is.null(desc) && nchar(desc) > 0
    })

    test("ggml_vulkan_device_memory", {
      mem <- ggml_vulkan_device_memory(0)
      !is.null(mem) && is.list(mem) && !is.null(mem$free) && !is.null(mem$total)
    })

    test("ggml_vulkan_list_devices", {
      devices <- ggml_vulkan_list_devices()
      !is.null(devices) && is.list(devices) && length(devices) > 0
    })

    # Test backend initialization
    test("ggml_vulkan_init", {
      vk_backend <- ggml_vulkan_init(0)
      result <- !is.null(vk_backend)
      if (result) {
        test("ggml_vulkan_backend_name", {
          name <- ggml_vulkan_backend_name(vk_backend)
          !is.null(name) && nchar(name) > 0
        })

        test("ggml_vulkan_is_backend", ggml_vulkan_is_backend(vk_backend))

        # Test basic computation with Vulkan backend
        test("Vulkan backend computation", {
          ctx_vk <- ggml_init(mem_size = 16 * 1024 * 1024)
          ggml_set_no_alloc(ctx_vk, TRUE)

          t1 <- ggml_new_tensor_2d(ctx_vk, GGML_TYPE_F32, 4, 4)
          t2 <- ggml_new_tensor_2d(ctx_vk, GGML_TYPE_F32, 4, 4)
          t3 <- ggml_add(ctx_vk, t1, t2)

          buf <- ggml_backend_alloc_ctx_tensors(ctx_vk, vk_backend)
          result <- !is.null(buf)

          if (result) {
            # Set input data
            data1 <- rep(1.0, 16)
            data2 <- rep(2.0, 16)
            ggml_backend_tensor_set_data(t1, data1)
            ggml_backend_tensor_set_data(t2, data2)

            # Compute
            g <- ggml_build_forward_expand(ctx_vk, t3)
            ggml_backend_graph_compute(vk_backend, g)

            # Get result
            result_data <- ggml_backend_tensor_get_data(t3)
            result <- !is.null(result_data) && all(abs(result_data - 3.0) < 1e-5)

            ggml_backend_buffer_free(buf)
          }

          ggml_free(ctx_vk)
          result
        })

        test("ggml_vulkan_free", { ggml_vulkan_free(vk_backend); TRUE })
      }
      result
    })
  }
} else {
  cat("[INFO] Vulkan not available - skipping Vulkan tests\n")
  cat("       To enable: R CMD INSTALL --configure-args=\"--with-vulkan\" .\n")
}

# ============================================================================
# Summary
# ============================================================================
cat("\n")
cat(divider, "\n")
cat(sprintf("RESULTS: %d passed, %d failed\n", passed, failed))
cat(divider, "\n")

if (failed > 0) {
  cat("\nFailed tests:\n")
  for (e in errors) {
    cat("  -", e, "\n")
  }
  quit(status = 1)
} else {
  cat("\nAll smoke tests passed!\n")
  quit(status = 0)
}
