pkgname <- "ggmlR"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('ggmlR')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("ggml_add")
### * ggml_add

flush(stderr()); flush(stdout())

### Name: ggml_add
### Title: Add tensors
### Aliases: ggml_add

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
##D b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
##D ggml_set_f32(a, c(1, 2, 3, 4, 5))
##D ggml_set_f32(b, c(5, 4, 3, 2, 1))
##D c <- ggml_add(ctx, a, b)
##D graph <- ggml_build_forward_expand(ctx, c)
##D ggml_graph_compute(ctx, graph)
##D result <- ggml_get_f32(c)
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_build_forward_expand")
### * ggml_build_forward_expand

flush(stderr()); flush(stdout())

### Name: ggml_build_forward_expand
### Title: Build forward expand
### Aliases: ggml_build_forward_expand

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
##D b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
##D ggml_set_f32(a, 1:10)
##D ggml_set_f32(b, 11:20)
##D c <- ggml_add(ctx, a, b)
##D graph <- ggml_build_forward_expand(ctx, c)
##D ggml_graph_compute(ctx, graph)
##D result <- ggml_get_f32(c)
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_cpu_add")
### * ggml_cpu_add

flush(stderr()); flush(stdout())

### Name: ggml_cpu_add
### Title: Element-wise Addition (CPU Direct)
### Aliases: ggml_cpu_add

### ** Examples

## Not run: 
##D ctx <- ggml_init(1024 * 1024)
##D a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
##D b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
##D ggml_set_f32(a, c(1, 2, 3, 4, 5))
##D ggml_set_f32(b, c(5, 4, 3, 2, 1))
##D result <- ggml_cpu_add(a, b)
##D print(result)  # [6, 6, 6, 6, 6]
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_cpu_mul")
### * ggml_cpu_mul

flush(stderr()); flush(stdout())

### Name: ggml_cpu_mul
### Title: Element-wise Multiplication (CPU Direct)
### Aliases: ggml_cpu_mul

### ** Examples

## Not run: 
##D ctx <- ggml_init(1024 * 1024)
##D a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
##D b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
##D ggml_set_f32(a, c(1, 2, 3, 4, 5))
##D ggml_set_f32(b, c(2, 2, 2, 2, 2))
##D result <- ggml_cpu_mul(a, b)
##D print(result)  # [2, 4, 6, 8, 10]
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_diag_mask_inf")
### * ggml_diag_mask_inf

flush(stderr()); flush(stdout())

### Name: ggml_diag_mask_inf
### Title: Diagonal Mask with -Inf (Graph)
### Aliases: ggml_diag_mask_inf

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D 
##D # Create attention scores matrix
##D scores <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4)
##D ggml_set_f32(scores, rep(1, 16))
##D 
##D # Apply causal mask
##D masked <- ggml_diag_mask_inf(ctx, scores, 0)
##D # After computation, upper triangle will be -Inf
##D 
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_dup_tensor")
### * ggml_dup_tensor

flush(stderr()); flush(stdout())

### Name: ggml_dup_tensor
### Title: Duplicate Tensor
### Aliases: ggml_dup_tensor

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
##D b <- ggml_dup_tensor(ctx, a)  # Same shape as a
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_estimate_memory")
### * ggml_estimate_memory

flush(stderr()); flush(stdout())

### Name: ggml_estimate_memory
### Title: Estimate Required Memory
### Aliases: ggml_estimate_memory

### ** Examples

# For 1000x1000 F32 matrix
ggml_estimate_memory(GGML_TYPE_F32, 1000, 1000)



cleanEx()
nameEx("ggml_flash_attn_ext")
### * ggml_flash_attn_ext

flush(stderr()); flush(stdout())

### Name: ggml_flash_attn_ext
### Title: Flash Attention (Graph)
### Aliases: ggml_flash_attn_ext

### ** Examples

## Not run: 
##D ctx <- ggml_init(64 * 1024 * 1024)
##D 
##D head_dim <- 64
##D n_head <- 8
##D n_head_kv <- 2  # GQA with 4:1 ratio
##D seq_len <- 32
##D 
##D q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, seq_len, 1)
##D k <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head_kv, seq_len, 1)
##D v <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head_kv, seq_len, 1)
##D 
##D ggml_set_f32(q, rnorm(head_dim * n_head * seq_len))
##D ggml_set_f32(k, rnorm(head_dim * n_head_kv * seq_len))
##D ggml_set_f32(v, rnorm(head_dim * n_head_kv * seq_len))
##D 
##D # Scale = 1/sqrt(head_dim)
##D scale <- 1.0 / sqrt(head_dim)
##D 
##D # Compute attention
##D out <- ggml_flash_attn_ext(ctx, q, k, v, NULL, scale, 0.0, 0.0)
##D 
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_geglu")
### * ggml_geglu

flush(stderr()); flush(stdout())

### Name: ggml_geglu
### Title: GeGLU (GELU Gated Linear Unit) (Graph)
### Aliases: ggml_geglu

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 3)
##D ggml_set_f32(a, rnorm(24))
##D r <- ggml_geglu(ctx, a)
##D graph <- ggml_build_forward_expand(ctx, r)
##D ggml_graph_compute(ctx, graph)
##D result <- ggml_get_f32(r)  # Shape: 4x3
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_get_i32")
### * ggml_get_i32

flush(stderr()); flush(stdout())

### Name: ggml_get_i32
### Title: Get I32 Data
### Aliases: ggml_get_i32

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D # Create matrix and find argmax
##D m <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3)
##D ggml_set_f32(m, c(1,5,2,3, 4,1,8,2, 2,3,1,9))
##D idx <- ggml_argmax(ctx, m)
##D graph <- ggml_build_forward_expand(ctx, idx)
##D ggml_graph_compute(ctx, graph)
##D indices <- ggml_get_i32(idx)  # Returns c(1, 2, 3) - indices of max values
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_get_mem_size")
### * ggml_get_mem_size

flush(stderr()); flush(stdout())

### Name: ggml_get_mem_size
### Title: Get Context Memory Size
### Aliases: ggml_get_mem_size

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D ggml_get_mem_size(ctx)
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_get_n_threads")
### * ggml_get_n_threads

flush(stderr()); flush(stdout())

### Name: ggml_get_n_threads
### Title: Get Number of Threads
### Aliases: ggml_get_n_threads

### ** Examples

ggml_get_n_threads()



cleanEx()
nameEx("ggml_get_rows")
### * ggml_get_rows

flush(stderr()); flush(stdout())

### Name: ggml_get_rows
### Title: Get Rows by Indices (Graph)
### Aliases: ggml_get_rows

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D 
##D # Create embedding matrix: 10 tokens, 4-dim embeddings
##D embeddings <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 10)
##D ggml_set_f32(embeddings, rnorm(40))
##D 
##D # Token indices to look up
##D indices <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3)
##D ggml_set_i32(indices, c(0, 5, 2))
##D 
##D # Get embeddings for tokens 0, 5, 2
##D result <- ggml_get_rows(ctx, embeddings, indices)
##D # result shape: [4, 3]
##D 
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_glu")
### * ggml_glu

flush(stderr()); flush(stdout())

### Name: ggml_glu
### Title: Generic GLU (Gated Linear Unit) (Graph)
### Aliases: ggml_glu

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D # Create tensor with 10 columns (will be split into 5 + 5)
##D a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 4)
##D ggml_set_f32(a, rnorm(40))
##D # Apply SwiGLU
##D r <- ggml_glu(ctx, a, GGML_GLU_OP_SWIGLU, FALSE)
##D graph <- ggml_build_forward_expand(ctx, r)
##D ggml_graph_compute(ctx, graph)
##D result <- ggml_get_f32(r)  # Shape: 5x4
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_graph_compute")
### * ggml_graph_compute

flush(stderr()); flush(stdout())

### Name: ggml_graph_compute
### Title: Compute graph
### Aliases: ggml_graph_compute

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
##D b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
##D ggml_set_f32(a, 1:10)
##D ggml_set_f32(b, 11:20)
##D c <- ggml_add(ctx, a, b)
##D graph <- ggml_build_forward_expand(ctx, c)
##D ggml_graph_compute(ctx, graph)
##D result <- ggml_get_f32(c)
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_graph_node")
### * ggml_graph_node

flush(stderr()); flush(stdout())

### Name: ggml_graph_node
### Title: Get Graph Node
### Aliases: ggml_graph_node

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
##D b <- ggml_add(ctx, a, a)
##D graph <- ggml_build_forward_expand(ctx, b)
##D # Get the last node (output)
##D output <- ggml_graph_node(graph, -1)
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_init_auto")
### * ggml_init_auto

flush(stderr()); flush(stdout())

### Name: ggml_init_auto
### Title: Create Context with Auto-sizing
### Aliases: ggml_init_auto

### ** Examples

## Not run: 
##D # For two 1000x1000 matrices
##D ctx <- ggml_init_auto(mat1 = c(1000, 1000), mat2 = c(1000, 1000))
## End(Not run)



cleanEx()
nameEx("ggml_mul_mat")
### * ggml_mul_mat

flush(stderr()); flush(stdout())

### Name: ggml_mul_mat
### Title: Matrix Multiplication (Graph)
### Aliases: ggml_mul_mat

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D # Create 10x20 and 20x5 matrices
##D A <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 20)
##D B <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 20, 5)
##D ggml_set_f32(A, rnorm(200))
##D ggml_set_f32(B, rnorm(100))
##D # Result will be 10x5
##D C <- ggml_mul_mat(ctx, A, B)
##D graph <- ggml_build_forward_expand(ctx, C)
##D ggml_graph_compute(ctx, graph)
##D result <- ggml_get_f32(C)
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_new_tensor_3d")
### * ggml_new_tensor_3d

flush(stderr()); flush(stdout())

### Name: ggml_new_tensor_3d
### Title: Create 3D Tensor
### Aliases: ggml_new_tensor_3d

### ** Examples

## Not run: 
##D ctx <- ggml_init(64 * 1024 * 1024)
##D # Create 10x20x30 tensor
##D t <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 10, 20, 30)
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_new_tensor_4d")
### * ggml_new_tensor_4d

flush(stderr()); flush(stdout())

### Name: ggml_new_tensor_4d
### Title: Create 4D Tensor
### Aliases: ggml_new_tensor_4d

### ** Examples

## Not run: 
##D ctx <- ggml_init(128 * 1024 * 1024)
##D # Create batch of images: 32 images, 3 channels, 224x224
##D t <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 224, 224, 3, 32)
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_permute")
### * ggml_permute

flush(stderr()); flush(stdout())

### Name: ggml_permute
### Title: Permute Tensor Dimensions (Graph)
### Aliases: ggml_permute

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D # Create 4D tensor: (2, 3, 4, 5)
##D t <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 3, 4, 5)
##D # Swap axes 0 and 1: result shape (3, 2, 4, 5)
##D t_perm <- ggml_permute(ctx, t, 1, 0, 2, 3)
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_print_mem_status")
### * ggml_print_mem_status

flush(stderr()); flush(stdout())

### Name: ggml_print_mem_status
### Title: Print Context Memory Status
### Aliases: ggml_print_mem_status

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D ggml_print_mem_status(ctx)
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_reglu")
### * ggml_reglu

flush(stderr()); flush(stdout())

### Name: ggml_reglu
### Title: ReGLU (ReLU Gated Linear Unit) (Graph)
### Aliases: ggml_reglu

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 3)
##D ggml_set_f32(a, rnorm(24))
##D r <- ggml_reglu(ctx, a)
##D graph <- ggml_build_forward_expand(ctx, r)
##D ggml_graph_compute(ctx, graph)
##D result <- ggml_get_f32(r)  # Shape: 4x3
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_reset")
### * ggml_reset

flush(stderr()); flush(stdout())

### Name: ggml_reset
### Title: Reset GGML Context
### Aliases: ggml_reset

### ** Examples

## Not run: 
##D ctx <- ggml_init(100 * 1024 * 1024)
##D 
##D # Use context
##D a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000000)
##D 
##D # Reset to reuse memory
##D ggml_reset(ctx)
##D 
##D # Create new tensors in same context
##D b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2000000)
##D 
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_rope")
### * ggml_rope

flush(stderr()); flush(stdout())

### Name: ggml_rope
### Title: Rotary Position Embedding (Graph)
### Aliases: ggml_rope

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D 
##D # Query tensor: head_dim=8, n_head=4, seq_len=16, batch=1
##D q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 4, 16, 1)
##D ggml_set_f32(q, rnorm(8 * 4 * 16))
##D 
##D # Position indices
##D pos <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 16)
##D ggml_set_i32(pos, 0:15)
##D 
##D # Apply RoPE
##D q_rope <- ggml_rope(ctx, q, pos, 8, GGML_ROPE_TYPE_NORM)
##D 
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_rope_ext")
### * ggml_rope_ext

flush(stderr()); flush(stdout())

### Name: ggml_rope_ext
### Title: Extended RoPE with Frequency Scaling (Graph)
### Aliases: ggml_rope_ext

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D 
##D q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 8, 32, 1)
##D pos <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 32)
##D ggml_set_i32(pos, 0:31)
##D 
##D # Standard RoPE with default freq_base
##D q_rope <- ggml_rope_ext(ctx, q, pos, NULL,
##D                         n_dims = 64, mode = 0L,
##D                         n_ctx_orig = 4096,
##D                         freq_base = 10000, freq_scale = 1.0,
##D                         ext_factor = 0.0, attn_factor = 1.0,
##D                         beta_fast = 32, beta_slow = 1)
##D 
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_set_i32")
### * ggml_set_i32

flush(stderr()); flush(stdout())

### Name: ggml_set_i32
### Title: Set I32 Data
### Aliases: ggml_set_i32

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D # Create position indices for RoPE
##D pos <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 16)
##D ggml_set_i32(pos, 0:15)
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_set_n_threads")
### * ggml_set_n_threads

flush(stderr()); flush(stdout())

### Name: ggml_set_n_threads
### Title: Set Number of Threads
### Aliases: ggml_set_n_threads

### ** Examples

## Not run: 
##D # Use 4 threads
##D ggml_set_n_threads(4)
##D 
##D # Use all available cores
##D ggml_set_n_threads(parallel::detectCores())
## End(Not run)



cleanEx()
nameEx("ggml_set_no_alloc")
### * ggml_set_no_alloc

flush(stderr()); flush(stdout())

### Name: ggml_set_no_alloc
### Title: Set No Allocation Mode
### Aliases: ggml_set_no_alloc

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D ggml_set_no_alloc(ctx, TRUE)
##D # Create graph without allocating tensor data
##D ggml_set_no_alloc(ctx, FALSE)
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_swiglu")
### * ggml_swiglu

flush(stderr()); flush(stdout())

### Name: ggml_swiglu
### Title: SwiGLU (Swish/SiLU Gated Linear Unit) (Graph)
### Aliases: ggml_swiglu

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 3)
##D ggml_set_f32(a, rnorm(24))
##D r <- ggml_swiglu(ctx, a)
##D graph <- ggml_build_forward_expand(ctx, r)
##D ggml_graph_compute(ctx, graph)
##D result <- ggml_get_f32(r)  # Shape: 4x3
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_tensor_overhead")
### * ggml_tensor_overhead

flush(stderr()); flush(stdout())

### Name: ggml_tensor_overhead
### Title: Get Tensor Overhead
### Aliases: ggml_tensor_overhead

### ** Examples

ggml_tensor_overhead()



cleanEx()
nameEx("ggml_tensor_shape")
### * ggml_tensor_shape

flush(stderr()); flush(stdout())

### Name: ggml_tensor_shape
### Title: Get Tensor Shape
### Aliases: ggml_tensor_shape

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D t <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 20)
##D shape <- ggml_tensor_shape(t)  # c(10, 20, 1, 1)
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_used_mem")
### * ggml_used_mem

flush(stderr()); flush(stdout())

### Name: ggml_used_mem
### Title: Get Used Memory
### Aliases: ggml_used_mem

### ** Examples

## Not run: 
##D ctx <- ggml_init(16 * 1024 * 1024)
##D a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000)
##D ggml_used_mem(ctx)
##D ggml_free(ctx)
## End(Not run)



cleanEx()
nameEx("ggml_with_temp_ctx")
### * ggml_with_temp_ctx

flush(stderr()); flush(stdout())

### Name: ggml_with_temp_ctx
### Title: Execute with Temporary Context
### Aliases: ggml_with_temp_ctx

### ** Examples

## Not run: 
##D # Create large matrix in temporary context
##D result <- ggml_with_temp_ctx(100 * 1024 * 1024, {
##D   a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3000, 3000)
##D   b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3000, 3000)
##D   ggml_set_f32(a, rnorm(9000000))
##D   ggml_set_f32(b, rnorm(9000000))
##D   ggml_cpu_add(a, b)
##D })
## End(Not run)



### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
