# Minimal GPU compute crash locator.
# Marks each step so the last printed line == crash point (C-level death,
# since tryCatch won't catch a segfault/abort).
library(ggmlR)
flush_cat <- function(...) { cat(...); flush(stdout()) }

flush_cat("M0: start, vulkan_available=", ggml_vulkan_available(), "\n")
flush_cat("M1: device_count=", ggml_vulkan_device_count(), "\n")

size <- 1024L
ctx <- ggml_init(mem_size = as.numeric(size) * 4 * 4)
ggml_set_no_alloc(ctx, TRUE)
flush_cat("M2: ctx + no_alloc OK\n")

t1 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size)
t2 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size)
t3 <- ggml_add(ctx, t1, t2)
flush_cat("M3: tensors + add node built\n")

flush_cat("M4: BEFORE ggml_vulkan_init\n")
backend <- ggml_vulkan_init(0L)
flush_cat("M5: AFTER ggml_vulkan_init OK\n")

buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)
flush_cat("M6: AFTER alloc_ctx_tensors OK\n")

ggml_backend_tensor_set_data(t1, rnorm(size))
ggml_backend_tensor_set_data(t2, rnorm(size))
flush_cat("M7: AFTER tensor_set_data OK\n")

graph <- ggml_build_forward_expand(ctx, t3)
flush_cat("M8: AFTER build_forward_expand OK\n")

flush_cat("M9: BEFORE graph_compute\n")
ggml_backend_graph_compute(backend, graph)
flush_cat("M10: AFTER graph_compute OK\n")

res <- ggml_backend_tensor_get_data(t3)
flush_cat("M11: result[1:3]=", paste(round(res[1:3], 4), collapse = " "), "\n")
flush_cat("=GPU_MINIMAL_DONE=\n")
