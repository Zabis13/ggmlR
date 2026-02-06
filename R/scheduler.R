# Backend Scheduler Functions for Multi-GPU Support

#' Create a new backend scheduler
#'
#' Creates a scheduler that can distribute computation across multiple backends (GPUs, CPU).
#' A CPU backend is automatically added as a fallback. Backends with lower index have higher priority.
#'
#' @param backends List of backend pointers (from ggml_vulkan_init() or ggml_backend_cpu_init()).
#'        Note: A CPU backend is automatically added, so you only need to specify GPU backends.
#' @param parallel Logical, whether to run backends in parallel (default: TRUE)
#' @param graph_size Expected maximum graph size (default: 2048)
#'
#' @return Scheduler pointer
#' @export
#' @examples
#' \donttest{
#' if (ggml_vulkan_available() && ggml_vulkan_device_count() >= 2) {
#'   # Create two GPU backends (CPU is added automatically)
#'   gpu1 <- ggml_vulkan_init(0)
#'   gpu2 <- ggml_vulkan_init(1)
#'
#'   # Create scheduler with both GPUs + CPU (automatic)
#'   sched <- ggml_backend_sched_new(list(gpu1, gpu2), parallel = TRUE)
#'
#'   # The scheduler now has 3 backends: GPU1, GPU2, CPU
#'   cat("Backends:", ggml_backend_sched_get_n_backends(sched), "\\n")
#'
#'   # Use scheduler...
#'
#'   # Cleanup
#'   ggml_backend_sched_free(sched)
#'   ggml_vulkan_free(gpu1)
#'   ggml_vulkan_free(gpu2)
#' }
#' }
ggml_backend_sched_new <- function(backends, parallel = TRUE, graph_size = 2048) {
  if (!is.list(backends) || length(backends) == 0) {
    stop("backends must be a non-empty list of backend pointers")
  }

  .Call("R_ggml_backend_sched_new",
        backends,
        as.logical(parallel),
        as.numeric(graph_size),
        PACKAGE = "ggmlR")
}

#' Free backend scheduler
#'
#' Releases resources associated with the backend scheduler.
#'
#' @param sched Scheduler pointer from ggml_backend_sched_new()
#' @return NULL (invisible)
#' @export
#' @examples
#' \donttest{
#' cpu <- ggml_backend_cpu_init()
#' sched <- ggml_backend_sched_new(list(cpu))
#' ggml_backend_sched_free(sched)
#' ggml_backend_free(cpu)
#' }
ggml_backend_sched_free <- function(sched) {
  invisible(.Call("R_ggml_backend_sched_free", sched, PACKAGE = "ggmlR"))
}

#' Reserve memory for scheduler
#'
#' Pre-allocates memory based on a measurement graph. This should be called
#' before using the scheduler to compute graphs.
#'
#' @param sched Scheduler pointer
#' @param graph Graph pointer to measure memory requirements
#' @return Logical indicating success
#' @export
#' @examples
#' \donttest{
#' cpu <- ggml_backend_cpu_init()
#' sched <- ggml_backend_sched_new(list(cpu))
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000)
#' c <- ggml_add(ctx, a, b)
#' graph <- ggml_build_forward_expand(ctx, c)
#' ggml_backend_sched_reserve(sched, graph)
#' ggml_backend_sched_free(sched)
#' ggml_backend_free(cpu)
#' ggml_free(ctx)
#' }
ggml_backend_sched_reserve <- function(sched, graph) {
  .Call("R_ggml_backend_sched_reserve", sched, graph, PACKAGE = "ggmlR")
}

#' Get number of backends in scheduler
#'
#' Returns the number of backends managed by the scheduler.
#'
#' @param sched Scheduler pointer
#' @return Integer count of backends
#' @export
ggml_backend_sched_get_n_backends <- function(sched) {
  .Call("R_ggml_backend_sched_get_n_backends", sched, PACKAGE = "ggmlR")
}

#' Get backend from scheduler
#'
#' Returns a specific backend from the scheduler by index.
#'
#' @param sched Scheduler pointer
#' @param index Backend index (0-based)
#' @return Backend pointer
#' @export
ggml_backend_sched_get_backend <- function(sched, index = 0L) {
  .Call("R_ggml_backend_sched_get_backend", sched, as.integer(index), PACKAGE = "ggmlR")
}

#' Get number of graph splits
#'
#' Returns the number of splits in the last computed graph.
#' Higher numbers indicate more distribution across backends.
#'
#' @param sched Scheduler pointer
#' @return Integer count of splits
#' @export
ggml_backend_sched_get_n_splits <- function(sched) {
  .Call("R_ggml_backend_sched_get_n_splits", sched, PACKAGE = "ggmlR")
}

#' Get number of tensor copies
#'
#' Returns the number of tensor copies made in the last computed graph.
#' Copies occur when data needs to be transferred between backends.
#'
#' @param sched Scheduler pointer
#' @return Integer count of copies
#' @export
ggml_backend_sched_get_n_copies <- function(sched) {
  .Call("R_ggml_backend_sched_get_n_copies", sched, PACKAGE = "ggmlR")
}

#' Set tensor backend assignment
#'
#' Manually assigns a specific tensor to run on a specific backend.
#' This overrides automatic scheduling.
#'
#' @param sched Scheduler pointer
#' @param tensor Tensor pointer
#' @param backend Backend pointer to assign tensor to
#' @return NULL (invisible)
#' @export
ggml_backend_sched_set_tensor_backend <- function(sched, tensor, backend) {
  invisible(.Call("R_ggml_backend_sched_set_tensor_backend",
                  sched, tensor, backend, PACKAGE = "ggmlR"))
}

#' Get tensor backend assignment
#'
#' Returns which backend a tensor is assigned to.
#'
#' @param sched Scheduler pointer
#' @param tensor Tensor pointer
#' @return Backend pointer or NULL if not assigned
#' @export
ggml_backend_sched_get_tensor_backend <- function(sched, tensor) {
  .Call("R_ggml_backend_sched_get_tensor_backend", sched, tensor, PACKAGE = "ggmlR")
}

#' Allocate graph on scheduler
#'
#' Allocates memory for a graph across the scheduler's backends.
#' Must be called before computing the graph.
#'
#' @param sched Scheduler pointer
#' @param graph Graph pointer
#' @return Logical indicating success
#' @export
ggml_backend_sched_alloc_graph <- function(sched, graph) {
  .Call("R_ggml_backend_sched_alloc_graph", sched, graph, PACKAGE = "ggmlR")
}

#' Compute graph using scheduler
#'
#' Computes a graph by distributing work across multiple backends.
#' This is the main function for multi-GPU computation.
#'
#' @param sched Scheduler pointer
#' @param graph Graph pointer
#' @return Status code (0 = success)
#' @export
#' @examples
#' \donttest{
#' # Multi-GPU example
#' if (ggml_vulkan_available() && ggml_vulkan_device_count() >= 2) {
#'   gpu1 <- ggml_vulkan_init(0)
#'   gpu2 <- ggml_vulkan_init(1)
#'   sched <- ggml_backend_sched_new(list(gpu1, gpu2))
#'
#'   ctx <- ggml_init(64 * 1024 * 1024)
#'   a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10000)
#'   b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10000)
#'   ggml_set_f32(a, rnorm(10000))
#'   ggml_set_f32(b, rnorm(10000))
#'
#'   c <- ggml_add(ctx, a, b)
#'   graph <- ggml_build_forward_expand(ctx, c)
#'
#'   # Reserve memory
#'   ggml_backend_sched_reserve(sched, graph)
#'
#'   # Compute using both GPUs
#'   ggml_backend_sched_graph_compute(sched, graph)
#'
#'   result <- ggml_get_f32(c)
#'
#'   cat("Splits:", ggml_backend_sched_get_n_splits(sched), "\n")
#'   cat("Copies:", ggml_backend_sched_get_n_copies(sched), "\n")
#'
#'   ggml_free(ctx)
#'   ggml_backend_sched_free(sched)
#'   ggml_vulkan_free(gpu1)
#'   ggml_vulkan_free(gpu2)
#' }
#' }
ggml_backend_sched_graph_compute <- function(sched, graph) {
  .Call("R_ggml_backend_sched_graph_compute", sched, graph, PACKAGE = "ggmlR")
}

#' Compute graph asynchronously
#'
#' Computes a graph asynchronously across backends.
#' Use ggml_backend_sched_synchronize() to wait for completion.
#'
#' @param sched Scheduler pointer
#' @param graph Graph pointer
#' @return Status code (0 = success)
#' @export
ggml_backend_sched_graph_compute_async <- function(sched, graph) {
  .Call("R_ggml_backend_sched_graph_compute_async", sched, graph, PACKAGE = "ggmlR")
}

#' Synchronize scheduler
#'
#' Waits for all asynchronous operations to complete.
#'
#' @param sched Scheduler pointer
#' @return NULL (invisible)
#' @export
ggml_backend_sched_synchronize <- function(sched) {
  invisible(.Call("R_ggml_backend_sched_synchronize", sched, PACKAGE = "ggmlR"))
}

#' Reset scheduler
#'
#' Resets the scheduler, deallocating all tensors.
#' Must be called before changing node backends or allocating a new graph.
#'
#' @param sched Scheduler pointer
#' @return NULL (invisible)
#' @export
ggml_backend_sched_reset <- function(sched) {
  invisible(.Call("R_ggml_backend_sched_reset", sched, PACKAGE = "ggmlR"))
}
