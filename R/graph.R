#' Build Forward Computation Graph
#' 
#' Creates a computation graph by expanding backwards from the output tensor
#'
#' @param ctx GGML context
#' @param tensor Output tensor of the computation
#' @return Graph object (external pointer)
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
#' ggml_set_f32(a, 1:10)
#' ggml_set_f32(b, 11:20)
#' c <- ggml_add(ctx, a, b)
#' graph <- ggml_build_forward_expand(ctx, c)
#' ggml_graph_compute(ctx, graph)
#' result <- ggml_get_f32(c)
#' ggml_free(ctx)
#' }
ggml_build_forward_expand <- function(ctx, tensor) {
  .Call("R_ggml_build_forward_expand", ctx, tensor, PACKAGE = "ggmlR")
}

#' Compute Computation Graph
#' 
#' Executes the computation graph using CPU backend
#'
#' @param ctx GGML context
#' @param graph Graph object created by ggml_build_forward_expand
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
#' ggml_set_f32(a, 1:10)
#' ggml_set_f32(b, 11:20)
#' c <- ggml_add(ctx, a, b)
#' graph <- ggml_build_forward_expand(ctx, c)
#' ggml_graph_compute(ctx, graph)
#' result <- ggml_get_f32(c)
#' ggml_free(ctx)
#' }
ggml_graph_compute <- function(ctx, graph) {
  invisible(.Call("R_ggml_graph_compute", ctx, graph, PACKAGE = "ggmlR"))
}

#' Get Number of Nodes in Graph
#' 
#' Returns the number of computation nodes in the graph
#'
#' @param graph Graph object
#' @return Integer number of nodes
#' @export
ggml_graph_n_nodes <- function(graph) {
  .Call("R_ggml_graph_n_nodes", graph, PACKAGE = "ggmlR")
}

#' Print Graph Information
#'
#' Prints debug information about the computation graph
#'
#' @param graph Graph object
#' @export
ggml_graph_print <- function(graph) {
  invisible(.Call("R_ggml_graph_print", graph, PACKAGE = "ggmlR"))
}

#' Reset Graph (for backpropagation)
#'
#' Resets the computation graph for a new backward pass.
#' NOTE: This function requires the graph to have gradients allocated
#' (used for training/backpropagation). For inference-only graphs,
#' this function will cause an error.
#'
#' @param graph Graph object with gradients allocated
#' @export
ggml_graph_reset <- function(graph) {
  invisible(.Call("R_ggml_graph_reset", graph, PACKAGE = "ggmlR"))
}

#' Get Graph Node
#'
#' Gets a specific node (tensor) from the computation graph by index
#'
#' @param graph Graph object
#' @param i Node index (0-based, negative indices count from end)
#' @return Tensor pointer
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
#' b <- ggml_add(ctx, a, a)
#' graph <- ggml_build_forward_expand(ctx, b)
#' # Get the last node (output)
#' output <- ggml_graph_node(graph, -1)
#' ggml_free(ctx)
#' }
ggml_graph_node <- function(graph, i) {
  .Call("R_ggml_graph_node", graph, as.integer(i), PACKAGE = "ggmlR")
}

#' Get Graph Overhead
#'
#' Returns the memory overhead required for a computation graph
#'
#' @return Size in bytes
#' @export
ggml_graph_overhead <- function() {
  .Call("R_ggml_graph_overhead", PACKAGE = "ggmlR")
}

#' Get Tensor from Graph by Name
#'
#' Finds a tensor in the computation graph by its name
#'
#' @param graph Graph object
#' @param name Character string with tensor name
#' @return Tensor pointer or NULL if not found
#' @export
ggml_graph_get_tensor <- function(graph, name) {
  .Call("R_ggml_graph_get_tensor", graph, as.character(name), PACKAGE = "ggmlR")
}

#' Compute Graph with Context (Alternative Method)
#'
#' Computes the computation graph using the context-based method.
#' This is an alternative to ggml_graph_compute() that uses
#' ggml_graph_plan() and ggml_graph_compute() internally.
#'
#' @param ctx GGML context
#' @param graph Graph object created by ggml_build_forward_expand
#' @param n_threads Number of threads to use (0 for auto-detect, default: 0)
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
#' ggml_set_f32(a, 1:10)
#' c <- ggml_relu(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, c)
#' ggml_graph_compute_with_ctx(ctx, graph)
#' result <- ggml_get_f32(c)
#' ggml_free(ctx)
#' }
ggml_graph_compute_with_ctx <- function(ctx, graph, n_threads = 0L) {
  invisible(.Call("R_ggml_graph_compute_with_ctx", ctx, graph,
                  as.integer(n_threads), PACKAGE = "ggmlR"))
}

#' Export Graph to DOT Format
#'
#' Exports the computation graph to a DOT file for visualization.
#' The DOT file can be converted to an image using Graphviz tools.
#'
#' @param graph Graph object
#' @param leafs Optional graph with leaf tensors (NULL for none)
#' @param filename Output filename (should end with .dot)
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
#' b <- ggml_relu(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, b)
#' ggml_graph_dump_dot(graph, NULL, tempfile(fileext = ".dot"))
#' ggml_free(ctx)
#' }
ggml_graph_dump_dot <- function(graph, leafs = NULL, filename) {
  invisible(.Call("R_ggml_graph_dump_dot", graph, leafs,
                  as.character(filename), PACKAGE = "ggmlR"))
}

# ============================================================================
# Graph Allocator Functions
# ============================================================================

#' Create Graph Allocator
#'
#' Creates a new graph allocator for efficient memory management.
#' The allocator can automatically allocate and reuse memory for graph tensors.
#'
#' @return Graph allocator object (external pointer)
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' galloc <- ggml_gallocr_new()
#'
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
#' b <- ggml_relu(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, b)
#'
#' # Allocate graph
#' ggml_gallocr_alloc_graph(galloc, graph)
#'
#' ggml_gallocr_free(galloc)
#' ggml_free(ctx)
#' }
ggml_gallocr_new <- function() {
  .Call("R_ggml_gallocr_new", PACKAGE = "ggmlR")
}

#' Free Graph Allocator
#'
#' Frees a graph allocator and all associated buffers.
#'
#' @param galloc Graph allocator object
#' @export
ggml_gallocr_free <- function(galloc) {
  invisible(.Call("R_ggml_gallocr_free", galloc, PACKAGE = "ggmlR"))
}

#' Reserve Memory for Graph
#'
#' Pre-allocates memory for a graph. This is optional but recommended
#' when running the same graph multiple times to avoid reallocation.
#'
#' @param galloc Graph allocator object
#' @param graph Graph object
#' @return TRUE on success, FALSE on failure
#' @export
ggml_gallocr_reserve <- function(galloc, graph) {
  .Call("R_ggml_gallocr_reserve", galloc, graph, PACKAGE = "ggmlR")
}

#' Allocate Memory for Graph
#'
#' Allocates memory for all tensors in the computation graph.
#' This must be called before computing the graph.
#'
#' @param galloc Graph allocator object
#' @param graph Graph object
#' @return TRUE on success, FALSE on failure
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' galloc <- ggml_gallocr_new()
#'
#' # Create graph
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
#' b <- ggml_relu(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, b)
#'
#' # Allocate and compute
#' ggml_gallocr_alloc_graph(galloc, graph)
#' ggml_graph_compute(ctx, graph)
#'
#' ggml_gallocr_free(galloc)
#' ggml_free(ctx)
#' }
ggml_gallocr_alloc_graph <- function(galloc, graph) {
  .Call("R_ggml_gallocr_alloc_graph", galloc, graph, PACKAGE = "ggmlR")
}

#' Get Graph Allocator Buffer Size
#'
#' Returns the size of the buffer used by the graph allocator.
#'
#' @param galloc Graph allocator object
#' @param buffer_id Buffer ID (default: 0 for single-buffer allocator)
#' @return Size in bytes
#' @export
ggml_gallocr_get_buffer_size <- function(galloc, buffer_id = 0L) {
  .Call("R_ggml_gallocr_get_buffer_size", galloc, as.integer(buffer_id),
        PACKAGE = "ggmlR")
}

# ============================================================================
# Backend Tensor Functions
# ============================================================================

#' Set Tensor Data via Backend
#'
#' Sets tensor data using the backend API. This works with tensors
#' allocated on any backend, not just CPU.
#'
#' @param tensor Tensor pointer
#' @param data R vector with data to set
#' @param offset Byte offset (default: 0)
#' @export
ggml_backend_tensor_set_data <- function(tensor, data, offset = 0) {
  invisible(.Call("R_ggml_backend_tensor_set", tensor, data,
                  as.numeric(offset), PACKAGE = "ggmlR"))
}

#' Get Tensor Data via Backend
#'
#' Gets tensor data using the backend API. This works with tensors
#' allocated on any backend, not just CPU.
#'
#' @param tensor Tensor pointer
#' @param offset Byte offset (default: 0)
#' @param n_elements Number of elements to retrieve (NULL for all)
#' @return R vector with tensor data
#' @export
ggml_backend_tensor_get_data <- function(tensor, offset = 0, n_elements = NULL) {
  .Call("R_ggml_backend_tensor_get", tensor, as.numeric(offset),
        n_elements, PACKAGE = "ggmlR")
}

#' Allocate Context Tensors to Backend
#'
#' Allocates all tensors in a GGML context to a specific backend.
#' Returns a buffer that must be freed when no longer needed.
#'
#' @param ctx GGML context
#' @param backend Backend handle
#' @return Backend buffer object
#' @export
ggml_backend_alloc_ctx_tensors <- function(ctx, backend) {
  .Call("R_ggml_backend_alloc_ctx_tensors", ctx, backend, PACKAGE = "ggmlR")
}

# ============================================================================
# Backend Buffer Functions
# ============================================================================

#' Free Backend Buffer
#'
#' Frees a backend buffer and all associated memory.
#'
#' @param buffer Backend buffer object
#' @export
ggml_backend_buffer_free <- function(buffer) {
  invisible(.Call("R_ggml_backend_buffer_free", buffer, PACKAGE = "ggmlR"))
}

#' Get Backend Buffer Size
#'
#' Returns the total size of a backend buffer.
#'
#' @param buffer Backend buffer object
#' @return Size in bytes
#' @export
ggml_backend_buffer_get_size <- function(buffer) {
  .Call("R_ggml_backend_buffer_get_size", buffer, PACKAGE = "ggmlR")
}

#' Get Backend Buffer Name
#'
#' Returns the name/type of a backend buffer.
#'
#' @param buffer Backend buffer object
#' @return Character string with buffer name
#' @export
ggml_backend_buffer_name <- function(buffer) {
  .Call("R_ggml_backend_buffer_name", buffer, PACKAGE = "ggmlR")
}
