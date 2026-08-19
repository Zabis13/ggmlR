#' Build Forward Computation Graph
#' 
#' Creates a computation graph by expanding backwards from the output tensor
#'
#' @param ctx GGML context
#' @param tensor Output tensor of the computation
#' @return Graph object (external pointer)
#' @export
#' @examples
#' \donttest{
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

#' Build a Forward Graph That Can Be Differentiated
#'
#' Like \code{\link{ggml_build_forward_expand}}, but the graph also reserves
#' storage for gradients, which is what \code{\link{ggml_build_backward_expand}}
#' requires.  A graph from the plain forward builder carries none, so passing it
#' to the backward builder is an error.
#'
#' Mark the tensors to differentiate with respect to using
#' \code{\link{ggml_set_param}} \emph{before} building the graph.
#'
#' @section Graph size:
#' \code{graph_size} is the node capacity.  The backward pass appends nodes to
#' the forward ones, so it needs more room than a forward-only graph: roughly
#' three times the forward node count is a safe starting point.  Too small a
#' value fails while the graph is being built, not silently.
#'
#' @param ctx GGML context
#' @param tensor Output tensor of the computation (usually a scalar loss)
#' @param graph_size Node capacity of the graph (default 2048)
#' @return Graph object (external pointer)
#' @seealso \code{\link{ggml_build_backward_expand}},
#'   \code{\link{ggml_graph_get_grad}}
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' ggml_set_f32(a, c(1, 2, 3, 4))
#' ggml_set_f32(b, c(5, 6, 7, 8))
#' ggml_set_param(a)
#' loss  <- ggml_sum(ctx, ggml_mul(ctx, a, b))
#' ggml_set_loss(loss)
#' graph <- ggml_build_forward_expand_grads(ctx, loss)
#' ggml_build_backward_expand(ctx, graph)
#' ggml_graph_reset(graph)     # seeds d(loss)/d(loss) = 1
#' ggml_graph_compute(ctx, graph)
#' # d(sum(a*b))/da == b
#' ggml_get_f32(ggml_graph_get_grad(graph, a))
#' ggml_free(ctx)
#' }
ggml_build_forward_expand_grads <- function(ctx, tensor, graph_size = 2048L) {
  .Call("R_ggml_build_forward_expand_grads", ctx, tensor,
        as.integer(graph_size), PACKAGE = "ggmlR")
}

#' Add the Backward Pass to a Computation Graph
#'
#' Extends \code{graph} in place with the nodes that compute gradients of its
#' output with respect to every tensor marked by \code{\link{ggml_set_param}}.
#' After computing the graph, read a gradient with
#' \code{\link{ggml_graph_get_grad}}.
#'
#' The graph must come from \code{\link{ggml_build_forward_expand_grads}}:
#' \code{\link{ggml_build_forward_expand}} produces one without gradient
#' storage, and that is rejected rather than allowed to abort inside ggml.
#'
#' @section The full sequence:
#' Every step is required; skipping one gives either an error or -- for the
#' last -- gradients that are silently all zero.
#' \enumerate{
#'   \item \code{\link{ggml_set_param}} on each tensor to differentiate
#'   \item build the forward ops
#'   \item \code{\link{ggml_set_loss}} on the output
#'   \item \code{\link{ggml_build_forward_expand_grads}}
#'   \item \code{ggml_build_backward_expand()}
#'   \item \code{\link{ggml_graph_reset}} -- seeds \code{d(loss)/d(loss) = 1}
#'   \item \code{\link{ggml_graph_compute}}
#'   \item \code{\link{ggml_graph_get_grad}} to read a gradient
#' }
#'
#' @section Which ops are differentiable:
#' Gradients flow only through operations that implement a backward pass.  Most
#' do, but not all -- \code{\link{ggml_clamp}} is a notable exception, and among
#' the state-space ops only \code{\link{ggml_ssm_conv}} is differentiable.
#'
#' @param ctx GGML context
#' @param graph Graph from \code{\link{ggml_build_forward_expand_grads}}
#' @return \code{NULL}, invisibly; \code{graph} is modified in place
#' @seealso \code{\link{ggml_build_forward_expand_grads}},
#'   \code{\link{ggml_graph_get_grad}}, \code{\link{ggml_set_param}}
#' @export
ggml_build_backward_expand <- function(ctx, graph) {
  invisible(.Call("R_ggml_build_backward_expand", ctx, graph, PACKAGE = "ggmlR"))
}

#' Get the Gradient Tensor of a Node
#'
#' Returns the tensor holding \code{d(output)/d(node)} after
#' \code{\link{ggml_build_backward_expand}} has added the backward pass and the
#' graph has been computed.
#'
#' Returns \code{NULL} when the node has no gradient in this graph -- because it
#' was never marked with \code{\link{ggml_set_param}}, or because nothing
#' differentiable connects it to the output.
#'
#' @param graph Graph carrying a backward pass
#' @param node Tensor whose gradient is wanted
#' @return The gradient tensor, or \code{NULL}
#' @seealso \code{\link{ggml_build_backward_expand}}
#' @export
ggml_graph_get_grad <- function(graph, node) {
  .Call("R_ggml_graph_get_grad", graph, node, PACKAGE = "ggmlR")
}

#' @rdname ggml_graph_get_grad
#' @return For \code{ggml_graph_get_grad_acc()}, the gradient accumulator
#'   tensor, or \code{NULL}
#' @export
ggml_graph_get_grad_acc <- function(graph, node) {
  .Call("R_ggml_graph_get_grad_acc", graph, node, PACKAGE = "ggmlR")
}

#' Add Another Root to an Existing Computation Graph
#'
#' \code{ggml_build_forward_expand()} always creates a fresh graph, so it can
#' only express a single root.  A model with several \emph{independent} output
#' branches needs every output expanded into the same graph: an output that is
#' unreachable from the first root never enters the graph, the scheduler never
#' assigns it a buffer, and reading it back fails.  This appends \code{tensor}
#' and its ancestors to a graph that already exists.
#'
#' @param graph Graph object returned by \code{ggml_build_forward_expand()}.
#' @param tensor Additional output tensor to expand into \code{graph}.
#' @return \code{NULL}, invisibly.  \code{graph} is modified in place.
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' ggml_set_f32(a, 1:4)
#' ggml_set_f32(b, 5:8)
#' # Two independent outputs -- neither reachable from the other.
#' o1 <- ggml_add(ctx, a, a)
#' o2 <- ggml_add(ctx, b, b)
#' graph <- ggml_build_forward_expand(ctx, o1)
#' ggml_graph_expand(graph, o2)
#' ggml_graph_compute(ctx, graph)
#' ggml_free(ctx)
#' }
ggml_graph_expand <- function(graph, tensor) {
  invisible(.Call("R_ggml_graph_expand", graph, tensor, PACKAGE = "ggmlR"))
}

#' Compute Computation Graph
#' 
#' Executes the computation graph using CPU backend
#'
#' @param ctx GGML context
#' @param graph Graph object created by ggml_build_forward_expand
#' @return No return value, called for side effects
#' @export
#' @examples
#' \donttest{
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
#' @return No return value, called for side effects
#' @export
ggml_graph_print <- function(graph) {
  invisible(.Call("R_ggml_graph_print", graph, PACKAGE = "ggmlR"))
}

#' Reset Graph (for backpropagation)
#'
#' Zeroes every gradient in the graph and seeds the loss node's own gradient
#' with 1 -- the value the chain rule starts from.  \strong{Call this after
#' \code{\link{ggml_build_backward_expand}} and before
#' \code{\link{ggml_graph_compute}}}: without it every gradient computes as
#' zero, since the backward pass multiplies through a seed that was never set.
#'
#' Requires a graph built by \code{\link{ggml_build_forward_expand_grads}};
#' an inference-only graph has no gradients to reset and is rejected.
#'
#' @param graph Graph object with gradients allocated
#' @return No return value, called for side effects
#' @seealso \code{\link{ggml_build_backward_expand}},
#'   \code{\link{ggml_build_forward_expand_grads}}
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
#' \donttest{
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
#' @return No return value, called for side effects
#' @export
#' @examples
#' \donttest{
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
#' @return No return value, called for side effects
#' @export
#' @examples
#' \donttest{
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
#' \donttest{
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
#' @return No return value, called for side effects
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
#' \donttest{
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
#' @return No return value, called for side effects
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
#' @return No return value, called for side effects
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

# ============================================================================
# Graph Introspection
# ============================================================================

#' Create a View of a Subgraph
#'
#' Creates a view of a portion of a computation graph, containing nodes
#' from index i0 to i1 (exclusive). The view shares the underlying nodes
#' but does not include leaf tensors or gradients.
#'
#' @param graph External pointer to computation graph
#' @param i0 Start index (0-based, inclusive)
#' @param i1 End index (exclusive)
#' @return External pointer to graph view
#' @export
#' @family graph
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
#' b <- ggml_relu(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, b)
#' n_nodes <- ggml_graph_n_nodes(graph)
#' view <- ggml_graph_view(graph, 0, n_nodes)
#' ggml_free(ctx)
#' }
ggml_graph_view <- function(graph, i0, i1) {
  .Call("R_ggml_graph_view", graph, as.integer(i0), as.integer(i1),
        PACKAGE = "ggmlR")
}

#' Check if Operation Can Be Done In-place
#'
#' Returns whether a GGML operation can reuse memory from its source tensors.
#' This is useful for memory optimization.
#'
#' @param op Operation code (integer)
#' @return Logical indicating if operation supports in-place execution
#' @export
#' @family graph
#' @examples
#' \donttest{
#' # Check if operation code 1 (ADD) can be in-place
#' can_inplace <- ggml_op_can_inplace(1L)
#' }
ggml_op_can_inplace <- function(op) {
  .Call("R_ggml_op_can_inplace", as.integer(op), PACKAGE = "ggmlR")
}

#' Check if Two Tensors Have the Same Layout
#'
#' Compares two tensors to check if they have identical type, shape,
#' and strides. Tensors with the same layout can be used interchangeably
#' for memory operations.
#'
#' @param a External pointer to first tensor
#' @param b External pointer to second tensor
#' @return Logical indicating if tensors have identical layout
#' @export
#' @family tensor
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4)
#' b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4)
#' same <- ggml_are_same_layout(a, b)  # TRUE
#' ggml_free(ctx)
#' }
ggml_are_same_layout <- function(a, b) {
  .Call("R_ggml_are_same_layout", a, b, PACKAGE = "ggmlR")
}

# ============================================================================
# Graph manipulation
#
# These wrap .Call entry points that were registered but had no R wrapper, so
# they were unreachable from R.
# ============================================================================

#' Duplicate a Computation Graph
#'
#' Copies \code{graph} into \code{ctx}.  With \code{force_grads = TRUE} the copy
#' carries gradient storage even when the original does not, which is one way to
#' turn a plain forward graph into one that
#' \code{\link{ggml_build_backward_expand}} accepts.
#'
#' @param ctx GGML context to allocate the copy in
#' @param graph Graph to duplicate
#' @param force_grads Logical; give the copy gradient storage (default
#'   \code{FALSE})
#' @return A new graph object (external pointer)
#' @seealso \code{\link{ggml_build_forward_expand_grads}}
#' @export
ggml_graph_dup <- function(ctx, graph, force_grads = FALSE) {
  .Call("R_ggml_graph_dup", ctx, graph, as.logical(force_grads),
        PACKAGE = "ggmlR")
}

#' Copy One Graph's Nodes Into Another
#'
#' Copies the contents of \code{src} into \code{dst}, which must already have
#' room for them.  Unlike \code{\link{ggml_graph_dup}} this allocates nothing.
#'
#' @param src Source graph
#' @param dst Destination graph
#' @return \code{NULL}, invisibly; \code{dst} is modified in place
#' @export
ggml_graph_cpy <- function(src, dst) {
  invisible(.Call("R_ggml_graph_cpy", src, dst, PACKAGE = "ggmlR"))
}

#' Clear a Computation Graph
#'
#' Removes every node from \code{graph}, keeping its allocated capacity so it
#' can be rebuilt without a fresh allocation.
#'
#' @param graph Graph to clear
#' @return \code{NULL}, invisibly; \code{graph} is modified in place
#' @export
ggml_graph_clear <- function(graph) {
  invisible(.Call("R_ggml_graph_clear", graph, PACKAGE = "ggmlR"))
}

#' Append a Node to a Computation Graph
#'
#' Adds \code{tensor} to \code{graph} as a node, without expanding its
#' dependencies.  For the usual case -- adding a tensor together with everything
#' it depends on -- use \code{\link{ggml_graph_expand}}.
#'
#' @param graph Graph to add to
#' @param tensor Tensor to append
#' @return \code{NULL}, invisibly; \code{graph} is modified in place
#' @seealso \code{\link{ggml_graph_expand}}
#' @export
ggml_graph_add_node <- function(graph, tensor) {
  invisible(.Call("R_ggml_graph_add_node", graph, tensor, PACKAGE = "ggmlR"))
}

#' Create a Graph Allocator for a Specific Buffer Type
#'
#' Like \code{\link{ggml_gallocr_new}}, but allocates from the given buffer type
#' rather than the CPU default -- so a graph can be allocated in GPU memory.
#'
#' @param buft Buffer type (e.g. from \code{ggml_backend_get_default_buffer_type()})
#' @return Graph allocator object (external pointer)
#' @seealso \code{\link{ggml_gallocr_new}}, \code{\link{ggml_gallocr_alloc_graph}}
#' @export
ggml_gallocr_new_buft <- function(buft) {
  .Call("R_ggml_gallocr_new_buft", buft, PACKAGE = "ggmlR")
}

#' Trace Every Node a Scheduler Computes
#'
#' Turns on a per-node trace: for each node the scheduler evaluates, its name,
#' op, shape and value summary (sum/min/max) are printed to stderr.  Nodes are
#' identified by NAME rather than index, so traces from two backends can be
#' diffed even when their graphs enumerate nodes differently -- which is what
#' makes this useful for finding where a CPU and a GPU run diverge.
#'
#' Off by default; leaving it on makes computation much slower.
#'
#' @param sched Backend scheduler
#' @param enable Logical; turn tracing on or off
#' @return \code{NULL}, invisibly
#' @export
ggml_backend_sched_trace <- function(sched, enable = TRUE) {
  invisible(.Call("R_ggml_backend_sched_trace", sched, as.logical(enable),
                  PACKAGE = "ggmlR"))
}
