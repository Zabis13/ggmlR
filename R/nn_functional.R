# Functional API for ggmlR
# Allows building arbitrary DAG computation graphs (skip connections, residual
# blocks, multi-input / multi-output models) using a Keras-functional style.
#
# Key design: ggml_tensor_node objects store the **configuration** of the graph
# (like Sequential model$layers), not live ggml tensors.  Real ggml tensors are
# created lazily in nn_build_functional_graph() when compile/fit is called and
# batch_size is known.

# ============================================================================
# Counter for auto-generated node IDs and layer IDs
# ============================================================================

.fn_node_counter <- new.env(parent = emptyenv())
.fn_node_counter$n <- 0L
# Per-type counters for auto-generated layer names (input_1, add_2, ...)
.fn_type_counters <- new.env(parent = emptyenv())
# Counter for ggml_layer object IDs (shared-layer identity)
.fn_layer_counter <- new.env(parent = emptyenv())
.fn_layer_counter$n <- 0L

nn_next_node_id <- function() {
  .fn_node_counter$n <- .fn_node_counter$n + 1L
  paste0("node_", .fn_node_counter$n)
}

nn_next_layer_id <- function() {
  .fn_layer_counter$n <- .fn_layer_counter$n + 1L
  paste0("layer_", .fn_layer_counter$n)
}

# Auto-generate a layer name like "input_1", "add_2"
nn_auto_name <- function(type) {
  cur <- if (is.null(.fn_type_counters[[type]])) 0L else .fn_type_counters[[type]]
  cur <- cur + 1L
  .fn_type_counters[[type]] <- cur
  paste0(type, "_", cur)
}

# ============================================================================
# Layer object constructors (for shared-layer / ggml_apply() workflow)
# ============================================================================

#' Create a Dense Layer Object
#'
#' Returns a reusable layer object for use with \code{ggml_apply()}.
#' Applying the same object to multiple tensor nodes shares weights.
#'
#' @param units Number of output units.
#' @param activation Activation function name or NULL.
#' @param name Optional character name.
#' @param trainable Logical; whether weights are updated during training.
#' @param time_distributed Logical; apply the kernel independently at every
#'   position of a sequence input rather than flattening it (default
#'   \code{FALSE}).  A \code{c(seq_len, features)} parent then yields
#'   \code{c(seq_len, units)} instead of a flat \code{units}, which is what a
#'   transformer's position-wise feed-forward sublayer needs.  See
#'   \code{\link{ggml_layer_dense}} for details.
#' @return A \code{ggml_layer} object.
#' @export
#' @examples
#' \donttest{
#' encoder <- ggml_dense(64L, activation = "relu")
#' x1 <- ggml_input(shape = 32L)
#' x2 <- ggml_input(shape = 32L)
#' out1 <- x1 |> ggml_apply(encoder)
#' out2 <- x2 |> ggml_apply(encoder)  # shared weights
#' }
ggml_dense <- function(units, activation = NULL, name = NULL, trainable = TRUE,
                       time_distributed = FALSE) {
  if (is.null(name)) name <- nn_auto_name("dense")
  structure(
    list(
      layer_id  = nn_next_layer_id(),
      node_type = "dense",
      name      = name,
      config    = list(units = as.integer(units), activation = activation,
                       time_distributed = isTRUE(time_distributed)),
      trainable = trainable
    ),
    class = "ggml_layer"
  )
}

#' Create a Reusable Conv2D Layer Object
#'
#' Builds a \emph{layer object} to be applied with \code{\link{ggml_apply}},
#' which is how a single set of convolution weights is shared between several
#' inputs.  This is the convolution counterpart of \code{\link{ggml_dense}}.
#'
#' For the ordinary pipe-style form that appends a conv layer to a graph
#' (\code{x |> ggml_layer_conv_2d(...)}), see \code{\link{ggml_layer_conv_2d}}.
#'
#' @param filters Number of output filters.
#' @param kernel_size Integer or length-2 integer vector.
#' @param activation Activation function name or NULL.
#' @param strides Integer or length-2 integer vector (default 1).
#' @param padding \code{"valid"} or \code{"same"}.
#' @param name Optional character name.
#' @param trainable Logical.
#' @return A \code{ggml_layer} object.
#' @seealso \code{\link{ggml_apply}}, \code{\link{ggml_dense}}
#' @export
#' @examples
#' \donttest{
#' # One convolution shared between two inputs.
#' shared <- ggml_conv_2d_layer(filters = 4L, kernel_size = 3L, activation = "relu")
#' x1 <- ggml_input(shape = c(8L, 8L, 1L))
#' x2 <- ggml_input(shape = c(8L, 8L, 1L))
#' o1 <- ggml_apply(x1, shared)
#' o2 <- ggml_apply(x2, shared)
#' stopifnot(identical(o1$layer_id, o2$layer_id))   # same weights
#' }
ggml_conv_2d_layer <- function(filters, kernel_size, activation = NULL,
                          strides = c(1L, 1L), padding = "valid",
                          name = NULL, trainable = TRUE) {
  if (length(kernel_size) == 1L) kernel_size <- rep(as.integer(kernel_size), 2L)
  if (length(strides)     == 1L) strides     <- rep(as.integer(strides),     2L)
  if (is.null(name)) name <- nn_auto_name("conv_2d")
  structure(
    list(
      layer_id  = nn_next_layer_id(),
      node_type = "conv_2d",
      name      = name,
      config    = list(filters     = as.integer(filters),
                       kernel_size = as.integer(kernel_size),
                       strides     = as.integer(strides),
                       padding     = padding,
                       activation  = activation),
      trainable = trainable
    ),
    class = "ggml_layer"
  )
}

#' Create a Reusable Conv1D Layer Object
#'
#' Builds a \emph{layer object} to be applied with \code{\link{ggml_apply}},
#' which is how a single set of convolution weights is shared between several
#' inputs.  This is the convolution counterpart of \code{\link{ggml_dense}}.
#'
#' For the ordinary pipe-style form that appends a conv layer to a graph
#' (\code{x |> ggml_layer_conv_1d(...)}), see \code{\link{ggml_layer_conv_1d}}.
#'
#' @param filters Number of output filters.
#' @param kernel_size Integer kernel size.
#' @param activation Activation function name or NULL.
#' @param strides Integer stride (default 1).
#' @param padding \code{"valid"} or \code{"same"}.
#' @param name Optional character name.
#' @param trainable Logical.
#' @return A \code{ggml_layer} object.
#' @seealso \code{\link{ggml_apply}}, \code{\link{ggml_dense}}
#' @export
#' @examples
#' \donttest{
#' # One convolution shared between two sequence inputs.
#' shared <- ggml_conv_1d_layer(filters = 4L, kernel_size = 3L)
#' x1 <- ggml_input(shape = c(16L, 2L))
#' x2 <- ggml_input(shape = c(16L, 2L))
#' o1 <- ggml_apply(x1, shared)
#' o2 <- ggml_apply(x2, shared)
#' stopifnot(identical(o1$layer_id, o2$layer_id))   # same weights
#' }
ggml_conv_1d_layer <- function(filters, kernel_size, activation = NULL,
                          strides = 1L, padding = "valid",
                          name = NULL, trainable = TRUE) {
  if (is.null(name)) name <- nn_auto_name("conv_1d")
  structure(
    list(
      layer_id  = nn_next_layer_id(),
      node_type = "conv_1d",
      name      = name,
      config    = list(filters     = as.integer(filters),
                       kernel_size = as.integer(kernel_size),
                       strides     = as.integer(strides),
                       padding     = padding,
                       activation  = activation),
      trainable = trainable
    ),
    class = "ggml_layer"
  )
}

#' Create a Batch Normalization Layer Object
#'
#' @param eps Small constant for numerical stability (default 1e-5).
#' @param name Optional character name.
#' @param trainable Logical.
#' @return A \code{ggml_layer} object.
#' @export
ggml_batch_norm <- function(eps = 1e-5, name = NULL, trainable = TRUE) {
  if (is.null(name)) name <- nn_auto_name("batch_norm")
  structure(
    list(
      layer_id  = nn_next_layer_id(),
      node_type = "batch_norm",
      name      = name,
      config    = list(eps = eps),
      trainable = trainable
    ),
    class = "ggml_layer"
  )
}

#' Create an Embedding Layer Object
#'
#' @param vocab_size Number of distinct tokens.
#' @param dim Embedding dimension.
#' @param name Optional character name.
#' @param trainable Logical.
#' @return A \code{ggml_layer} object.
#' @export
ggml_embedding <- function(vocab_size, dim, name = NULL, trainable = TRUE) {
  if (is.null(name)) name <- nn_auto_name("embedding")
  structure(
    list(
      layer_id  = nn_next_layer_id(),
      node_type = "embedding",
      name      = name,
      config    = list(vocab_size = as.integer(vocab_size),
                       dim        = as.integer(dim)),
      trainable = trainable
    ),
    class = "ggml_layer"
  )
}

#' Create an LSTM Layer Object
#'
#' @param units Integer, number of hidden units.
#' @param return_sequences Logical.
#' @param activation Cell gate activation (default \code{"tanh"}).
#' @param recurrent_activation Recurrent gate activation (default \code{"sigmoid"}).
#' @param name Optional character name.
#' @param trainable Logical.
#' @return A \code{ggml_layer} object.
#' @export
ggml_lstm <- function(units, return_sequences = FALSE,
                       activation = "tanh", recurrent_activation = "sigmoid",
                       name = NULL, trainable = TRUE) {
  if (is.null(name)) name <- nn_auto_name("lstm")
  structure(
    list(
      layer_id  = nn_next_layer_id(),
      node_type = "lstm",
      name      = name,
      config    = list(units                = as.integer(units),
                       return_sequences     = return_sequences,
                       activation           = activation,
                       recurrent_activation = recurrent_activation),
      trainable = trainable
    ),
    class = "ggml_layer"
  )
}

#' Create a GRU Layer Object
#'
#' @param units Integer, number of hidden units.
#' @param return_sequences Logical.
#' @param activation Candidate activation (default \code{"tanh"}).
#' @param recurrent_activation Gate activation (default \code{"sigmoid"}).
#' @param name Optional character name.
#' @param trainable Logical.
#' @return A \code{ggml_layer} object.
#' @export
ggml_gru <- function(units, return_sequences = FALSE,
                      activation = "tanh", recurrent_activation = "sigmoid",
                      name = NULL, trainable = TRUE) {
  if (is.null(name)) name <- nn_auto_name("gru")
  structure(
    list(
      layer_id  = nn_next_layer_id(),
      node_type = "gru",
      name      = name,
      config    = list(units                = as.integer(units),
                       return_sequences     = return_sequences,
                       activation           = activation,
                       recurrent_activation = recurrent_activation),
      trainable = trainable
    ),
    class = "ggml_layer"
  )
}

#' Create a Multi-Head Attention Layer Object
#'
#' Returns a reusable layer object for use with \code{\link{ggml_apply}}, the
#' scaled dot-product attention of \emph{Attention Is All You Need} with
#' \code{n_heads} heads computed in parallel.  Applying the same object to
#' several tensor nodes shares one set of projection weights, which is how an
#' encoder block is reused across a stack.
#'
#' For the pipe-style form that appends attention to a graph
#' (\code{x |> ggml_layer_attention(...)}), see
#' \code{\link{ggml_layer_attention}}.
#'
#' @section Shapes:
#' The parent must be a sequence node of shape \code{c(seq_len, d_model)} --
#' the same layout \code{\link{ggml_layer_lstm}} takes -- and the output has
#' that same shape, so attention blocks stack directly.
#'
#' @section Self- and cross-attention:
#' Applied to a single node the layer is self-attention: queries, keys and
#' values all come from that node.  Applied to a list of two nodes,
#' \code{ggml_apply(list(query, context), attn)}, it is cross-attention:
#' queries come from the first, keys and values from the second, which may
#' have a different sequence length (but must share \code{d_model}).  This is
#' the decoder-to-encoder path of a full transformer.
#'
#' @section Weights:
#' Four projections \code{W_q}, \code{W_k}, \code{W_v}, \code{W_o}, each
#' \code{d_model x d_model}, plus an output bias \code{b_o} when
#' \code{bias = TRUE}. Glorot-uniform initialised, zero bias.
#'
#' @param d_model Integer, model width. Must be divisible by \code{n_heads}
#'   and must match the parent's feature dimension.
#' @param n_heads Integer, number of attention heads (default 1).
#' @param causal Logical: mask out positions after the current one, so a query
#'   attends only to keys at or before it (default \code{FALSE}). This is what
#'   makes a decoder autoregressive. Only meaningful for self-attention --
#'   masking a cross-attention score matrix by position compares indices in two
#'   unrelated sequences, so it is rejected rather than silently applied.
#' @param bias Logical: add a bias to the output projection (default
#'   \code{TRUE}).
#' @param name Optional character name.
#' @param trainable Logical; whether weights are updated during training.
#' @return A \code{ggml_layer} object.
#' @seealso \code{\link{ggml_layer_attention}}, \code{\link{ggml_apply}}
#' @export
#' @examples
#' \donttest{
#' # One attention block shared between two sequences.
#' attn <- ggml_attention(d_model = 32L, n_heads = 4L)
#' x1   <- ggml_input(shape = c(10L, 32L))
#' x2   <- ggml_input(shape = c(10L, 32L))
#' o1   <- ggml_apply(x1, attn)
#' o2   <- ggml_apply(x2, attn)          # shared weights
#' stopifnot(identical(o1$layer_id, o2$layer_id))
#'
#' # Cross-attention: queries from x1, keys/values from a longer context.
#' ctx  <- ggml_input(shape = c(15L, 32L))
#' o3   <- ggml_apply(list(x1, ctx), ggml_attention(32L, 4L))
#' }
ggml_attention <- function(d_model, n_heads = 1L, causal = FALSE, bias = TRUE,
                           name = NULL, trainable = TRUE) {
  d_model <- as.integer(d_model)
  n_heads <- as.integer(n_heads)
  if (is.na(d_model) || d_model < 1L) {
    stop("'d_model' must be a positive integer.", call. = FALSE)
  }
  if (is.na(n_heads) || n_heads < 1L) {
    stop("'n_heads' must be a positive integer.", call. = FALSE)
  }
  # Every head takes an equal slice of the feature axis, so an indivisible
  # d_model would leave a remainder with nowhere to go.
  if (d_model %% n_heads != 0L) {
    stop("'d_model' (", d_model, ") must be divisible by 'n_heads' (", n_heads,
         ").", call. = FALSE)
  }
  if (is.null(name)) name <- nn_auto_name("attention")
  structure(
    list(
      layer_id  = nn_next_layer_id(),
      node_type = "attention",
      name      = name,
      config    = list(d_model = d_model,
                       n_heads = n_heads,
                       causal  = isTRUE(causal),
                       bias    = isTRUE(bias)),
      trainable = trainable
    ),
    class = "ggml_layer"
  )
}

# ============================================================================
# ggml_apply() -- apply a ggml_layer object to a tensor node
# ============================================================================

#' Apply a Layer Object to a Tensor Node
#'
#' Applies a \code{ggml_layer} object (created with \code{ggml_dense()},
#' \code{ggml_lstm()}, etc.) to a \code{ggml_tensor_node}.  Applying the
#' \emph{same} layer object to multiple tensor nodes produces shared weights --
#' the identity of the layer object (\code{layer$layer_id}) is used as the
#' sharing key, not its name.
#'
#' @param tensor A \code{ggml_tensor_node} (e.g. from \code{ggml_input()}).
#' @param layer A \code{ggml_layer} object.
#' @return A new \code{ggml_tensor_node}.
#' @export
#' @examples
#' \donttest{
#' encoder <- ggml_dense(64L, activation = "relu")
#' x1 <- ggml_input(shape = 32L)
#' x2 <- ggml_input(shape = 32L)
#' out1 <- x1 |> ggml_apply(encoder)
#' out2 <- x2 |> ggml_apply(encoder)  # shared weights
#' model <- ggml_model(inputs = list(x1, x2),
#'                     outputs = list(out1, out2))
#' }
ggml_apply <- function(tensor, layer) {
  if (!inherits(layer, "ggml_layer")) {
    stop("'layer' must be a ggml_layer object (from ggml_dense(), ggml_lstm(), etc.).")
  }
  # A list of nodes is how a layer taking several inputs is applied -- currently
  # cross-attention, ggml_apply(list(query, context), attn). A bare node stays
  # the single-parent form every other layer uses.
  parents <- if (inherits(tensor, "ggml_tensor_node")) {
    list(tensor)
  } else if (is.list(tensor) &&
             length(tensor) > 0L &&
             all(vapply(tensor, inherits, logical(1), "ggml_tensor_node"))) {
    tensor
  } else {
    stop("'tensor' must be a ggml_tensor_node (from ggml_input() or a layer ",
         "call), or a list of them for a layer taking several inputs.",
         call. = FALSE)
  }
  if (length(parents) > 1L && !identical(layer$node_type, "attention")) {
    stop("'", layer$node_type, "' takes a single input; a list of ",
         length(parents), " was given.", call. = FALSE)
  }
  if (identical(layer$node_type, "attention") && length(parents) > 2L) {
    stop("attention takes one input (self-attention) or two ",
         "(cross-attention: query, context); ", length(parents), " were given.",
         call. = FALSE)
  }
  structure(
    list(
      id        = nn_next_node_id(),
      node_type = layer$node_type,
      layer_id  = layer$layer_id,   # sharing key -- identity of the layer object
      trainable = layer$trainable,
      config    = c(layer$config, list(name = layer$name)),
      parents   = parents
    ),
    class = "ggml_tensor_node"
  )
}

# ============================================================================
# ggml_input() -- declare an input tensor
# ============================================================================

#' Declare a Functional API Input Tensor
#'
#' Creates a symbolic input node for the Functional API.  The node records
#' only the \emph{shape} of one sample (without batch dimension); actual
#' memory is allocated when \code{ggml_compile()} is called.
#'
#' @param shape Integer vector describing the shape of a single sample.
#'   For flat feature vectors use a scalar, e.g. \code{shape = 64L}.
#'   For 2-D inputs (sequences) use \code{c(length, channels)}.
#'   For 3-D inputs (images) use \code{c(H, W, C)}.
#' @param name Optional character name for the input tensor.
#' @param dtype Data type of the input: \code{"float32"} (default) or
#'   \code{"int32"} (for embedding/token-index inputs).
#' @return A \code{ggml_tensor_node} object.
#' @export
#' @examples
#' \donttest{
#' x <- ggml_input(shape = 64L)
#' x <- ggml_input(shape = c(28L, 28L, 1L), name = "image")
#' x <- ggml_input(shape = 10L, dtype = "int32")  # token indices
#' }
ggml_input <- function(shape, name = NULL, dtype = "float32") {
  shape <- as.integer(shape)
  if (is.null(name)) name <- nn_auto_name("input")
  if (!dtype %in% c("float32", "int32")) {
    stop("dtype must be 'float32' or 'int32', got: ", dtype)
  }

  structure(
    list(
      id        = nn_next_node_id(),
      node_type = "input",
      config    = list(shape = shape, name = name, dtype = dtype),
      parents   = list()
    ),
    class = "ggml_tensor_node"
  )
}

# ============================================================================
# ggml_model() -- assemble a functional model from input/output nodes
# ============================================================================

#' Create a Functional Model
#'
#' Assembles a \code{ggml_functional_model} from symbolic input and output
#' nodes produced by \code{ggml_input()} and \code{ggml_layer_*()} calls.
#'
#' @param inputs A \code{ggml_tensor_node} or a list of them (model inputs).
#' @param outputs A \code{ggml_tensor_node} or a list of them (model outputs).
#' @return A \code{ggml_functional_model} object.
#' @export
#' @examples
#' \donttest{
#' x   <- ggml_input(shape = 64L)
#' out <- x |> ggml_layer_dense(10, activation = "softmax")
#' model <- ggml_model(inputs = x, outputs = out)
#' }
ggml_model <- function(inputs, outputs) {
  if (inherits(inputs, "ggml_tensor_node")) inputs <- list(inputs)
  if (inherits(outputs, "ggml_tensor_node")) outputs <- list(outputs)

  if (!is.list(inputs) || !all(vapply(inputs, inherits, logical(1), "ggml_tensor_node"))) {
    stop("'inputs' must be a ggml_tensor_node or a list of ggml_tensor_node objects.")
  }
  if (!is.list(outputs) || !all(vapply(outputs, inherits, logical(1), "ggml_tensor_node"))) {
    stop("'outputs' must be a ggml_tensor_node or a list of ggml_tensor_node objects.")
  }
  # All inputs must be declared with ggml_input() (node_type == "input")
  bad <- which(vapply(inputs, function(n) n$node_type, character(1)) != "input")
  if (length(bad) > 0L) {
    stop("'inputs[[", bad[1], "]]' has node_type '",
         inputs[[bad[1]]]$node_type,
         "' -- only nodes created with ggml_input() are valid model inputs.")
  }

  structure(
    list(
      inputs      = inputs,
      outputs     = outputs,
      compiled    = FALSE,
      compilation = list(
        sched      = NULL,
        backend    = NULL,
        optimizer  = NULL,
        loss       = NULL,
        metrics    = NULL
      )
    ),
    class = c("ggml_functional_model", "list")
  )
}

# ============================================================================
# ggml_layer_attention()
# ============================================================================

#' Add a Multi-Head Attention Layer
#'
#' Scaled dot-product attention with \code{n_heads} heads, in the pipe-style
#' form that appends a layer to a functional graph.  For the reusable
#' \emph{layer object} that shares one set of weights between several
#' applications -- how an encoder block is reused across a stack -- see
#' \code{\link{ggml_attention}}.
#'
#' @section Shapes:
#' \code{x} must be a sequence node of shape \code{c(seq_len, d_model)}, the
#' same layout \code{\link{ggml_layer_lstm}} takes.  The output has that same
#' shape, so attention blocks stack directly and a residual connection
#' (\code{ggml_layer_add(list(x, attn_out))}) needs no reshaping.
#'
#' @section How it is computed:
#' All heads are computed in one batched pass rather than in a loop: the
#' projections are reshaped to \code{[d_head, n_heads, seq, batch]} and
#' permuted so that \code{ggml_mul_mat()} batches over the head and batch axes.
#' The node count is therefore independent of \code{n_heads}.
#'
#' @section Cross-attention:
#' Pass a list of two nodes, \code{ggml_layer_attention(list(query, context),
#' ...)}, to take queries from the first and keys/values from the second.  The
#' two may differ in sequence length but must share \code{d_model}.  A single
#' node is self-attention.
#'
#' @param x A \code{ggml_tensor_node} of shape \code{c(seq_len, d_model)}, or a
#'   list of two such nodes \code{list(query, context)} for cross-attention.
#' @param d_model Integer, model width. Must be divisible by \code{n_heads} and
#'   match the parent's feature dimension.
#' @param n_heads Integer, number of attention heads (default 1).
#' @param causal Logical: mask out positions after the current one, making the
#'   layer autoregressive (default \code{FALSE}). Self-attention only.
#' @param bias Logical: add a bias to the output projection (default
#'   \code{TRUE}).
#' @param name Optional layer name.
#' @param trainable Logical.
#' @return A new \code{ggml_tensor_node} of shape \code{c(seq_len, d_model)}.
#' @seealso \code{\link{ggml_attention}} for the shared-weights layer object.
#' @export
#' @examples
#' \donttest{
#' # A transformer encoder block: attention + residual, then a feed-forward
#' # sublayer + residual.
#' x    <- ggml_input(shape = c(10L, 32L))
#' attn <- x |> ggml_layer_attention(d_model = 32L, n_heads = 4L)
#' h    <- ggml_layer_add(list(x, attn))
#' ff   <- h |> ggml_layer_dense(32L, activation = "relu")
#' out  <- ggml_layer_add(list(h, ff))
#'
#' # GPT-style causal self-attention.
#' dec <- x |> ggml_layer_attention(32L, n_heads = 4L, causal = TRUE)
#' }
ggml_layer_attention <- function(x, d_model, n_heads = 1L, causal = FALSE,
                                 bias = TRUE, name = NULL, trainable = TRUE) {
  # Built from the layer object so the two forms cannot validate differently.
  layer <- ggml_attention(d_model = d_model, n_heads = n_heads, causal = causal,
                          bias = bias, name = name, trainable = trainable)
  ggml_apply(x, layer)
}

# ============================================================================
# ggml_layer_add() / ggml_layer_concatenate()
# ============================================================================

#' Element-wise Addition of Two Tensor Nodes
#'
#' Adds two (or more) tensor nodes element-wise.  All tensors must have the
#' same shape.  This is the functional equivalent of a residual / skip
#' connection.
#'
#' @param tensors A list of \code{ggml_tensor_node} objects (length >= 2).
#' @param name Optional character name for the layer.
#' @return A new \code{ggml_tensor_node} representing the sum.
#' @export
#' @examples
#' \donttest{
#' x    <- ggml_input(shape = 64L)
#' a    <- x |> ggml_layer_dense(64, activation = "relu")
#' b    <- x |> ggml_layer_dense(64)
#' out  <- ggml_layer_add(list(a, b))
#' }
ggml_layer_add <- function(tensors, name = NULL) {
  if (!is.list(tensors) || length(tensors) < 2L) {
    stop("'tensors' must be a list of at least 2 ggml_tensor_node objects.")
  }
  if (!all(vapply(tensors, inherits, logical(1), "ggml_tensor_node"))) {
    stop("All elements of 'tensors' must be ggml_tensor_node objects.")
  }
  if (is.null(name)) name <- nn_auto_name("add")

  structure(
    list(
      id        = nn_next_node_id(),
      node_type = "add",
      config    = list(name = name),
      parents   = tensors
    ),
    class = "ggml_tensor_node"
  )
}

#' Concatenate Tensor Nodes Along an Axis
#'
#' Concatenates two or more tensor nodes along the specified axis.
#'
#' @param tensors A list of \code{ggml_tensor_node} objects (length >= 2).
#' @param axis Integer axis along which to concatenate (0-based, ggml convention).
#'   Default \code{0L} concatenates along the first dimension (features for
#'   flat tensors).
#' @param name Optional character name for the layer.
#' @return A new \code{ggml_tensor_node} representing the concatenated tensor.
#' @export
#' @examples
#' \donttest{
#' x   <- ggml_input(shape = 32L)
#' y   <- ggml_input(shape = 32L)
#' out <- ggml_layer_concatenate(list(x, y), axis = 0L)
#' }
ggml_layer_concatenate <- function(tensors, axis = 0L, name = NULL) {
  if (!is.list(tensors) || length(tensors) < 2L) {
    stop("'tensors' must be a list of at least 2 ggml_tensor_node objects.")
  }
  if (!all(vapply(tensors, inherits, logical(1), "ggml_tensor_node"))) {
    stop("All elements of 'tensors' must be ggml_tensor_node objects.")
  }
  if (is.null(name)) name <- nn_auto_name("concatenate")
  # axis stored as-is (may be negative); resolved at shape inference time

  structure(
    list(
      id        = nn_next_node_id(),
      node_type = "concatenate",
      config    = list(axis = as.integer(axis), name = name),
      parents   = tensors
    ),
    class = "ggml_tensor_node"
  )
}

# ============================================================================
# ggml_layer_custom()
# ============================================================================

#' Define a Custom Layer From an R Function
#'
#' Turns an arbitrary composition of ggml operations into a reusable functional
#' layer.  \code{ggml_layer_custom()} does not build anything itself — it
#' returns a \emph{layer function} that behaves like any other
#' \code{ggml_layer_*()} and can be piped into a model.
#'
#' The \code{forward} function is called once while the compute graph is being
#' built, with the active \code{ggml_context} and the parent tensor.  Because
#' the returned tensor becomes an ordinary graph node, gradients flow through a
#' custom layer automatically — as long as every operation used inside
#' \code{forward} implements a backward pass.
#'
#' @section Weights:
#' Custom layers are stateless: \code{forward} maps activations to activations
#' and owns no trainable parameters.  Put learnable weights in a neighbouring
#' \code{ggml_layer_dense()}, or use the autograd API (\code{ag_*}) when a layer
#' needs its own parameters.
#'
#' @param forward A function of \code{(ctx, x)} returning a ggml tensor, where
#'   \code{ctx} is the compute context and \code{x} the parent tensor.  All
#'   ggml operations take \code{ctx} as their first argument.
#' @param name Optional character name for the layer.  Used as the base for
#'   auto-generated node names.
#' @param output_shape Output shape excluding the batch dimension.  Defaults to
#'   \code{NULL}, meaning the layer preserves its input shape (correct for
#'   element-wise layers such as custom activations).  Supply an integer vector
#'   when \code{forward} changes the shape, or a function of the input shape.
#' @return A layer function of \code{(x, name = NULL)} that appends a custom
#'   node to the graph and returns the new \code{ggml_tensor_node}.
#' @export
#' @examples
#' \donttest{
#' # Mish activation: x * tanh(softplus(x))
#' layer_mish <- ggml_layer_custom(
#'   name    = "mish",
#'   forward = function(ctx, x) ggml_mul(ctx, x, ggml_tanh(ctx, ggml_softplus(ctx, x)))
#' )
#'
#' inp <- ggml_input(shape = 8L)
#' out <- inp |>
#'   ggml_layer_dense(16L) |>
#'   layer_mish() |>
#'   ggml_layer_dense(2L, activation = "softmax")
#' model <- ggml_model(inputs = inp, outputs = out)
#' }
ggml_layer_custom <- function(forward, name = NULL, output_shape = NULL) {
  if (!is.function(forward)) {
    stop("'forward' must be a function of (ctx, x) returning a ggml tensor.")
  }
  if (length(formals(forward)) < 2L) {
    stop("'forward' must accept two arguments: (ctx, x). ",
         "Every ggml operation takes the context as its first argument, ",
         "e.g. forward = function(ctx, x) ggml_relu(ctx, x).")
  }
  if (!is.null(output_shape) &&
      !is.function(output_shape) && !is.numeric(output_shape)) {
    stop("'output_shape' must be NULL, an integer vector, or a function ",
         "of the input shape.")
  }

  base_name <- if (is.null(name)) "custom" else name

  # The returned layer function is what the user pipes into.
  function(x, name = NULL) {
    if (!inherits(x, "ggml_tensor_node")) {
      stop("A custom layer must be applied to a ggml_tensor_node ",
           "(the output of ggml_input() or another layer).")
    }
    node_name <- if (!is.null(name)) name else nn_auto_name(base_name)

    structure(
      list(
        id        = nn_next_node_id(),
        node_type = "custom",
        config    = list(
          name         = node_name,
          forward      = forward,
          output_shape = output_shape
        ),
        parents = list(x)
      ),
      class = "ggml_tensor_node"
    )
  }
}

# ============================================================================
# Topological sort
# ============================================================================

# Loss of one output head, computed on the R side for evaluate(). Mirrors the
# formulas the ggml graph uses, so evaluate() and fit() report comparable
# numbers for the same loss name.
nn_head_loss <- function(loss_name, preds_mat, y) {
  if (nn_loss_is_ce(loss_name)) {
    eps <- 1e-7
    preds_clipped <- pmax(pmin(preds_mat, 1 - eps), eps)
    -mean(rowSums(y * log(preds_clipped)))
  } else if (loss_name %in% c("mse", "mean_squared_error")) {
    mean(rowSums((y - preds_mat)^2) / ncol(y))
  } else if (loss_name %in% c("mae", "mean_absolute_error")) {
    mean(abs(y - preds_mat))
  } else if (loss_name %in% c("huber", "huber_loss")) {
    # delta = 1, matching the training graph: quadratic inside the unit
    # interval, linear outside it.
    e <- abs(y - preds_mat)
    mean(ifelse(e <= 1, 0.5 * e^2, e - 0.5))
  } else if (loss_name %in% c("binary_crossentropy", "binary_cross_entropy")) {
    # Element-wise, so the mean is over every element rather than over rows --
    # the same normalisation the training graph uses.
    eps <- 1e-7
    p <- pmax(pmin(preds_mat, 1 - eps), eps)
    -mean(y * log(p) + (1 - y) * log(1 - p))
  } else {
    NA_real_
  }
}

# Append the requested regression metrics to an evaluate() result.
#
# Accuracy is handled separately: it is reported for multi-class outputs whether
# or not it was asked for. These are computed only on request, and are shared by
# the sequential and functional evaluate() paths so both report the same number
# under the same name.
#
# Unknown names are dropped silently here -- ggml_compile() has already warned
# about them, and warning again at every evaluate() would be noise.
nn_add_extra_metrics <- function(out, metrics, preds_mat, y) {
  extra <- setdiff(as.character(metrics), c("accuracy", "acc"))
  for (m in extra) {
    val <- switch(m,
      "mae"  = , "mean_absolute_error" = mean(abs(y - preds_mat)),
      "mse"  = , "mean_squared_error"  = mean((y - preds_mat)^2),
      "rmse" = sqrt(mean((y - preds_mat)^2)),
      NULL
    )
    if (!is.null(val)) out[[m]] <- val
  }
  out
}

# Names of a functional model's output heads, used to match `loss`/`loss_weights`
# entries by name and to label the per-head training history. Falls back to the
# node id when a layer was created without an explicit name.
nn_output_names <- function(model) {
  vapply(model$outputs, function(n) {
    nm <- n$config$name
    if (is.null(nm) || !nzchar(nm)) n$id else nm
  }, character(1))
}

#' Topologically sort nodes reachable from output nodes
#'
#' @param outputs List of output ggml_tensor_node objects
#' @return Named list: nodes in topological order (inputs first, outputs last)
#' @export
nn_topo_sort <- function(outputs) {
  visited <- list()
  ordered <- list()

  visit <- function(node) {
    if (isTRUE(visited[[node$id]])) return()
    visited[[node$id]] <<- TRUE
    for (parent in node$parents) {
      visit(parent)
    }
    ordered[[length(ordered) + 1L]] <<- node
  }

  for (out in outputs) visit(out)
  ordered
}

# ============================================================================
# Build functional graph (analogous to nn_build_graph for Sequential)
# ============================================================================

#' Estimate the scheduler graph size a functional model needs
#'
#' The backend scheduler sizes its hash set once, before the graph exists, and
#' aborts inside ggml if the graph turns out larger. Most layers contribute a
#' handful of nodes, but a recurrent layer unrolls per timestep, so the count
#' has to follow the sequence length rather than the layer count.
#'
#' Generous on purpose: the cost of overshooting is a slightly larger hash set,
#' the cost of undershooting is a crash with no R-level error to catch.
#'
#' @return A double, the graph size to allocate.
#' @keywords internal
nn_functional_graph_size <- function(model) {
  nodes <- nn_topo_sort(model$outputs)

  shapes <- list()
  total  <- 0
  for (node in nodes) {
    parent_shapes <- lapply(node$parents, function(p) shapes[[p$id]])

    # Sizing the scheduler is not the place to validate the model. A bad axis
    # or a mismatched shape has to surface from the build, with the build's
    # message and its backtrace -- so an inference failure here just falls back
    # to the default and lets the real check downstream report it.
    shape <- tryCatch(nn_functional_output_shape(node, parent_shapes),
                      error = function(e) NULL)
    if (is.null(shape)) return(2048)
    shapes[[node$id]] <- shape

    total <- total + switch(node$node_type,
      # One unrolled block per timestep, plus the gate arithmetic inside it.
      "lstm" = ,
      "gru"  = {
        psh <- if (length(parent_shapes) > 0) parent_shapes[[1]] else NULL
        seq_len <- if (length(psh) >= 1) as.numeric(psh[1]) else 1
        seq_len * 40 + 32
      },
      # All heads are computed in one batched pass, so the node count is fixed
      # rather than proportional to n_heads: three projections with their
      # reshape/permute, the two score matmuls, the softmax and the output
      # projection, rounded up.
      "attention" = 48,
      32       # everything else: a few nodes, rounded up for backward
    )
  }

  # Backward roughly doubles the node count; 2048 stays the floor so nothing
  # that used to fit is given less than it was.
  max(2048, total * 3)
}

# Features one dense kernel maps from, given its parent's shape.
#
# An ordinary dense layer flattens its input, so a c(seq_len, d_model) parent
# contributes seq_len*d_model features and the sequence axis disappears. A
# time-distributed one applies the same kernel at every position instead, so it
# maps from d_model alone and the sequence survives -- which is what a
# transformer's position-wise feed-forward sublayer needs.
#
# Shared by the shape inference, the weight sizing, the builder and the weight
# initialisation, so the four cannot disagree about the kernel's width.
nn_dense_fan_in <- function(node, psh) {
  if (isTRUE(node$config$time_distributed) && length(psh) >= 2L) {
    as.numeric(psh[length(psh)])
  } else if (length(psh) == 1L) {
    as.numeric(psh)
  } else {
    prod(as.numeric(psh))
  }
}

#' Count the weight elements a functional node will allocate
#'
#' Mirrors the tensor shapes \code{nn_build_functional_node} creates in
#' \code{ctx_weights}, so the context can be sized before the tensors exist.
#' Weights do not scale with the batch, which is what makes this a separate
#' count from the activations rather than a factor applied to them.
#'
#' Returns a double: a wide dense layer can exceed integer range on its own.
#'
#' @return A double, the number of float elements the node's weights occupy.
#' @keywords internal
nn_functional_weight_elements <- function(node, parent_shapes) {
  psh    <- if (length(parent_shapes) > 0) parent_shapes[[1]] else NULL
  fan_in <- if (is.null(psh)) 0 else prod(as.numeric(psh))

  switch(node$node_type,
    "dense" = {
      units <- as.numeric(node$config$units)
      nn_dense_fan_in(node, psh) * units + units    # kernel + bias
    },
    "embedding" = as.numeric(node$config$dim) * as.numeric(node$config$vocab_size),
    "batch_norm" = 2 * fan_in,                     # gamma + beta
    "attention" = {
      # Four d_model x d_model projections (q, k, v, out) plus the output bias.
      d <- as.numeric(node$config$d_model)
      4 * d * d + if (isTRUE(node$config$bias)) d else 0
    },
    "conv_2d" = {
      k  <- as.numeric(node$config$kernel_size)
      ic <- if (length(psh) >= 3) as.numeric(psh[3]) else 1
      oc <- as.numeric(node$config$filters)
      k[1] * k[2] * ic * oc + oc
    },
    "conv_1d" = {
      k  <- as.numeric(node$config$kernel_size)
      ic <- if (length(psh) >= 2) as.numeric(psh[2]) else 1
      oc <- as.numeric(node$config$filters)
      k * ic * oc + oc
    },
    "lstm" = {
      units <- as.numeric(node$config$units)
      isz   <- if (length(psh) >= 2) as.numeric(psh[2]) else fan_in
      isz * 4 * units + units * 4 * units + 4 * units + 2 * units
    },
    "gru" = {
      units <- as.numeric(node$config$units)
      isz   <- if (length(psh) >= 2) as.numeric(psh[2]) else fan_in
      isz * 2 * units + units * 2 * units + 2 * units +
        isz * units + units * units + units + units
    },
    0                                              # input, flatten, dropout, ...
  )
}

#' Infer output shape of a functional node given its parent shapes
#' @return An integer vector with the inferred output shape (excluding the batch dimension).
#' @keywords internal
nn_functional_output_shape <- function(node, parent_shapes) {
  switch(node$node_type,
    "input" = node$config$shape,
    "dense" = {
      # An ordinary dense layer flattens, so its output is just `units`. A
      # time-distributed one keeps the sequence axis and replaces only the
      # feature axis: c(seq_len, d_in) -> c(seq_len, units).
      if (isTRUE(node$config$time_distributed)) {
        psh <- parent_shapes[[1]]
        if (length(psh) != 2L) {
          stop("dense(time_distributed = TRUE) expects a sequence input of ",
               "shape c(seq_len, features); got a shape with ", length(psh),
               " dimension(s).", call. = FALSE)
        }
        as.integer(c(psh[1], node$config$units))
      } else {
        as.integer(node$config$units)
      }
    },
    "flatten" = {
      psh <- parent_shapes[[1]]
      as.integer(prod(psh))
    },
    "batch_norm" = parent_shapes[[1]],
    "add" = parent_shapes[[1]],
    "attention" = {
      # Queries decide the output length, so cross-attention returns the query
      # sequence's length with the context's contribution folded in. The width
      # is d_model either way -- the output projection maps back to it.
      psh <- parent_shapes[[1]]
      if (length(psh) != 2L) {
        stop("attention expects a sequence input of shape c(seq_len, d_model); ",
             "got a shape with ", length(psh), " dimension(s).", call. = FALSE)
      }
      d_model <- node$config$d_model
      if (psh[2] != d_model) {
        stop("attention: 'd_model' is ", d_model, " but the input has ",
             psh[2], " features.", call. = FALSE)
      }
      # Keys and values must live in the same space as the queries: the same
      # W_k/W_v project both, and the scores are a dot product between them.
      if (length(parent_shapes) > 1L) {
        ksh <- parent_shapes[[2]]
        if (length(ksh) != 2L || ksh[2] != d_model) {
          stop("attention: the context input must be c(seq_len, ", d_model,
               "); got c(", paste(ksh, collapse = ", "), ").", call. = FALSE)
        }
      }
      as.integer(c(psh[1], d_model))
    },
    "custom" = {
      osh <- node$config$output_shape
      if (is.null(osh)) {
        # Element-wise by default: the layer preserves its input shape.
        parent_shapes[[1]]
      } else if (is.function(osh)) {
        as.integer(osh(parent_shapes[[1]]))
      } else {
        as.integer(osh)
      }
    },
    "concatenate" = {
      ndim <- length(parent_shapes[[1]])
      axis <- node$config$axis  # 0-based, may be negative
      # Resolve negative axis (e.g. -1 -> last dimension)
      if (axis < 0L) axis <- ndim + axis
      if (axis < 0L || axis >= ndim) {
        stop("ggml_layer_concatenate: axis ", node$config$axis,
             " is out of range for tensors with ", ndim, " dimensions ",
             "(valid range: [", -ndim, ", ", ndim - 1L, "]).")
      }
      total <- 0L
      for (psh in parent_shapes) {
        total <- total + psh[axis + 1L]
      }
      out <- parent_shapes[[1]]
      out[axis + 1L] <- total
      out
    },
    "conv_2d" = {
      psh <- parent_shapes[[1]]  # c(H, W, C) R-order
      H <- psh[1]; W <- psh[2]
      kh <- node$config$kernel_size[1]; kw <- node$config$kernel_size[2]
      sh <- node$config$strides[1]; sw <- node$config$strides[2]
      if (node$config$padding == "same") {
        H_out <- ceiling(H / sh); W_out <- ceiling(W / sw)
      } else {
        H_out <- floor((H - kh) / sh) + 1L
        W_out <- floor((W - kw) / sw) + 1L
      }
      as.integer(c(H_out, W_out, node$config$filters))
    },
    "max_pooling_2d" = {
      psh <- parent_shapes[[1]]
      H <- psh[1]; W <- psh[2]; C <- psh[3]
      ph <- node$config$pool_size[1]; pw <- node$config$pool_size[2]
      sh <- node$config$strides[1]; sw <- node$config$strides[2]
      H_out <- floor((H - ph) / sh) + 1L
      W_out <- floor((W - pw) / sw) + 1L
      as.integer(c(H_out, W_out, C))
    },
    "conv_1d" = {
      psh <- parent_shapes[[1]]  # c(L, C)
      L <- psh[1]
      k <- node$config$kernel_size; s <- node$config$strides
      if (node$config$padding == "same") {
        L_out <- ceiling(L / s)
      } else {
        L_out <- floor((L - k) / s) + 1L
      }
      as.integer(c(L_out, node$config$filters))
    },
    "global_max_pooling_2d" = ,
    "global_average_pooling_2d" = {
      # [H, W, C] -> [C]
      psh <- parent_shapes[[1]]
      as.integer(psh[3])
    },
    "lstm" = {
      # input shape: c(seq_len, input_size)
      psh   <- parent_shapes[[1]]
      units <- node$config$units
      if (isTRUE(node$config$return_sequences)) {
        as.integer(c(psh[1], units))
      } else {
        as.integer(units)
      }
    },
    "gru" = {
      psh   <- parent_shapes[[1]]
      units <- node$config$units
      if (isTRUE(node$config$return_sequences)) {
        as.integer(c(psh[1], units))
      } else {
        as.integer(units)
      }
    },
    "dropout" = parent_shapes[[1]],  # shape unchanged
    "embedding" = {
      # input shape: c(seq_len) -> output: c(dim, seq_len)
      psh <- parent_shapes[[1]]
      seq_len <- if (length(psh) == 1L) psh else prod(psh)
      as.integer(c(node$config$dim, seq_len))
    },
    stop("Unknown node_type in shape inference: ", node$node_type)
  )
}

#' Build a single ggml tensor for one functional node
#' @param reuse_weights Named list of pre-allocated weight tensors to reuse
#'   (for shared layers -- second+ application of a named layer).  When not
#'   NULL the function uses these tensors instead of allocating new ones.
#' @return A \code{ggml_tensor} produced by building the given functional graph node.
#' @keywords internal
nn_build_functional_node <- function(node, built_tensors, built_shapes,
                                      ctx_weights, ctx_compute, batch_size,
                                      training = FALSE,
                                      reuse_weights = NULL) {
  switch(node$node_type,

    "input" = {
      shape <- node$config$shape
      dtype <- if (!is.null(node$config$dtype)) node$config$dtype else "float32"
      ggml_type <- if (dtype == "int32") GGML_TYPE_I32 else GGML_TYPE_F32
      # Create tensor with proper dimensionality so spatial ops (conv, pool)
      # see the correct ne[0..3] fields.
      t <- if (length(shape) == 3L) {
        # Image: R [H, W, C] -> ggml [W, H, C, N]
        ggml_new_tensor_4d(ctx_weights, ggml_type,
                           shape[2L], shape[1L], shape[3L], batch_size)
      } else if (length(shape) == 2L) {
        # Sequence: R [seq_len, input_size] -> ggml [input_size, seq_len, N]
        ggml_new_tensor_3d(ctx_weights, ggml_type,
                           shape[2L], shape[1L], batch_size)
      } else {
        # Flat: R [n] -> ggml [n, N]
        ggml_new_tensor_2d(ctx_weights, ggml_type, prod(shape), batch_size)
      }
      ggml_set_name(t, node$config$name)
      ggml_set_input(t)
      list(tensor = t, weights = list())
    },

    "dense" = {
      parent_id <- node$parents[[1]]$id
      input_t   <- built_tensors[[parent_id]]
      psh       <- built_shapes[[parent_id]]
      fan_in    <- as.integer(nn_dense_fan_in(node, psh))
      units     <- node$config$units
      td        <- isTRUE(node$config$time_distributed)

      if (!is.null(reuse_weights)) {
        W <- reuse_weights$weight
        b <- reuse_weights$bias
      } else {
        W <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, fan_in, units)
        b <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, units)
        nm <- if (!is.null(node$config$name)) node$config$name else node$id
        ggml_set_name(W, paste0(nm, "_weight"))
        ggml_set_name(b, paste0(nm, "_bias"))
      }

      # Time-distributed: one kernel applied at every position, sharing weights
      # across the sequence. ggml_mul_mat contracts ne[0], so feeding it
      # [fan_in, seq, N] against a [fan_in, units] kernel already batches over
      # the sequence and batch axes -- no loop, and the sequence axis survives
      # as [units, seq, N]. A parent that arrived flat gets its axes back first.
      if (td) {
        seq_len_i <- as.integer(psh[1])
        if (length(ggml_tensor_shape(input_t)) == 2L) {
          input_t <- ggml_reshape_3d(ctx_compute, input_t, fan_in, seq_len_i, batch_size)
        }
      }

      out <- ggml_mul_mat(ctx_compute, W, input_t)
      # ggml_add broadcasts the [units] bias over the trailing axes, so the same
      # call covers both the flat and the time-distributed case.
      out <- ggml_add(ctx_compute, out, b)
      out <- nn_apply_activation(ctx_compute, out, node$config$activation)

      list(tensor = out, weights = list(weight = W, bias = b))
    },

    "batch_norm" = {
      parent_id <- node$parents[[1]]$id
      input_t   <- built_tensors[[parent_id]]
      psh       <- built_shapes[[parent_id]]
      n_features <- if (length(psh) == 1L) psh
                    else if (length(psh) == 2L) psh[2]
                    else psh[3]

      if (!is.null(reuse_weights)) {
        gamma <- reuse_weights$gamma
        beta  <- reuse_weights$beta
      } else {
        gamma <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, n_features)
        beta  <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, n_features)
        nm <- if (!is.null(node$config$name)) node$config$name else node$id
        ggml_set_name(gamma, paste0(nm, "_gamma"))
        ggml_set_name(beta,  paste0(nm, "_beta"))
      }

      eps    <- node$config$eps
      normed <- ggml_rms_norm(ctx_compute, input_t, eps = eps)

      if (length(psh) == 3L) {
        gamma_r <- ggml_reshape_4d(ctx_compute, gamma, 1L, 1L, as.integer(psh[3]), 1L)
        beta_r  <- ggml_reshape_4d(ctx_compute, beta,  1L, 1L, as.integer(psh[3]), 1L)
      } else if (length(psh) == 2L) {
        gamma_r <- ggml_reshape_3d(ctx_compute, gamma, 1L, as.integer(psh[2]), 1L)
        beta_r  <- ggml_reshape_3d(ctx_compute, beta,  1L, as.integer(psh[2]), 1L)
      } else {
        gamma_r <- gamma
        beta_r  <- beta
      }

      out <- ggml_mul(ctx_compute, normed, gamma_r)
      out <- ggml_add(ctx_compute, out, beta_r)

      list(tensor = out, weights = list(gamma = gamma, beta = beta))
    },

    "attention" = {
      # Multi-head scaled dot-product attention, all heads in one batched pass.
      #
      # Shapes, in ggml order (ne[0] innermost). The parent is a sequence node
      # c(seq_len, d_model) in R order, so its tensor is [d_model, seq, N].
      # Every head owns d_head = d_model / n_heads contiguous features, which is
      # what makes the split a reshape rather than a gather.
      q_id    <- node$parents[[1]]$id
      q_in    <- built_tensors[[q_id]]
      q_psh   <- built_shapes[[q_id]]
      # Self-attention reads keys and values from the query node; cross-
      # attention from the second parent, which may have a different length.
      kv_id   <- if (length(node$parents) > 1L) node$parents[[2]]$id else q_id
      kv_in   <- built_tensors[[kv_id]]
      kv_psh  <- built_shapes[[kv_id]]

      d_model <- as.integer(node$config$d_model)
      n_heads <- as.integer(node$config$n_heads)
      d_head  <- d_model %/% n_heads
      seq_q   <- as.integer(q_psh[1])
      seq_kv  <- as.integer(kv_psh[1])
      causal  <- isTRUE(node$config$causal)
      use_b   <- isTRUE(node$config$bias)
      nm      <- if (!is.null(node$config$name)) node$config$name else node$id

      # Masking by position compares a query index against a key index, which
      # only means anything when the two come from the same sequence.
      if (causal && !identical(q_id, kv_id)) {
        stop("attention: 'causal' applies to self-attention; a cross-attention ",
             "layer has two unrelated sequences to compare positions in.",
             call. = FALSE)
      }

      if (!is.null(reuse_weights)) {
        W_q <- reuse_weights$W_q; W_k <- reuse_weights$W_k
        W_v <- reuse_weights$W_v; W_o <- reuse_weights$W_o
        b_o <- reuse_weights$b_o
      } else {
        W_q <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, d_model, d_model)
        W_k <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, d_model, d_model)
        W_v <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, d_model, d_model)
        W_o <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, d_model, d_model)
        b_o <- if (use_b) ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, d_model) else NULL
        ggml_set_name(W_q, paste0(nm, "_W_q"))
        ggml_set_name(W_k, paste0(nm, "_W_k"))
        ggml_set_name(W_v, paste0(nm, "_W_v"))
        ggml_set_name(W_o, paste0(nm, "_W_o"))
        if (use_b) ggml_set_name(b_o, paste0(nm, "_b_o"))
      }

      # A parent that came out of a dense/flatten layer is [d_model*seq, N];
      # give it back its sequence axis before projecting.
      as_seq3d <- function(t, seq_len_i) {
        if (length(ggml_tensor_shape(t)) == 2L) {
          ggml_reshape_3d(ctx_compute, t, d_model, seq_len_i, batch_size)
        } else {
          t
        }
      }
      q_3d  <- as_seq3d(q_in,  seq_q)
      kv_3d <- as_seq3d(kv_in, seq_kv)

      # Project: [d_model, d_model] x [d_model, seq, N] -> [d_model, seq, N].
      Q <- ggml_mul_mat(ctx_compute, W_q, q_3d)
      K <- ggml_mul_mat(ctx_compute, W_k, kv_3d)
      V <- ggml_mul_mat(ctx_compute, W_v, kv_3d)

      # Split the feature axis into heads and move the head axis out to dim 2,
      # where mul_mat batches over it: [d_model, seq, N] -> [d_head, seq, n_heads, N].
      # ggml_permute()'s arguments are DESTINATION positions: source axis 1
      # (n_heads) goes to position 2, source axis 2 (seq) to position 1.
      to_heads <- function(t, seq_len_i) {
        t4 <- ggml_reshape_4d(ctx_compute, t, d_head, n_heads, seq_len_i, batch_size)
        ggml_cont(ctx_compute, ggml_permute(ctx_compute, t4, 0L, 2L, 1L, 3L))
      }
      Qh <- to_heads(Q, seq_q)    # [d_head, seq_q,  n_heads, N]
      Kh <- to_heads(K, seq_kv)   # [d_head, seq_kv, n_heads, N]
      Vh <- to_heads(V, seq_kv)   # [d_head, seq_kv, n_heads, N]

      # scores[j, i] = <k_j, q_i>: ggml_mul_mat(A, B) contracts ne[0] of both,
      # so this is [seq_kv, seq_q, n_heads, N] -- one row per query, holding
      # that query's score against every key.
      scores <- ggml_mul_mat(ctx_compute, Kh, Qh)

      # Causal mask: -Inf above the diagonal, so query i cannot see key j > i.
      # Applied before the softmax, which then gives those positions weight 0.
      if (causal) {
        scores <- ggml_diag_mask_inf(ctx_compute, scores, 0L)
      }

      # Softmax runs over ne[0] = the key axis, which is exactly the axis each
      # query's weights must sum over. The 1/sqrt(d_head) scaling is fused in
      # rather than applied as a separate node.
      attn <- ggml_soft_max_ext(ctx_compute, scores, NULL,
                                scale = 1.0 / sqrt(d_head), max_bias = 0.0)

      # Weighted sum of values. mul_mat contracts ne[0], and attn's ne[0] is the
      # key axis, so V has to present its key axis there: transpose Vh to
      # [seq_kv, d_head, n_heads, N] and contract against attn's seq_kv.
      # Result: [d_head, seq_q, n_heads, N].
      Vt  <- ggml_cont(ctx_compute, ggml_transpose(ctx_compute, Vh))
      ctx_heads <- ggml_mul_mat(ctx_compute, Vt, attn)

      # Merge the heads back into one feature axis, undoing to_heads(): move
      # the head axis back beside d_head, then flatten the two.
      merged <- ggml_cont(ctx_compute,
                          ggml_permute(ctx_compute, ctx_heads, 0L, 2L, 1L, 3L))
      merged <- ggml_reshape_3d(ctx_compute, merged, d_model, seq_q, batch_size)

      # Output projection, back to [d_model, seq_q, N].
      out <- ggml_mul_mat(ctx_compute, W_o, merged)
      if (use_b) out <- ggml_add(ctx_compute, out, b_o)

      list(tensor = out,
           weights = c(list(W_q = W_q, W_k = W_k, W_v = W_v, W_o = W_o),
                       if (use_b) list(b_o = b_o) else list()))
    },

    "flatten" = {
      parent_id  <- node$parents[[1]]$id
      input_t    <- built_tensors[[parent_id]]
      psh        <- built_shapes[[parent_id]]
      n_features <- prod(psh)
      # Derive the batch size from the element count rather than from
      # ggml_n_dims(): ggml reports a trailing unit dimension as absent, so a
      # batch of 1 makes a [W, H, C, 1] input look 3-D and the batch size would
      # be read off the channel axis instead.
      bs <- as.integer(ggml_nelements(input_t) / n_features)
      out <- ggml_reshape_2d(ctx_compute, input_t, n_features, bs)
      list(tensor = out, weights = list())
    },

    "custom" = {
      input_t <- built_tensors[[node$parents[[1]]$id]]
      out <- node$config$forward(ctx_compute, input_t)
      if (is.null(out)) {
        stop("Custom layer '", node$config$name, "': forward() returned NULL. ",
             "It must return the ggml tensor produced by its operations.")
      }
      list(tensor = out, weights = list())
    },

    "add" = {
      tensors <- lapply(node$parents, function(p) built_tensors[[p$id]])
      # Validate shapes match
      ref_shape <- built_shapes[[node$parents[[1]]$id]]
      for (i in seq(2L, length(node$parents))) {
        sh <- built_shapes[[node$parents[[i]]$id]]
        if (!identical(as.integer(ref_shape), as.integer(sh))) {
          stop("ggml_layer_add: shape mismatch -- input 1 has shape [",
               paste(ref_shape, collapse = ", "), "] but input ", i,
               " has shape [", paste(sh, collapse = ", "), "].")
        }
      }
      out <- tensors[[1]]
      for (i in seq(2L, length(tensors))) {
        out <- ggml_add(ctx_compute, out, tensors[[i]])
      }
      list(tensor = out, weights = list())
    },

    "concatenate" = {
      parent_tensors <- lapply(node$parents, function(p) built_tensors[[p$id]])
      # Resolve axis (negative allowed)
      ndim <- length(built_shapes[[node$parents[[1]]$id]])
      axis <- node$config$axis
      if (axis < 0L) axis <- ndim + axis
      out <- parent_tensors[[1]]
      for (i in seq(2L, length(parent_tensors))) {
        out <- ggml_concat(ctx_compute, out, parent_tensors[[i]], dim = axis)
      }
      list(tensor = out, weights = list())
    },

    "conv_2d" = {
      parent_id <- node$parents[[1]]$id
      input_t   <- built_tensors[[parent_id]]
      psh       <- built_shapes[[parent_id]]  # c(H, W, C) R-order
      kh <- node$config$kernel_size[1]
      kw <- node$config$kernel_size[2]
      ic <- psh[3]
      oc <- node$config$filters

      if (!is.null(reuse_weights)) {
        kernel <- reuse_weights$kernel
        bias   <- reuse_weights$bias
      } else {
        kernel <- ggml_new_tensor_4d(ctx_weights, GGML_TYPE_F32, kw, kh, ic, oc)
        bias   <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, oc)
        nm <- if (!is.null(node$config$name)) node$config$name else node$id
        ggml_set_name(kernel, paste0(nm, "_kernel"))
        ggml_set_name(bias,   paste0(nm, "_bias"))
      }

      s0 <- node$config$strides[2]; s1 <- node$config$strides[1]
      if (node$config$padding == "same") {
        p0 <- as.integer(floor(kw / 2)); p1 <- as.integer(floor(kh / 2))
      } else {
        p0 <- 0L; p1 <- 0L
      }
      out <- ggml_conv_2d(ctx_compute, kernel, input_t,
                          s0 = s0, s1 = s1, p0 = p0, p1 = p1, d0 = 1L, d1 = 1L)
      bias_4d <- ggml_reshape_4d(ctx_compute, bias, 1L, 1L, oc, 1L)
      out <- ggml_add(ctx_compute, out, bias_4d)
      out <- nn_apply_activation(ctx_compute, out, node$config$activation)

      list(tensor = out, weights = list(kernel = kernel, bias = bias))
    },

    "max_pooling_2d" = {
      parent_id <- node$parents[[1]]$id
      input_t   <- built_tensors[[parent_id]]
      k0 <- node$config$pool_size[2]; k1 <- node$config$pool_size[1]
      s0 <- node$config$strides[2];   s1 <- node$config$strides[1]
      out <- ggml_pool_2d(ctx_compute, input_t, GGML_OP_POOL_MAX,
                          k0 = k0, k1 = k1, s0 = s0, s1 = s1, p0 = 0L, p1 = 0L)
      list(tensor = out, weights = list())
    },

    "global_max_pooling_2d" = ,
    "global_average_pooling_2d" = {
      parent_id <- node$parents[[1]]$id
      input_t   <- built_tensors[[parent_id]]
      sh <- ggml_tensor_shape(input_t)   # [W, H, C, N] (ggml order)
      W  <- sh[1]; H <- sh[2]; C <- sh[3]; N <- sh[4]
      pool_op <- if (node$node_type == "global_max_pooling_2d") {
        GGML_OP_POOL_MAX
      } else {
        GGML_OP_POOL_AVG
      }
      pooled <- ggml_pool_2d(ctx_compute, input_t, pool_op,
                              k0 = W, k1 = H, s0 = W, s1 = H,
                              p0 = 0L, p1 = 0L)
      out <- ggml_reshape_2d(ctx_compute, pooled, C, N)
      list(tensor = out, weights = list())
    },

    "conv_1d" = {
      parent_id <- node$parents[[1]]$id
      input_t   <- built_tensors[[parent_id]]
      psh       <- built_shapes[[parent_id]]  # c(L, C)
      k  <- node$config$kernel_size
      ic <- psh[2]
      oc <- node$config$filters

      if (!is.null(reuse_weights)) {
        kernel <- reuse_weights$kernel
        bias   <- reuse_weights$bias
      } else {
        kernel <- ggml_new_tensor_3d(ctx_weights, GGML_TYPE_F32, k, ic, oc)
        bias   <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, oc)
        nm <- if (!is.null(node$config$name)) node$config$name else node$id
        ggml_set_name(kernel, paste0(nm, "_kernel"))
        ggml_set_name(bias,   paste0(nm, "_bias"))
      }

      s0 <- node$config$strides
      p0 <- if (node$config$padding == "same") as.integer(floor(k / 2)) else 0L
      out <- ggml_conv_1d(ctx_compute, kernel, input_t, s0 = s0, p0 = p0, d0 = 1L)
      bias_3d <- ggml_reshape_3d(ctx_compute, bias, 1L, oc, 1L)
      out <- ggml_add(ctx_compute, out, bias_3d)
      out <- nn_apply_activation(ctx_compute, out, node$config$activation)

      list(tensor = out, weights = list(kernel = kernel, bias = bias))
    },

    "dropout" = {
      parent_id  <- node$parents[[1]]$id
      input_t    <- built_tensors[[parent_id]]
      stochastic <- isTRUE(node$config$stochastic)
      if (!training) {
        out <- input_t  # identity at inference
        list(tensor = out, weights = list())
      } else if (stochastic) {
        # Inverted dropout: input * mask * (1 / (1 - rate))
        # mask is a F32 tensor of 0/1 values, same shape as input_t
        psh   <- built_shapes[[parent_id]]
        ne    <- prod(psh)
        mask  <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, ne, batch_size)
        nm    <- if (!is.null(node$config$name)) node$config$name else node$id
        ggml_set_name(mask, paste0(nm, "_mask"))
        out <- ggml_mul(ctx_compute, input_t, mask)
        out <- ggml_scale(ctx_compute, out, 1.0 / (1.0 - node$config$rate))
        list(tensor = out, weights = list(mask = mask))
      } else {
        # Deterministic expected-value scaling
        out <- ggml_scale(ctx_compute, input_t, 1.0 - node$config$rate)
        list(tensor = out, weights = list())
      }
    },

    "embedding" = {
      parent_id  <- node$parents[[1]]$id
      input_t    <- built_tensors[[parent_id]]  # I32 [seq_len, N]
      vocab_size <- node$config$vocab_size
      dim        <- node$config$dim

      # Embedding table: [dim, vocab_size]
      if (!is.null(reuse_weights)) {
        E <- reuse_weights$weight
      } else {
        E  <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, dim, vocab_size)
        nm <- if (!is.null(node$config$name)) node$config$name else node$id
        ggml_set_name(E, paste0(nm, "_weight"))
      }

      # ggml_get_rows requires 1D index tensor
      # Flatten [seq_len, N] -> [seq_len * N], lookup -> [dim, seq_len*N]
      psh_in <- built_shapes[[parent_id]]
      seq_len <- if (length(psh_in) == 1L) psh_in else prod(psh_in)
      total   <- as.integer(seq_len * batch_size)
      idx_1d  <- ggml_reshape_1d(ctx_compute, input_t, total)
      flat    <- ggml_get_rows(ctx_compute, E, idx_1d)
      # Reshape to [dim, seq_len, N]
      out <- ggml_reshape_3d(ctx_compute, flat, dim, seq_len, batch_size)
      list(tensor = out, weights = list(weight = E))
    },

    "lstm" = {
      parent_id  <- node$parents[[1]]$id
      input_t    <- built_tensors[[parent_id]]
      psh        <- built_shapes[[parent_id]]  # c(seq_len, input_size)
      seq_len    <- psh[1]; input_sz <- psh[2]
      units      <- node$config$units
      nm         <- if (!is.null(node$config$name)) node$config$name else node$id

      if (!is.null(reuse_weights)) {
        W_gates <- reuse_weights$W_gates
        U_gates <- reuse_weights$U_gates
        b_gates <- reuse_weights$b_gates
        h0      <- reuse_weights$h0
        c0      <- reuse_weights$c0
      } else {
        W_gates <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, input_sz, 4L * units)
        U_gates <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, units,    4L * units)
        b_gates <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, 4L * units)
        h0      <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, units)
        c0      <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, units)
        ggml_set_name(W_gates, paste0(nm, "_W_gates"))
        ggml_set_name(U_gates, paste0(nm, "_U_gates"))
        ggml_set_name(b_gates, paste0(nm, "_b_gates"))
        ggml_set_name(h0,      paste0(nm, "_h0"))
        ggml_set_name(c0,      paste0(nm, "_c0"))
      }

      # Build input tensor [input_sz, seq_len, N] from parent [seq_len*input_sz, N]
      # Parent shape is c(seq_len, input_size) in R order -> ggml [input_sz, seq_len, N]
      input_3d <- if (length(ggml_tensor_shape(input_t)) == 2L) {
        ggml_reshape_3d(ctx_compute, input_t, input_sz, seq_len, batch_size)
      } else {
        input_t
      }

      act_cell <- node$config$activation
      act_rec  <- node$config$recurrent_activation
      # Use properly allocated zero tensors from ctx_weights to avoid uninitialized
      # memory in the compute context (NaN * 0 = NaN under IEEE 754).
      h_shape <- ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, units, batch_size)
      c_shape <- ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, units, batch_size)
      h_t <- ggml_repeat(ctx_compute, h0, h_shape)
      c_t <- ggml_repeat(ctx_compute, c0, c_shape)
      h_steps <- vector("list", seq_len)

      for (t in seq_len(seq_len)) {
        offset_t <- as.integer((t - 1L) * input_sz * 4L)
        x_t <- ggml_view_2d(ctx_compute, input_3d, input_sz, batch_size,
                              nb1 = as.integer(input_sz * seq_len * 4L),
                              offset = offset_t)
        step <- nn_lstm_step(ctx_compute, x_t, h_t, c_t,
                              W_gates, U_gates, b_gates,
                              units, act_cell, act_rec)
        h_t <- step$h
        c_t <- step$c
        h_steps[[t]] <- h_t
      }

      if (isTRUE(node$config$return_sequences)) {
        out <- h_steps[[1]]
        for (t in seq(2L, seq_len)) out <- ggml_concat(ctx_compute, out, h_steps[[t]], dim = 1L)
      } else {
        out <- h_t
      }

      list(tensor = out,
           weights = list(W_gates = W_gates, U_gates = U_gates, b_gates = b_gates,
                          h0 = h0, c0 = c0))
    },

    "gru" = {
      parent_id  <- node$parents[[1]]$id
      input_t    <- built_tensors[[parent_id]]
      psh        <- built_shapes[[parent_id]]
      seq_len    <- psh[1]; input_sz <- psh[2]
      units      <- node$config$units
      nm         <- if (!is.null(node$config$name)) node$config$name else node$id

      if (!is.null(reuse_weights)) {
        W_zh <- reuse_weights$W_zh; U_zh <- reuse_weights$U_zh; b_zh <- reuse_weights$b_zh
        W_n  <- reuse_weights$W_n;  U_n  <- reuse_weights$U_n;  b_n  <- reuse_weights$b_n
        h0   <- reuse_weights$h0
      } else {
        W_zh <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, input_sz, 2L * units)
        U_zh <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, units,    2L * units)
        b_zh <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, 2L * units)
        W_n  <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, input_sz, units)
        U_n  <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, units,    units)
        b_n  <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, units)
        h0   <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, units)
        ggml_set_name(W_zh, paste0(nm, "_W_zh")); ggml_set_name(U_zh, paste0(nm, "_U_zh"))
        ggml_set_name(b_zh, paste0(nm, "_b_zh")); ggml_set_name(W_n,  paste0(nm, "_W_n"))
        ggml_set_name(U_n,  paste0(nm, "_U_n"));  ggml_set_name(b_n,  paste0(nm, "_b_n"))
        ggml_set_name(h0,   paste0(nm, "_h0"))
      }

      input_3d <- if (length(ggml_tensor_shape(input_t)) == 2L) {
        ggml_reshape_3d(ctx_compute, input_t, input_sz, seq_len, batch_size)
      } else {
        input_t
      }

      act_cell <- node$config$activation
      act_rec  <- node$config$recurrent_activation
      h_shape <- ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, units, batch_size)
      h_t <- ggml_repeat(ctx_compute, h0, h_shape)
      h_steps <- vector("list", seq_len)

      for (t in seq_len(seq_len)) {
        offset_t <- as.integer((t - 1L) * input_sz * 4L)
        x_t <- ggml_view_2d(ctx_compute, input_3d, input_sz, batch_size,
                              nb1 = as.integer(input_sz * seq_len * 4L),
                              offset = offset_t)
        step <- nn_gru_step(ctx_compute, x_t, h_t, W_zh, U_zh, b_zh,
                             W_n, U_n, b_n, units, act_cell, act_rec)
        h_t <- step$h
        h_steps[[t]] <- h_t
      }

      if (isTRUE(node$config$return_sequences)) {
        out <- h_steps[[1]]
        for (t in seq(2L, seq_len)) out <- ggml_concat(ctx_compute, out, h_steps[[t]], dim = 1L)
      } else {
        out <- h_t
      }

      list(tensor = out,
           weights = list(W_zh = W_zh, U_zh = U_zh, b_zh = b_zh,
                          W_n = W_n, U_n = U_n, b_n = b_n, h0 = h0))
    },

    stop("Unknown node_type in graph build: ", node$node_type)
  )
}

#' Build ggml computation graph for a functional model
#' @param model A ggml_functional_model
#' @param batch_size Integer batch size
#' @param training Logical; TRUE during fit (activates dropout scaling), FALSE
#'   during evaluate/predict (dropout becomes identity).
#' @return Named list with inputs, outputs, ctx_weights, ctx_compute, buffer, node_weights
#' @keywords internal
# logits_output: see the note on nn_build_graph() in nn_model.R.
# ggml_cross_entropy_loss() applies log_softmax internally, so a model ending in
# a softmax activation would have it applied twice during cross-entropy
# training. Set for fit(), left FALSE for inference.
nn_build_functional_graph <- function(model, batch_size, training = FALSE,
                                      logits_output = FALSE) {
  backend      <- model$compilation$backend
  saved_weights <- model$node_weights  # NULL before first fit, list after
  # R-vector weights from ggml_load_model (node_id -> named list of numeric)
  saved_weights_data <- model$node_weights_data

  # Strip a final softmax from the output nodes when logits are wanted. The
  # nodes are R lists, so editing the local `model` copy leaves the caller's
  # model definition (and therefore inference) untouched. Node identity is by
  # $id, which is preserved.
  #
  # logits_output may be a single flag for every output, or one flag per output
  # head: a multi-output model can mix a cross-entropy head (needs logits) with
  # an MSE head (must keep its activation), so the softmax is stripped per head.
  if (any(logits_output)) {
    lo <- if (length(logits_output) == 1L) {
      rep(as.logical(logits_output), length(model$outputs))
    } else {
      as.logical(logits_output)
    }
    if (length(lo) != length(model$outputs)) {
      stop("'logits_output' must be length 1 or one flag per output (",
           length(logits_output), " given, ", length(model$outputs),
           " outputs).", call. = FALSE)
    }
    model$outputs <- lapply(seq_along(model$outputs), function(i) {
      n <- model$outputs[[i]]
      if (lo[[i]] && identical(n$config$activation, "softmax")) {
        n$config$activation <- NULL
      }
      n
    })
  }

  # Topological sort -- inputs first, outputs last
  nodes_sorted <- nn_topo_sort(model$outputs)

  # ---- Memory estimation ----
  total_elements <- 0L
  shapes <- list()  # node_id -> R-order shape

  # First pass: compute shapes
  weight_elements <- 0    # double: a wide dense layer overflows integer here
  for (node in nodes_sorted) {
    parent_shapes <- lapply(node$parents, function(p) shapes[[p$id]])
    out_shape <- nn_functional_output_shape(node, parent_shapes)
    shapes[[node$id]] <- out_shape
    total_elements <- total_elements + prod(out_shape) * batch_size
    weight_elements <- weight_elements +
      nn_functional_weight_elements(node, parent_shapes)
  }

  # ctx_weights holds the weights, so its size has to be estimated from the
  # weights. Sizing it from the activations instead makes it depend on
  # batch_size, which the weights do not, and a small batch then starves a
  # layer whose kernel is large -- a 1024-unit dense layer on a 1536-wide input
  # needs 6 MB of kernel however few rows are pushed through it.
  mem_size <- max((total_elements + weight_elements + 1000) * 4 +
                    length(nodes_sorted) * 2048,
                  2 * 1024 * 1024)

  ctx_weights <- ggml_init(mem_size, no_alloc = TRUE)

  compute_mem <- max(64L * 1024L * 1024L,
                     total_elements * 4L * 20L)
  ctx_compute <- ggml_init(compute_mem, no_alloc = TRUE)

  # ---- Second pass: build tensors ----
  built_tensors      <- list()   # node_id -> ggml tensor (external pointer)
  node_weights       <- list()   # node_id -> list of weight tensors
  # Shared-layer cache keyed by layer_id (object identity from ggml_apply()).
  # Nodes without layer_id (created via ggml_layer_*() pipe style) are never
  # shared -- they each allocate their own weights.
  shared_weight_cache <- list()  # layer_id -> weight list

  for (node in nodes_sorted) {
    layer_id     <- node$layer_id  # NULL for non-shared nodes
    is_shareable <- !is.null(layer_id) &&
                    node$node_type %in% c("dense", "batch_norm",
                                          "conv_2d", "conv_1d", "embedding",
                                          "lstm", "gru", "attention")

    reuse_w <- if (is_shareable && !is.null(shared_weight_cache[[layer_id]])) {
      shared_weight_cache[[layer_id]]
    } else {
      NULL
    }

    result <- nn_build_functional_node(
      node, built_tensors, shapes, ctx_weights, ctx_compute, batch_size,
      training = training,
      reuse_weights = reuse_w
    )
    built_tensors[[node$id]] <- result$tensor
    node_weights[[node$id]]  <- result$weights

    # Cache weights for first occurrence of a shared layer
    if (is_shareable && is.null(shared_weight_cache[[layer_id]]) &&
        length(result$weights) > 0L) {
      shared_weight_cache[[layer_id]] <- result$weights
    }
  }

  # Allocate weights on backend
  buffer <- ggml_backend_alloc_ctx_tensors(ctx_weights, backend)

  # ---- Initialize weights ----
  # Track which layer_ids have already been initialized so that secondary
  # applications of a shared layer skip init and ggml_set_param.
  initialized_layer_ids <- character(0L)

  frozen_nodes <- if (!is.null(model$frozen_nodes)) model$frozen_nodes else list()

  for (node in nodes_sorted) {
    w <- node_weights[[node$id]]
    # frozen_nodes override takes priority over node$trainable
    trainable <- if (!is.null(frozen_nodes[[node$id]])) {
      isTRUE(frozen_nodes[[node$id]])
    } else if (is.null(node$trainable)) {
      TRUE
    } else {
      isTRUE(node$trainable)
    }

    # R-vector weights from ggml_load_model -- always checked first to avoid
    # any risk of accessing freed ggml tensor pointers on a loaded model.
    swd <- if (!is.null(saved_weights_data)) saved_weights_data[[node$id]] else NULL

    # Saved ggml-tensor weights from a previous fit (keyed by node$id).
    sw <- if (is.null(swd) && !is.null(saved_weights)) saved_weights[[node$id]] else NULL

    # Skip init for secondary applications of a shared layer (by layer_id).
    layer_id     <- node$layer_id
    is_shareable <- !is.null(layer_id) &&
                    node$node_type %in% c("dense", "batch_norm",
                                          "conv_2d", "conv_1d", "embedding",
                                          "lstm", "gru", "attention")
    if (is_shareable && layer_id %in% initialized_layer_ids) {
      next  # weights already initialized and params set by primary occurrence
    }

    if (node$node_type == "dense") {
      psh    <- shapes[[node$parents[[1]]$id]]
      fan_in <- nn_dense_fan_in(node, psh)
      fan_out <- node$config$units

      if (!is.null(sw$weight)) {
        ggml_backend_tensor_set_data(w$weight, ggml_backend_tensor_get_data(sw$weight))
        ggml_backend_tensor_set_data(w$bias,   ggml_backend_tensor_get_data(sw$bias))
      } else if (!is.null(swd$weight)) {
        ggml_backend_tensor_set_data(w$weight, swd$weight)
        ggml_backend_tensor_set_data(w$bias,   swd$bias)
      } else {
        nn_init_glorot_uniform(w$weight, fan_in, fan_out)
        nn_init_zeros(w$bias)
      }
      if (trainable) {
        ggml_set_param(w$weight)
        ggml_set_param(w$bias)
      }
      if (is_shareable) initialized_layer_ids <- c(initialized_layer_ids, layer_id)

    } else if (node$node_type == "batch_norm") {
      if (!is.null(sw$gamma)) {
        ggml_backend_tensor_set_data(w$gamma, ggml_backend_tensor_get_data(sw$gamma))
        ggml_backend_tensor_set_data(w$beta,  ggml_backend_tensor_get_data(sw$beta))
      } else if (!is.null(swd$gamma)) {
        ggml_backend_tensor_set_data(w$gamma, swd$gamma)
        ggml_backend_tensor_set_data(w$beta,  swd$beta)
      } else {
        n <- ggml_nelements(w$gamma)
        ggml_backend_tensor_set_data(w$gamma, rep(1.0, n))
        nn_init_zeros(w$beta)
      }
      if (trainable) {
        ggml_set_param(w$gamma)
        ggml_set_param(w$beta)
      }
      if (is_shareable) initialized_layer_ids <- c(initialized_layer_ids, layer_id)

    } else if (node$node_type == "conv_2d") {
      psh    <- shapes[[node$parents[[1]]$id]]
      kh <- node$config$kernel_size[1]; kw <- node$config$kernel_size[2]
      fan_in <- kh * kw * psh[3]

      if (!is.null(sw$kernel)) {
        ggml_backend_tensor_set_data(w$kernel, ggml_backend_tensor_get_data(sw$kernel))
        ggml_backend_tensor_set_data(w$bias,   ggml_backend_tensor_get_data(sw$bias))
      } else if (!is.null(swd$kernel)) {
        ggml_backend_tensor_set_data(w$kernel, swd$kernel)
        ggml_backend_tensor_set_data(w$bias,   swd$bias)
      } else {
        nn_init_he_uniform(w$kernel, fan_in)
        nn_init_zeros(w$bias)
      }
      if (trainable) {
        ggml_set_param(w$kernel)
        ggml_set_param(w$bias)
      }
      if (is_shareable) initialized_layer_ids <- c(initialized_layer_ids, layer_id)

    } else if (node$node_type == "conv_1d") {
      psh    <- shapes[[node$parents[[1]]$id]]
      fan_in <- node$config$kernel_size * psh[2]

      if (!is.null(sw$kernel)) {
        ggml_backend_tensor_set_data(w$kernel, ggml_backend_tensor_get_data(sw$kernel))
        ggml_backend_tensor_set_data(w$bias,   ggml_backend_tensor_get_data(sw$bias))
      } else if (!is.null(swd$kernel)) {
        ggml_backend_tensor_set_data(w$kernel, swd$kernel)
        ggml_backend_tensor_set_data(w$bias,   swd$bias)
      } else {
        nn_init_he_uniform(w$kernel, fan_in)
        nn_init_zeros(w$bias)
      }
      if (trainable) {
        ggml_set_param(w$kernel)
        ggml_set_param(w$bias)
      }
      if (is_shareable) initialized_layer_ids <- c(initialized_layer_ids, layer_id)

    } else if (node$node_type == "embedding") {
      if (!is.null(sw$weight)) {
        ggml_backend_tensor_set_data(w$weight, ggml_backend_tensor_get_data(sw$weight))
      } else if (!is.null(swd$weight)) {
        ggml_backend_tensor_set_data(w$weight, swd$weight)
      } else {
        n <- ggml_nelements(w$weight)
        ggml_backend_tensor_set_data(w$weight, runif(n, -0.05, 0.05))
      }
      if (trainable) {
        ggml_set_param(w$weight)
      }
      if (is_shareable) initialized_layer_ids <- c(initialized_layer_ids, layer_id)

    } else if (node$node_type == "lstm") {
      psh      <- shapes[[node$parents[[1]]$id]]
      input_sz <- psh[2]; units <- node$config$units
      if (!is.null(sw$W_gates)) {
        ggml_backend_tensor_set_data(w$W_gates, ggml_backend_tensor_get_data(sw$W_gates))
        ggml_backend_tensor_set_data(w$U_gates, ggml_backend_tensor_get_data(sw$U_gates))
        ggml_backend_tensor_set_data(w$b_gates, ggml_backend_tensor_get_data(sw$b_gates))
      } else if (!is.null(swd$W_gates)) {
        ggml_backend_tensor_set_data(w$W_gates, swd$W_gates)
        ggml_backend_tensor_set_data(w$U_gates, swd$U_gates)
        ggml_backend_tensor_set_data(w$b_gates, swd$b_gates)
      } else {
        nn_init_recurrent_uniform(w$W_gates)
        nn_init_recurrent_uniform(w$U_gates)
        nn_init_zeros(w$b_gates)
      }
      nn_init_zeros(w$h0)
      nn_init_zeros(w$c0)
      if (trainable) {
        ggml_set_param(w$W_gates); ggml_set_param(w$U_gates); ggml_set_param(w$b_gates)
      }
      if (is_shareable) initialized_layer_ids <- c(initialized_layer_ids, layer_id)

    } else if (node$node_type == "gru") {
      psh      <- shapes[[node$parents[[1]]$id]]
      input_sz <- psh[2]; units <- node$config$units
      if (!is.null(sw$W_zh)) {
        ggml_backend_tensor_set_data(w$W_zh, ggml_backend_tensor_get_data(sw$W_zh))
        ggml_backend_tensor_set_data(w$U_zh, ggml_backend_tensor_get_data(sw$U_zh))
        ggml_backend_tensor_set_data(w$b_zh, ggml_backend_tensor_get_data(sw$b_zh))
        ggml_backend_tensor_set_data(w$W_n,  ggml_backend_tensor_get_data(sw$W_n))
        ggml_backend_tensor_set_data(w$U_n,  ggml_backend_tensor_get_data(sw$U_n))
        ggml_backend_tensor_set_data(w$b_n,  ggml_backend_tensor_get_data(sw$b_n))
      } else if (!is.null(swd$W_zh)) {
        ggml_backend_tensor_set_data(w$W_zh, swd$W_zh); ggml_backend_tensor_set_data(w$U_zh, swd$U_zh)
        ggml_backend_tensor_set_data(w$b_zh, swd$b_zh); ggml_backend_tensor_set_data(w$W_n,  swd$W_n)
        ggml_backend_tensor_set_data(w$U_n,  swd$U_n);  ggml_backend_tensor_set_data(w$b_n,  swd$b_n)
      } else {
        nn_init_recurrent_uniform(w$W_zh)
        nn_init_recurrent_uniform(w$U_zh)
        nn_init_zeros(w$b_zh)
        nn_init_recurrent_uniform(w$W_n)
        nn_init_recurrent_uniform(w$U_n)
        nn_init_zeros(w$b_n)
      }
      nn_init_zeros(w$h0)
      if (trainable) {
        ggml_set_param(w$W_zh); ggml_set_param(w$U_zh); ggml_set_param(w$b_zh)
        ggml_set_param(w$W_n);  ggml_set_param(w$U_n);  ggml_set_param(w$b_n)
      }
      if (is_shareable) initialized_layer_ids <- c(initialized_layer_ids, layer_id)

    } else if (node$node_type == "attention") {
      d_model <- node$config$d_model
      if (!is.null(sw$W_q)) {
        for (k in c("W_q", "W_k", "W_v", "W_o")) {
          ggml_backend_tensor_set_data(w[[k]], ggml_backend_tensor_get_data(sw[[k]]))
        }
        if (!is.null(w$b_o) && !is.null(sw$b_o)) {
          ggml_backend_tensor_set_data(w$b_o, ggml_backend_tensor_get_data(sw$b_o))
        }
      } else if (!is.null(swd$W_q)) {
        for (k in c("W_q", "W_k", "W_v", "W_o")) {
          ggml_backend_tensor_set_data(w[[k]], swd[[k]])
        }
        if (!is.null(w$b_o) && !is.null(swd$b_o)) {
          ggml_backend_tensor_set_data(w$b_o, swd$b_o)
        }
      } else {
        # Every projection is square, so fan_in and fan_out are both d_model.
        for (k in c("W_q", "W_k", "W_v", "W_o")) {
          nn_init_glorot_uniform(w[[k]], d_model, d_model)
        }
        if (!is.null(w$b_o)) nn_init_zeros(w$b_o)
      }
      if (trainable) {
        for (k in c("W_q", "W_k", "W_v", "W_o")) ggml_set_param(w[[k]])
        if (!is.null(w$b_o)) ggml_set_param(w$b_o)
      }
      if (is_shareable) initialized_layer_ids <- c(initialized_layer_ids, layer_id)

    } else if (node$node_type == "dropout" && !is.null(w$mask)) {
      # Stochastic dropout: initialize mask to all-ones (identity until first epoch update)
      n <- ggml_nelements(w$mask)
      ggml_backend_tensor_set_data(w$mask, rep(1.0, n))
      # mask is NOT a param -- not trained, updated externally each epoch
    }
    # input / flatten / add / concatenate / custom / max_pooling_2d / det.dropout
    # have no weights
  }

  # Collect input/output ggml tensors (always lists)
  input_tensors  <- lapply(model$inputs,  function(n) built_tensors[[n$id]])
  output_tensors <- lapply(model$outputs, function(n) built_tensors[[n$id]])

  # Mark outputs
  for (t in output_tensors) ggml_set_output(t)

  # Collect stochastic dropout masks (node_id -> mask tensor)
  dropout_masks <- list()
  for (node in nodes_sorted) {
    if (node$node_type == "dropout" && isTRUE(node$config$stochastic)) {
      w <- node_weights[[node$id]]
      if (!is.null(w$mask)) {
        dropout_masks[[node$id]] <- list(
          mask = w$mask,
          rate = node$config$rate,
          ne   = ggml_nelements(w$mask)
        )
      }
    }
  }

  list(
    ctx_weights   = ctx_weights,
    ctx_compute   = ctx_compute,
    inputs        = input_tensors,
    outputs       = output_tensors,
    buffer        = buffer,
    node_weights  = node_weights,
    built_tensors = built_tensors,
    shapes        = shapes,
    dropout_masks = dropout_masks
  )
}

# ============================================================================
# Compile -- S3 method for ggml_functional_model
# ============================================================================

#' @rdname ggml_compile
#' @export
ggml_compile.ggml_functional_model <- function(model,
                                                optimizer = "adam",
                                                loss = "categorical_crossentropy",
                                                metrics = c("accuracy"),
                                                backend = "auto",
                                                loss_weights = NULL) {
  nn_validate_compilation(optimizer, loss, metrics)

  # Resolve loss/loss_weights against the output heads now, so a mismatch is
  # reported at compile time rather than part-way into training.
  output_names <- nn_output_names(model)
  loss_spec    <- nn_resolve_losses(loss, loss_weights, output_names)

  # Backend selection (same logic as Sequential)
  use_vulkan <- FALSE
  if (backend == "auto") {
    if (ggml_vulkan_available() && ggml_vulkan_device_count() > 0) use_vulkan <- TRUE
  } else if (backend == "vulkan") {
    if (!ggml_vulkan_available() || ggml_vulkan_device_count() == 0) {
      stop("Vulkan backend requested but not available.")
    }
    use_vulkan <- TRUE
  } else if (backend != "cpu") {
    stop("Unknown backend: '", backend, "'. Use 'auto', 'cpu', or 'vulkan'.")
  }

  # The scheduler's hash set has to be big enough for the graph it will be
  # handed, and the graph is not built yet -- but the layers that will build it
  # are. A recurrent layer unrolls one block of nodes per timestep, so a
  # 96-step LSTM alone outgrows the 2048 default and aborts inside ggml with an
  # assert on hash_set.size. Estimating from the layers keeps that from being a
  # fixed ceiling on sequence length.
  sched_size <- nn_functional_graph_size(model)

  if (use_vulkan) {
    gpu_backend <- ggml_vulkan_init(0L)
    sched       <- ggml_backend_sched_new(list(gpu_backend), parallel = FALSE,
                                          graph_size = sched_size)
    cpu_backend <- ggml_backend_cpu_init()
    if (!isTRUE(.ggmlr_state$backend_msg_shown)) {
      message("Using Vulkan GPU backend: ", ggml_vulkan_device_description(0L))
      .ggmlr_state$backend_msg_shown <- TRUE
    }
  } else {
    cpu_backend <- ggml_backend_cpu_init()
    sched       <- ggml_backend_sched_new(list(cpu_backend), parallel = FALSE,
                                          graph_size = sched_size)
    if (!isTRUE(.ggmlr_state$backend_msg_shown)) {
      message("Using CPU backend")
      .ggmlr_state$backend_msg_shown <- TRUE
    }
  }

  if (use_vulkan) {
    model$compilation$backend     <- gpu_backend
    model$compilation$cpu_backend <- cpu_backend
  } else {
    model$compilation$backend <- cpu_backend
  }

  model$compilation$sched        <- sched
  model$compilation$optimizer    <- optimizer
  model$compilation$loss         <- loss
  model$compilation$metrics      <- metrics
  # Per-head resolution of loss/loss_weights (one entry per output).
  model$compilation$loss_spec    <- loss_spec
  model$compilation$loss_weights <- loss_weights
  model$compilation$output_names <- output_names
  # Record requested vs actually-used backend so a silent "auto" -> CPU
  # fallback is inspectable later (see ggml_model_backend()).
  model$compilation$backend_requested <- backend
  model$compilation$backend_used      <- if (use_vulkan) "vulkan" else "cpu"
  model$compilation$device <- if (use_vulkan) {
    ggml_vulkan_device_description(0L)
  } else "cpu"
  model$compiled              <- TRUE

  invisible(model)
}

# ============================================================================
# Multi-input helpers
# ============================================================================

# Normalise x for multi-input models.
# Returns a list: list(x_ggml, ne_per_input, is_multi)
#   x_ggml       : numeric vector, column-major, [ne_total * N] for dataset
#   ne_per_input : integer vector, one element per input node
#   is_multi     : TRUE when model has >1 inputs
#
# For single-input models x may be matrix [N, ne] or array [N, ...].
# For multi-input models x must be list(x1, x2, ...) where each xi is
# a matrix [N, ne_i].  All xi must have the same nrow.
nn_prepare_x <- function(model, x) {
  n_inputs <- length(model$inputs)
  if (n_inputs == 1L) {
    shape <- model$inputs[[1L]]$config$shape
    ne    <- prod(shape)
    dtype <- if (!is.null(model$inputs[[1L]]$config$dtype)) model$inputs[[1L]]$config$dtype else "float32"
    if (dtype == "int32") {
      x_ggml <- as.integer(t(x))
    } else {
      x_ggml <- if (length(shape) == 3L) as.vector(aperm(x, c(3L, 2L, 4L, 1L)))
                else if (length(shape) == 2L) as.vector(aperm(x, c(3L, 2L, 1L)))
                else as.vector(t(x))
    }
    return(list(x_ggml = x_ggml, ne_per_input = as.integer(ne), is_multi = FALSE))
  }

  # Multi-input
  if (!is.list(x) || is.data.frame(x))
    stop("For multi-input models x must be a list: list(x1, x2, ...)")
  if (length(x) != n_inputs)
    stop("x has ", length(x), " elements but model has ", n_inputs, " inputs.")

  ne_per_input <- vapply(model$inputs, function(inp) as.integer(prod(inp$config$shape)), integer(1))
  # Each xi must be a matrix [N, ne_i].  t(xi) is [ne_i, N] (column-major),
  # which matches the layout ggml_backend_tensor_set_data expects for a
  # [ne_i, batch] tensor.
  N <- nrow(as.matrix(x[[1]]))
  # cbind of transposed mats -> [ne_total, N], then as.vector = column-major
  x_ggml <- as.numeric(do.call(rbind, lapply(seq_len(n_inputs), function(i) {
    xi  <- x[[i]]
    ne_i <- ne_per_input[i]
    xi_mat <- matrix(as.numeric(xi), nrow = N, ncol = ne_i)
    t(xi_mat)   # [ne_i, N]
  })))   # result: [ne_total, N] column-major = ne_total * N values
  list(x_ggml = x_ggml, ne_per_input = ne_per_input, is_multi = TRUE)
}

# Fill each ggml input tensor for one batch from the full interleaved vector.
# x_ggml: full flat vector [ne_total * N] in interleaved sample layout (see nn_prepare_x)
# ne_per_input: elements per sample for each input
# input_tensors: list of ggml tensor pointers (one per input)
# batch_size: number of samples in this batch
# samp_start: 0-based index of first sample in this batch
nn_fill_inputs <- function(x_ggml, ne_per_input, input_tensors, batch_size, samp_start) {
  nn_fill_inputs_idx(x_ggml, ne_per_input, input_tensors,
                     samp_start + seq_len(batch_size) - 1L)
}

# As nn_fill_inputs(), but taking explicit 0-based sample indices instead of a
# contiguous run. The multi-input path has no ggml_opt_dataset to permute, so
# shuffling there is done by handing this function a permuted index vector.
nn_fill_inputs_idx <- function(x_ggml, ne_per_input, input_tensors, samp_idx) {
  ne_total <- sum(ne_per_input)
  for (i in seq_along(input_tensors)) {
    ne_i <- ne_per_input[i]
    # offsets of this input's block within each sample's interleaved row
    inp_offset <- sum(ne_per_input[seq_len(i - 1L)])
    # collect ne_i values for each requested sample
    chunk <- unlist(lapply(samp_idx, function(s) {
      base <- s * ne_total + inp_offset
      x_ggml[(base + 1L):(base + ne_i)]
    }), use.names = FALSE)
    ggml_backend_tensor_set_data(input_tensors[[i]], chunk)
  }
}

# Gather the label rows for a set of 0-based sample indices.
nn_gather_labels <- function(y_ggml, ne_label, samp_idx) {
  unlist(lapply(samp_idx, function(s) {
    y_ggml[(s * ne_label + 1L):((s + 1L) * ne_label)]
  }), use.names = FALSE)
}

# As nn_gather_labels(), but taking only one head's columns out of each label
# row. On the multi-output path `y` is the heads concatenated column-wise, so a
# head owns the `width` columns starting at `off` (0-based); each head has its
# own labels tensor and is filled from its own slice. `off = 0` with
# `width = ne_label` reduces to nn_gather_labels().
nn_gather_labels_head <- function(y_ggml, ne_label, samp_idx, off, width) {
  if (off == 0 && width == ne_label) {
    return(nn_gather_labels(y_ggml, ne_label, samp_idx))
  }
  unlist(lapply(samp_idx, function(s) {
    base <- s * ne_label + off
    y_ggml[(base + 1L):(base + width)]
  }), use.names = FALSE)
}

# Fill every head's labels tensor for one batch of 0-based sample indices.
# Each head owns `widths[i]` columns of a label row starting at `offs[i]`, so a
# single-head model (offs = 0, widths = ne_label) writes the whole row into the
# one tensor. Every loss type reachable from ggml_compile() -- MSE and cross
# entropy -- has labels, so a NULL tensor here would mean the context was built
# with fewer heads than the model has, which is a bug rather than a case to skip.
nn_fill_labels_idx <- function(y_ggml, ne_label, labels_tensors, offs, widths, samp_idx) {
  for (i in seq_along(labels_tensors)) {
    ggml_backend_tensor_set_data(
      labels_tensors[[i]],
      nn_gather_labels_head(y_ggml, ne_label, samp_idx, offs[[i]], widths[[i]]))
  }
}

# ============================================================================
# Fit -- S3 method for ggml_functional_model
# ============================================================================

#' @rdname ggml_fit
#' @param model A compiled model object.
#' @param x Training data (matrix or array).
#' @param y Training labels (matrix, one-hot encoded).
#' @param epochs Number of training epochs (default: 1).
#' @param batch_size Batch size (default: 32).
#' @param validation_split Fraction of data for validation (default: 0).
#' @param validation_data Optional list(x_val, y_val). Overrides validation_split.
#' @param verbose 0 = silent, 1 = progress (default: 1).
#' @param shuffle Shuffle the data (default \code{TRUE}).  The dataset is
#'   shuffled once before the train/validation split, then the training portion
#'   is reshuffled each epoch while the validation portion stays fixed.  Set to
#'   \code{FALSE} for time series or exactly reproducible runs.
#' @param callbacks List of callback objects, e.g.
#'   \code{\link{ggml_callback_early_stopping}} or an LR scheduler.  Each is a
#'   list with \code{on_epoch_begin(epoch, logs, state)} and/or
#'   \code{on_epoch_end(epoch, logs, state)}; setting \code{state$stop <- TRUE}
#'   stops training.  \code{logs} holds \code{train_loss},
#'   \code{train_accuracy}, \code{val_loss} and \code{val_accuracy}, and for a
#'   multi-output model also \code{train_<output>_loss},
#'   \code{train_<output>_accuracy}, \code{val_<output>_loss} and
#'   \code{val_<output>_accuracy}, so a callback can monitor one head rather
#'   than the total.  These are the same names \code{model$history} uses, so a
#'   \code{monitor=} naming one of them matches a column there.
#'
#'   Both phases carry a prefix on purpose: it keeps an output named
#'   \code{val_x} from colliding with another output's validation key.
#'   \code{\link{ggml_evaluate}} reports the same per-head quantities under the
#'   bare \code{<output>_loss}, since a single evaluation has no train/val
#'   phases to tell apart.
#' @param ... Additional arguments (ignored).
#' @export
ggml_fit.ggml_functional_model <- function(model, x, y,
                                            epochs = 1L,
                                            batch_size = 32L,
                                            validation_split = 0.0,
                                            validation_data = NULL,
                                            verbose = 1L,
                                            shuffle = TRUE,
                                            callbacks = list(),
                                            ...) {
  if (!model$compiled) {
    stop("Model must be compiled before training. Call ggml_compile() first.")
  }

  # Prepare input data (handles both single and multi-input)
  xp <- nn_prepare_x(model, x)
  is_multi      <- xp$is_multi
  x_ggml        <- xp$x_ggml
  ne_per_input  <- xp$ne_per_input
  ne_datapoint  <- sum(ne_per_input)   # total elements per sample across all inputs

  # Multi-output: `y` arrives as a list of matrices, one per head. They are
  # concatenated column-wise into the single label matrix the dataset holds;
  # each head's slice is then addressed by an offset (see ggml_opt_fit_multi).
  loss_spec <- model$compilation$loss_spec
  n_head    <- if (is.null(loss_spec)) 1L else length(loss_spec)

  # A plain matrix `y` on a multi-output model keeps the historical meaning:
  # only the last output is a trained head, the earlier ones are intermediate
  # activations exposed for inspection through ggml_predict(). Training every
  # head requires one `y` per head, i.e. a list.
  if (n_head > 1L && !is.list(y)) {
    loss_spec <- loss_spec[n_head]
    n_head    <- 1L
    trained_output_idx <- length(model$outputs)

    # `y` describes the trained head only, so its width has to match that head.
    # Without this the labels tensor and the loss node disagree and ggml aborts
    # the process from ggml_opt_dataset_get_batch instead of reporting an error.
    trained_units <- tryCatch(
      nn_functional_output_shape(model$outputs[[trained_output_idx]],
                                 lapply(model$outputs[[trained_output_idx]]$parents,
                                        function(p) NULL)),
      error = function(e) NULL)
    if (!is.null(trained_units) && length(trained_units) == 1L &&
        !is.na(trained_units) && ncol(y) != trained_units) {
      stop("'y' has ", ncol(y), " column(s) but the trained output ('",
           nn_output_names(model)[[trained_output_idx]], "') has ",
           trained_units, ". With a plain matrix 'y' only the last output is ",
           "trained; pass a list with one 'y' per output to train them all.",
           call. = FALSE)
    }
  } else {
    trained_output_idx <- NULL
  }

  if (n_head > 1L) {
    if (length(y) != n_head) {
      stop("'y' must have one entry per output (", length(y), " given, ",
           n_head, " expected).", call. = FALSE)
    }
    ynames <- names(y)
    if (!is.null(ynames) && !any(ynames == "")) {
      onames  <- vapply(loss_spec, function(s) s$name, character(1))
      unknown <- setdiff(ynames, onames)
      if (length(unknown) > 0L) {
        stop("'y' names do not match model outputs: ",
             paste(unknown, collapse = ", "), ". Model outputs: ",
             paste(onames, collapse = ", "), call. = FALSE)
      }
      y <- y[match(onames, ynames)]
    }
    y <- lapply(y, function(yi) if (is.matrix(yi)) yi else as.matrix(yi))
    nrows <- vapply(y, nrow, integer(1))
    if (length(unique(nrows)) != 1L) {
      stop("All entries of 'y' must have the same number of rows (got ",
           paste(nrows, collapse = ", "), ").", call. = FALSE)
    }
    head_widths <- vapply(y, ncol, integer(1))
    y <- do.call(cbind, y)
  } else {
    head_widths <- NULL
  }

  # Handle validation_data
  # Shuffling before the split is only safe when the split is a fraction we
  # choose; an explicit validation_data set is positional and must stay put.
  shuffle_all <- shuffle

  if (!is.null(validation_data)) {
    if (!is.list(validation_data) || length(validation_data) < 2L) {
      stop("validation_data must be a list: list(x_val, y_val)")
    }
    x_val <- validation_data[[1]]
    y_val <- validation_data[[2]]
    xp_val <- nn_prepare_x(model, x_val)
    n_val   <- length(xp_val$x_ggml) %/% ne_datapoint
    n_train <- length(x_ggml)         %/% ne_datapoint
    x_ggml  <- c(x_ggml, xp_val$x_ggml)
    y       <- rbind(y, y_val)
    validation_split <- n_val / (n_train + n_val)
    # The split is positional: these rows ARE the user's validation set. A
    # pre-split shuffle would mix them back into training, so it is suppressed
    # here. Per-epoch shuffling of the training portion is unaffected.
    shuffle_all <- FALSE
  }

  n_samples <- length(x_ggml) %/% ne_datapoint
  ne_label  <- ncol(y)

  # Truncate to batch boundary
  usable <- (n_samples %/% batch_size) * batch_size
  if (usable < n_samples) {
    if (verbose > 0) {
      message("Truncating data from ", n_samples, " to ", usable,
              " samples (batch_size=", batch_size, " must divide evenly)")
    }
    keep_idx <- seq_len(usable * ne_datapoint)
    x_ggml   <- x_ggml[keep_idx]
    y        <- y[seq_len(usable), , drop = FALSE]
    n_samples <- usable
  }

  y_ggml <- as.vector(t(y))

  # Determine input dtype (first input only; multi-input always float32 for now)
  input_dtype <- if (!is.null(model$inputs[[1L]]$config$dtype)) {
    model$inputs[[1L]]$config$dtype
  } else {
    "float32"
  }
  if (is_multi) input_dtype <- "float32"

  optimizer_type <- switch(model$compilation$optimizer,
    "adam" = , "adamw" = ggml_opt_optimizer_type_adamw(),
    "sgd"  = ggml_opt_optimizer_type_sgd(),
    stop("Unsupported optimizer: ", model$compilation$optimizer, call. = FALSE)
  )
  # Per-head loss types and weights, resolved at compile time.
  loss_types   <- vapply(loss_spec, function(s) s$loss_type, integer(1))
  loss_weights <- vapply(loss_spec, function(s) s$weight,    numeric(1))
  loss_type    <- loss_types[[1L]]  # single-head path below uses this

  # Cross-entropy training needs logits, because ggml_cross_entropy_loss()
  # softmaxes its own input.
  # One flag per model output: only the cross-entropy heads get their softmax
  # stripped, so a CE head and an MSE head can coexist in the same model.
  ce_flags <- vapply(loss_spec, function(s) s$is_ce, logical(1))
  if (is.null(trained_output_idx)) {
    use_ce_loss <- ce_flags
  } else {
    # Legacy single-y path: only the trained output is affected; the exposed
    # intermediate outputs keep their activations for ggml_predict().
    use_ce_loss <- rep(FALSE, length(model$outputs))
    use_ce_loss[trained_output_idx] <- ce_flags[[1L]]
  }

  train_loss_vec <- numeric(epochs)
  train_acc_vec  <- numeric(epochs)
  val_loss_vec   <- numeric(epochs)
  val_acc_vec    <- numeric(epochs)
  head_loss_mat  <- NULL   # [epochs x n_head], filled on the multi-output path
  head_acc_mat   <- NULL   # [epochs x n_head], NA for heads without accuracy
  val_head_loss_mat <- NULL  # same, for the validation portion
  val_head_acc_mat  <- NULL

  if (!is_multi) {
    # -----------------------------------------------------------------------
    # Single-input path — use dataset + ggml_opt_fit / ggml_opt_epoch
    # -----------------------------------------------------------------------
    data_type <- if (input_dtype == "int32") GGML_TYPE_I32 else GGML_TYPE_F32
    dataset <- ggml_opt_dataset_init(
      type_data    = data_type,
      type_label   = GGML_TYPE_F32,
      ne_datapoint = ne_datapoint,
      ne_label     = ne_label,
      ndata        = n_samples,
      ndata_shard  = 1L
    )
    ggml_backend_tensor_set_data(ggml_opt_dataset_data(dataset),   x_ggml)
    ggml_backend_tensor_set_data(ggml_opt_dataset_labels(dataset), y_ggml)

    graph_info <- nn_build_functional_graph(model, batch_size, training = TRUE,
                                            logits_output = use_ce_loss)
    fit_input  <- graph_info$inputs[[1L]]
    # Every output head is trained, not just the last one. graph_info$outputs is
    # in the same order as model$outputs, hence as loss_spec.
    fit_outputs <- graph_info$outputs
    # trained_output_idx is set only on the legacy single-y path, where just one
    # of several outputs is trained.
    fit_output  <- if (is.null(trained_output_idx)) {
      fit_outputs[[length(fit_outputs)]]
    } else {
      fit_outputs[[trained_output_idx]]
    }

    has_stochastic_dropout <- length(graph_info$dropout_masks) > 0L

    if (n_head > 1L) {
      if (length(fit_outputs) != n_head) {
        stop("Model has ", n_head, " compiled output heads but the built graph ",
             "produced ", length(fit_outputs), ".", call. = FALSE)
      }
      if (has_stochastic_dropout && verbose > 0L) {
        message("Note: per-batch dropout resampling is not applied on the ",
                "multi-output training path.")
      }
      # Offsets of each head inside the concatenated label rows, 0-based.
      labels_offs <- c(0, cumsum(head_widths))[seq_len(n_head)]

      history_raw <- ggml_fit_opt_multi(
        sched          = model$compilation$sched,
        ctx_compute    = graph_info$ctx_compute,
        inputs         = fit_input,
        outputs        = fit_outputs,
        dataset        = dataset,
        loss_types     = loss_types,
        loss_weights   = loss_weights,
        labels_offs    = labels_offs,
        # The heads keep their output-layer names, so a callback monitoring
        # e.g. "val_<output>_loss" uses the same key the history reports.
        head_names     = vapply(loss_spec, function(s) s$name, character(1)),
        optimizer      = optimizer_type,
        nepoch         = epochs,
        nbatch_logical = batch_size,
        val_split      = validation_split,
        shuffle        = shuffle,
        shuffle_all    = shuffle_all,
        callbacks      = callbacks,
        silent         = (verbose == 0L)
      )
      train_loss_vec <- history_raw$train_loss
      train_acc_vec  <- history_raw$train_accuracy
      val_loss_vec   <- history_raw$val_loss
      val_acc_vec    <- history_raw$val_accuracy
      head_loss_mat  <- attr(history_raw, "head_loss")
      head_acc_mat   <- attr(history_raw, "head_accuracy")
      val_head_loss_mat <- attr(history_raw, "val_head_loss")
      val_head_acc_mat  <- attr(history_raw, "val_head_accuracy")
      # One row per epoch actually run: a callback stopping early shortens it.
      epochs_run     <- nrow(history_raw)

    } else if (!has_stochastic_dropout) {

      # R-side epoch loop (ggml_fit_opt), not the single C call (ggml_opt_fit),
      # so callbacks get a hook between epochs.
      history_raw <- ggml_fit_opt(
        sched          = model$compilation$sched,
        ctx_compute    = graph_info$ctx_compute,
        inputs         = fit_input,
        outputs        = fit_output,
        dataset        = dataset,
        loss_type      = loss_type,
        optimizer      = optimizer_type,
        nepoch         = epochs,
        nbatch_logical = batch_size,
        val_split      = validation_split,
        shuffle        = shuffle,
        shuffle_all    = shuffle_all,
        callbacks      = callbacks,
        silent         = (verbose == 0L)
      )
      train_loss_vec <- history_raw$train_loss
      train_acc_vec  <- history_raw$train_accuracy
      val_loss_vec   <- history_raw$val_loss
      val_acc_vec    <- history_raw$val_accuracy
      epochs_run     <- nrow(history_raw)

    } else {
      n_batches_log <- n_samples %/% batch_size
      idata_split   <- as.integer((1.0 - validation_split) * n_batches_log) * batch_size

      init_info <- ggml_opt_init_for_fit(
        sched       = model$compilation$sched,
        loss_type   = loss_type,
        optimizer   = optimizer_type,
        opt_period  = 1L,
        ctx_compute = graph_info$ctx_compute,
        inputs      = fit_input,
        outputs     = fit_output
      )
      opt_ctx <- init_info$opt_ctx

      # Same shuffling contract as ggml_fit_opt(): the whole dataset once,
      # before the split, so the validation tail is a random sample rather than
      # the end of the input; then the training portion each epoch.
      if (shuffle_all && batch_size < n_samples) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1)
      }

      result_train <- ggml_opt_result_init()
      result_val   <- ggml_opt_result_init()

      # Mutable state shared with callbacks -- same contract as ggml_fit_opt().
      cb_state <- new.env(parent = emptyenv())
      cb_state$stop   <- FALSE
      cb_state$lr_ud  <- init_info$lr_ud
      cb_state$nepoch <- as.integer(epochs)

      epochs_run <- 0L
      for (ep in seq_len(epochs)) {
        logs <- list()
        for (cb in callbacks) {
          if (is.function(cb$on_epoch_begin)) cb$on_epoch_begin(ep, logs, cb_state)
          if (isTRUE(cb_state$stop)) break
        }
        if (isTRUE(cb_state$stop)) break

        if (shuffle && batch_size < idata_split) {
          ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split)
        }

        for (dm in graph_info$dropout_masks) {
          keep_prob <- 1.0 - dm$rate
          mask_vals <- as.numeric(runif(dm$ne) < keep_prob)
          ggml_backend_tensor_set_data(dm$mask, mask_vals)
        }
        # See ggml_fit_opt(): an R-side loop re-syncs the thread count itself.
        .ggml_sched_sync_threads(model$compilation$sched)
        if (verbose > 0L) cat(sprintf("Epoch %d/%d:\n", ep, epochs))
        ggml_opt_result_reset(result_train)
        ggml_opt_result_reset(result_val)
        ggml_opt_epoch(opt_ctx, dataset, result_train, result_val,
                       idata_split = idata_split,
                       callback_train = (verbose > 0L),
                       callback_eval  = (verbose > 0L))
        tl <- ggml_opt_result_loss(result_train)
        ta <- ggml_opt_result_accuracy(result_train)
        vl <- ggml_opt_result_loss(result_val)
        va <- ggml_opt_result_accuracy(result_val)
        train_loss_vec[ep] <- tl[["loss"]]
        train_acc_vec[ep]  <- ta[["accuracy"]]
        val_loss_vec[ep]   <- if (validation_split > 0) vl[["loss"]] else NA_real_
        val_acc_vec[ep]    <- if (validation_split > 0) va[["accuracy"]] else NA_real_
        epochs_run <- ep

        logs$train_loss     <- train_loss_vec[ep]
        logs$train_accuracy <- train_acc_vec[ep]
        logs$val_loss       <- val_loss_vec[ep]
        logs$val_accuracy   <- val_acc_vec[ep]
        for (cb in callbacks) {
          if (is.function(cb$on_epoch_end)) cb$on_epoch_end(ep, logs, cb_state)
          if (isTRUE(cb_state$stop)) break
        }
        if (isTRUE(cb_state$stop)) break
      }

      # Drop the tail of epochs a callback cut short.
      if (epochs_run < epochs) {
        keep <- seq_len(epochs_run)
        train_loss_vec <- train_loss_vec[keep]
        train_acc_vec  <- train_acc_vec[keep]
        val_loss_vec   <- val_loss_vec[keep]
        val_acc_vec    <- val_acc_vec[keep]
      }

      ggml_opt_result_free(result_train)
      ggml_opt_result_free(result_val)
      ggml_opt_free(opt_ctx)
    }

    model$node_weights            <- graph_info$node_weights
    model$compilation$ctx_weights <- graph_info$ctx_weights
    model$compilation$buffer      <- graph_info$buffer
    ggml_free(graph_info$ctx_compute)
    ggml_opt_dataset_free(dataset)

  } else {
    # -----------------------------------------------------------------------
    # Multi-input path — manual batch loop filling each input tensor
    # -----------------------------------------------------------------------
    # Split into train / val portions.  Which samples land in each is decided by
    # `perm` below, so with shuffle = TRUE the validation portion is a random
    # subset rather than whatever sits at the end of the input.
    n_train_samples <- as.integer(floor((1.0 - validation_split) * n_samples %/% batch_size) * batch_size)
    if (n_train_samples == 0L) n_train_samples <- n_samples

    graph_info <- nn_build_functional_graph(model, batch_size, training = TRUE,
                                            logits_output = use_ce_loss)
    fit_output <- if (is.null(trained_output_idx)) {
      graph_info$outputs[[length(graph_info$outputs)]]
    } else {
      graph_info$outputs[[trained_output_idx]]
    }

    if (n_head > 1L) {
      if (length(graph_info$outputs) != n_head) {
        stop("Model has ", n_head, " compiled output heads but the built graph ",
             "produced ", length(graph_info$outputs), ".", call. = FALSE)
      }
      # Offsets of each head inside the concatenated label rows, 0-based --
      # the same layout the single-input path builds. There is no
      # ggml_opt_dataset here to slice, so the offsets are used by the batch
      # loop below when it fills each head's labels tensor; they are still
      # handed to the context so it carries the same description of the label
      # layout as the dataset-driven path.
      labels_offs <- c(0, cumsum(head_widths))[seq_len(n_head)]

      init_info <- ggml_opt_init_for_fit_multi(
        sched        = model$compilation$sched,
        loss_types   = loss_types,
        loss_weights = loss_weights,
        labels_offs  = labels_offs,
        optimizer    = optimizer_type,
        opt_period   = 1L,
        ctx_compute  = graph_info$ctx_compute,
        inputs       = graph_info$inputs[[1L]],
        outputs      = graph_info$outputs
      )
    } else {
      labels_offs <- 0
      init_info <- ggml_opt_init_for_fit(
        sched       = model$compilation$sched,
        loss_type   = loss_type,
        optimizer   = optimizer_type,
        opt_period  = 1L,
        ctx_compute = graph_info$ctx_compute,
        inputs      = graph_info$inputs[[1L]],
        outputs     = fit_output
      )
    }
    opt_ctx      <- init_info$opt_ctx
    # One labels tensor per head. For a single head this is ggml_opt_labels().
    labels_tensors <- if (n_head > 1L) {
      lapply(seq_len(n_head), function(i) ggml_opt_labels_i(opt_ctx, i))
    } else {
      list(ggml_opt_labels(opt_ctx))
    }
    # Width of each head's slice of a label row; a single head owns the row.
    label_widths <- if (n_head > 1L) head_widths else ne_label

    # Per-head metric matrices, [epochs x n_head], with the same shape and
    # column names ggml_fit_opt_multi() returns, so the history assembly at the
    # end of ggml_fit() treats both paths alike.
    if (n_head > 1L) {
      head_names <- vapply(loss_spec, function(s) s$name, character(1))
      new_head_mat <- function() {
        matrix(NA_real_, nrow = epochs, ncol = n_head,
               dimnames = list(NULL, head_names))
      }
      head_loss_mat     <- new_head_mat()
      head_acc_mat      <- new_head_mat()
      val_head_loss_mat <- new_head_mat()
      val_head_acc_mat  <- new_head_mat()
    }

    result_train <- ggml_opt_result_init()
    result_val   <- ggml_opt_result_init()

    n_batches_train <- n_train_samples %/% batch_size
    n_batches_val   <- (n_samples - n_train_samples) %/% batch_size

    # Mutable state shared with callbacks -- same contract as ggml_fit_opt().
    cb_state <- new.env(parent = emptyenv())
    cb_state$stop   <- FALSE
    cb_state$lr_ud  <- init_info$lr_ud
    cb_state$nepoch <- as.integer(epochs)

    # There is no ggml_opt_dataset on this path, so shuffling is a permutation
    # of 0-based sample indices. Same contract as ggml_fit_opt(): permute
    # everything once before the split, then only the training portion each
    # epoch, leaving the validation samples fixed.
    perm <- seq_len(n_samples) - 1L
    if (shuffle_all) perm <- sample(perm)

    epochs_run <- 0L
    for (ep in seq_len(epochs)) {
      logs <- list()
      for (cb in callbacks) {
        if (is.function(cb$on_epoch_begin)) cb$on_epoch_begin(ep, logs, cb_state)
        if (isTRUE(cb_state$stop)) break
      }
      if (isTRUE(cb_state$stop)) break

      # Reshuffle the training portion only; perm[1:n_train_samples] are the
      # training samples, the tail stays put so validation is comparable
      # across epochs.
      if (shuffle && n_train_samples > 1L && n_train_samples < n_samples) {
        perm[seq_len(n_train_samples)] <- sample(perm[seq_len(n_train_samples)])
      } else if (shuffle && n_train_samples == n_samples && n_samples > 1L) {
        perm <- sample(perm)
      }

      # Regenerate dropout masks
      for (dm in graph_info$dropout_masks) {
        keep_prob <- 1.0 - dm$rate
        mask_vals <- as.numeric(runif(dm$ne) < keep_prob)
        ggml_backend_tensor_set_data(dm$mask, mask_vals)
      }

      # See ggml_fit_opt(): an R-side loop re-syncs the thread count itself.
      .ggml_sched_sync_threads(model$compilation$sched)

      if (verbose > 0L) cat(sprintf("Epoch %d/%d:\n", ep, epochs))

      ggml_opt_result_reset(result_train)
      ggml_opt_result_reset(result_val)

      # Training batches
      for (ib in seq_len(n_batches_train)) {
        idx <- perm[(ib - 1L) * batch_size + seq_len(batch_size)]
        nn_fill_inputs_idx(x_ggml, ne_per_input, graph_info$inputs, idx)
        nn_fill_labels_idx(y_ggml, ne_label, labels_tensors,
                           labels_offs, label_widths, idx)

        ggml_opt_alloc(opt_ctx, backward = TRUE)
        ggml_opt_eval(opt_ctx, result_train)
      }

      # Validation batches (forward only)
      if (n_batches_val > 0L) {
        for (ib in seq_len(n_batches_val)) {
          idx <- perm[n_train_samples + (ib - 1L) * batch_size + seq_len(batch_size)]
          nn_fill_inputs_idx(x_ggml, ne_per_input, graph_info$inputs, idx)
          nn_fill_labels_idx(y_ggml, ne_label, labels_tensors,
                             labels_offs, label_widths, idx)

          ggml_opt_alloc(opt_ctx, backward = FALSE)
          ggml_opt_eval(opt_ctx, result_val)
        }
      }

      tl <- ggml_opt_result_loss(result_train)
      ta <- ggml_opt_result_accuracy(result_train)
      vl <- ggml_opt_result_loss(result_val)
      va <- ggml_opt_result_accuracy(result_val)

      train_loss_vec[ep] <- tl[["loss"]]
      train_acc_vec[ep]  <- ta[["accuracy"]]
      val_loss_vec[ep]   <- if (validation_split > 0 && n_batches_val > 0L) vl[["loss"]]     else NA_real_
      val_acc_vec[ep]    <- if (validation_split > 0 && n_batches_val > 0L) va[["accuracy"]] else NA_real_

      if (verbose > 0L) {
        cat(sprintf("  train_loss=%.4f  train_acc=%.4f",
                    train_loss_vec[ep], train_acc_vec[ep]))
        if (!is.na(val_loss_vec[ep]))
          cat(sprintf("  val_loss=%.4f  val_acc=%.4f",
                      val_loss_vec[ep], val_acc_vec[ep]))
        cat("\n")
      }
      epochs_run <- ep

      logs$train_loss     <- train_loss_vec[ep]
      logs$train_accuracy <- train_acc_vec[ep]
      logs$val_loss       <- val_loss_vec[ep]
      logs$val_accuracy   <- val_acc_vec[ep]

      # Per-head metrics, same keys and semantics as ggml_fit_opt_multi():
      # both phases carry an explicit prefix so a head named "val_x" cannot
      # collide with another head's validation key. A result only holds heads
      # once an epoch has accumulated into it, so its own head count decides
      # what can be read rather than n_head.
      if (n_head > 1L) {
        n_res_train <- ggml_opt_result_n_loss(result_train)
        n_res_val   <- if (n_batches_val > 0L) ggml_opt_result_n_loss(result_val) else 0L
        for (i in seq_len(n_head)) {
          nm <- head_names[i]
          if (i <= n_res_train) {
            head_loss_mat[ep, i] <- ggml_opt_result_loss_i(result_train, i)[["loss"]]
            head_acc_mat[ep, i]  <- ggml_opt_result_accuracy_i(result_train, i)[["accuracy"]]
          }
          # unname(): the matrices carry the head names as column names, which a
          # scalar subset would drag into `logs` -- callbacks compare these
          # against plain numbers, so they must stay bare.
          logs[[paste0("train_", nm, "_loss")]]     <- unname(head_loss_mat[ep, i])
          logs[[paste0("train_", nm, "_accuracy")]] <- unname(head_acc_mat[ep, i])

          if (i <= n_res_val) {
            val_head_loss_mat[ep, i] <- ggml_opt_result_loss_i(result_val, i)[["loss"]]
            val_head_acc_mat[ep, i]  <- ggml_opt_result_accuracy_i(result_val, i)[["accuracy"]]
          }
          logs[[paste0("val_", nm, "_loss")]]     <- unname(val_head_loss_mat[ep, i])
          logs[[paste0("val_", nm, "_accuracy")]] <- unname(val_head_acc_mat[ep, i])
        }
      }

      for (cb in callbacks) {
        if (is.function(cb$on_epoch_end)) cb$on_epoch_end(ep, logs, cb_state)
        if (isTRUE(cb_state$stop)) break
      }
      if (isTRUE(cb_state$stop)) break
    }

    # Drop the tail of epochs a callback cut short.
    if (epochs_run < epochs) {
      keep <- seq_len(epochs_run)
      train_loss_vec <- train_loss_vec[keep]
      train_acc_vec  <- train_acc_vec[keep]
      val_loss_vec   <- val_loss_vec[keep]
      val_acc_vec    <- val_acc_vec[keep]
    }

    ggml_opt_result_free(result_train)
    ggml_opt_result_free(result_val)
    ggml_opt_free(opt_ctx)

    model$node_weights            <- graph_info$node_weights
    model$compilation$ctx_weights <- graph_info$ctx_weights
    model$compilation$buffer      <- graph_info$buffer
    ggml_free(graph_info$ctx_compute)
  }

  # Length of the metric vectors, not `epochs`: a callback may have stopped
  # training early, in which case every branch above truncated them.
  hist_list <- list(
    train_loss     = train_loss_vec,
    train_accuracy = train_acc_vec,
    val_loss       = val_loss_vec,
    val_accuracy   = val_acc_vec,
    epochs         = seq_along(train_loss_vec)
  )

  # Per-head metrics, named after the output layers ("train_<output>_loss",
  # "val_<output>_loss"), so it is visible which head is not learning -- the
  # aggregate train_loss can fall while one head stalls. These are the same
  # keys the callbacks see in `logs`, so a monitor= naming one of them matches
  # a column here. Both phases are prefixed, so an output named "val_x" cannot
  # collide with another output's validation column.
  if (n_head > 1L && !is.null(head_loss_mat)) {
    n_ep <- length(train_loss_vec)
    # A column is reported only if it holds something: accuracy exists for
    # cross-entropy heads only, and the validation columns are all-NA without a
    # validation split.
    # unname(): the matrices carry the head names as column names, and a
    # single-epoch subset would keep one as the vector's name -- so the columns
    # would be named for epochs = 1 and bare otherwise.
    put_col <- function(key, mat, i) {
      if (is.null(mat)) return(invisible(NULL))
      col <- unname(mat[seq_len(n_ep), i])
      if (all(is.na(col))) return(invisible(NULL))
      hist_list[[key]] <<- col
    }
    for (i in seq_len(n_head)) {
      nm <- loss_spec[[i]]$name
      hist_list[[paste0("train_", nm, "_loss")]] <- unname(head_loss_mat[seq_len(n_ep), i])
      put_col(paste0("train_", nm, "_accuracy"),   head_acc_mat,      i)
      put_col(paste0("val_", nm, "_loss"),         val_head_loss_mat, i)
      put_col(paste0("val_", nm, "_accuracy"),     val_head_acc_mat,  i)
    }
  }

  model$history <- structure(hist_list, class = "ggml_history")

  invisible(model)
}

# ============================================================================
# Evaluate -- S3 method for ggml_functional_model
# ============================================================================

#' @rdname ggml_evaluate
#' @param ... Additional arguments (ignored).
#' @export
ggml_evaluate.ggml_functional_model <- function(model, x, y,
                                                  batch_size = 32L, ...) {
  if (!model$compiled) stop("Model must be compiled before evaluation.")

  loss_spec <- model$compilation$loss_spec
  n_head    <- if (is.null(loss_spec)) 1L else length(loss_spec)

  # Multi-output: evaluate each head against its own y, then report the
  # weighted total that training actually optimizes, plus the per-head values.
  # As in ggml_fit(): a plain matrix means only the last output is scored,
  # the earlier ones being intermediate activations.
  if (n_head > 1L && !is.list(y)) {
    loss_spec <- loss_spec[n_head]
    n_head    <- 1L
  }

  if (n_head > 1L) {
    if (length(y) != n_head) {
      stop("This model has ", n_head, " outputs, so 'y' must be a list of ",
           n_head, " matrices (one per output).", call. = FALSE)
    }
    onames <- vapply(loss_spec, function(s) s$name, character(1))
    ynames <- names(y)
    if (!is.null(ynames) && !any(ynames == "")) {
      y <- y[match(onames, ynames)]
    }
    y <- lapply(y, function(yi) if (is.matrix(yi)) yi else as.matrix(yi))

    preds <- ggml_predict(model, x, batch_size = batch_size)
    if (!is.list(preds)) preds <- list(preds)

    out <- list(loss = 0, n_samples = nrow(y[[1L]]))
    total <- 0
    for (i in seq_len(n_head)) {
      pm <- preds[[i]]
      yi <- y[[i]]
      li <- nn_head_loss(loss_spec[[i]]$loss, pm, yi)
      out[[paste0(onames[[i]], "_loss")]] <- li
      total <- total + loss_spec[[i]]$weight * li
      if (ncol(yi) > 1L) {
        out[[paste0(onames[[i]], "_accuracy")]] <- mean(max.col(pm) == max.col(yi))
      }
    }
    out$loss <- total
    # Head-1 accuracy under the plain name, so callers that expect `accuracy`
    # still find one; the per-head values above are the unambiguous ones.
    acc1 <- out[[paste0(onames[[1L]], "_accuracy")]]
    out$accuracy <- if (is.null(acc1)) NA_real_ else acc1
    return(out)
  }

  n_samples <- nrow(y)
  ne_label  <- ncol(y)

  # Get predictions for ALL samples (no truncation)
  preds <- ggml_predict(model, x, batch_size = batch_size)
  # With several outputs and a plain `y` it is the LAST output that is scored
  # (the earlier ones are exposed intermediates), matching ggml_fit().
  preds_mat <- if (is.matrix(preds)) preds else preds[[length(preds)]]

  # Compute loss through the shared per-head implementation, so the single- and
  # multi-output paths agree and a newly added loss is not silently NA here.
  loss_name <- model$compilation$loss
  if (is.list(loss_name)) loss_name <- loss_name[[1L]]
  loss_val <- nn_head_loss(loss_name, preds_mat, y)

  # Compute accuracy (classification: argmax match)
  if (ne_label > 1L) {
    pred_classes <- max.col(preds_mat)
    true_classes <- max.col(y)
    acc_val <- mean(pred_classes == true_classes)
  } else {
    acc_val <- NA_real_
  }

  out <- list(loss = loss_val, accuracy = acc_val, n_samples = n_samples)

  # Additional metrics
  out <- nn_add_extra_metrics(out, model$compilation$metrics, preds_mat, y)
  out
}

# ============================================================================
# Predict -- S3 method for ggml_functional_model
# ============================================================================

#' @rdname ggml_predict
#' @param ... Additional arguments (ignored).
#' @export
ggml_predict.ggml_functional_model <- function(model, x, batch_size = 32L, ...) {
  if (!model$compiled) stop("Model must be compiled before prediction.")

  xp           <- nn_prepare_x(model, x)
  is_multi     <- xp$is_multi
  ne_per_input <- xp$ne_per_input
  ne_datapoint <- sum(ne_per_input)

  # Determine n_samples_orig before possible padding
  n_samples_orig <- if (is_multi) {
    nrow(as.matrix(x[[1L]]))
  } else if (is.matrix(x)) {
    nrow(x)
  } else {
    dim(x)[1L]
  }

  if (n_samples_orig < batch_size) {
    stop("Not enough samples (", n_samples_orig, ") for batch_size=", batch_size)
  }

  # Pad to batch boundary
  remainder <- n_samples_orig %% batch_size
  if (remainder != 0L) {
    n_pad <- batch_size - remainder
    if (is_multi) {
      x <- lapply(x, function(xi) {
        xi_mat <- matrix(as.numeric(xi), nrow = n_samples_orig)
        rbind(xi_mat, matrix(0.0, nrow = n_pad, ncol = ncol(xi_mat)))
      })
    } else if (is.matrix(x)) {
      x <- rbind(x, matrix(0.0, nrow = n_pad, ncol = ncol(x)))
    } else {
      pad_dims <- dim(x); pad_dims[1L] <- n_pad
      x <- abind_first(x, array(0.0, dim = pad_dims))
    }
    xp <- nn_prepare_x(model, x)
  }

  x_ggml    <- xp$x_ggml
  n_samples <- length(x_ggml) %/% ne_datapoint
  n_batches <- n_samples %/% batch_size

  graph_info <- nn_build_functional_graph(model, batch_size, training = FALSE)
  n_outputs  <- length(graph_info$outputs)
  sched      <- model$compilation$sched

  # Build forward graph covering all outputs.
  # Expanding from the last output alone is not enough: with several INDEPENDENT
  # branches (a multi-input model whose outputs share a layer but not an input)
  # the other outputs are unreachable from that root, so they never enter the
  # graph, the scheduler never assigns them a buffer, and reading them back
  # fails. Expand every output into the same graph.
  graph <- ggml_build_forward_expand(graph_info$ctx_compute,
                                     graph_info$outputs[[1L]])
  for (io in seq_len(n_outputs)[-1L]) {
    ggml_graph_expand(graph, graph_info$outputs[[io]])
  }

  out_shapes     <- lapply(model$outputs, function(o) graph_info$shapes[[o$id]])
  ne_outputs_vec <- vapply(out_shapes, prod, numeric(1))
  all_preds_list <- lapply(ne_outputs_vec, function(ne) {
    matrix(0.0, nrow = n_samples, ncol = ne)
  })

  # Allocate once, outside the loop -- re-running reset + alloc_graph per batch
  # lets the scheduler re-lay the intermediate buffers, and on Vulkan every pass
  # after the first then reads stale data. See nn_predict_batch_run() in
  # nn_model.R for the same fix and the measurements behind it.
  ggml_backend_sched_reset(sched)
  ggml_backend_sched_alloc_graph(sched, graph)

  for (ib in seq_len(n_batches)) {
    samp_start <- (ib - 1L) * batch_size

    if (is_multi) {
      nn_fill_inputs(x_ggml, ne_per_input, graph_info$inputs, batch_size, samp_start)
    } else {
      data_start <- samp_start * ne_datapoint + 1L
      data_end   <- data_start + batch_size * ne_datapoint - 1L
      ggml_backend_tensor_set_data(graph_info$inputs[[1L]], x_ggml[data_start:data_end])
    }

    ggml_backend_sched_graph_compute(sched, graph)

    row_start <- samp_start + 1L
    row_end   <- samp_start + batch_size
    for (io in seq_len(n_outputs)) {
      ne_out    <- ne_outputs_vec[io]
      batch_out <- ggml_backend_tensor_get_data(graph_info$outputs[[io]])
      mat       <- matrix(batch_out, nrow = ne_out, ncol = batch_size)
      all_preds_list[[io]][row_start:row_end, ] <- t(mat)
    }
  }

  ggml_free(graph_info$ctx_compute)
  ggml_backend_buffer_free(graph_info$buffer)
  ggml_free(graph_info$ctx_weights)

  # Trim padding and return
  if (n_outputs == 1L) {
    return(all_preds_list[[1L]][seq_len(n_samples_orig), , drop = FALSE])
  } else {
    return(lapply(all_preds_list, function(m) m[seq_len(n_samples_orig), , drop = FALSE]))
  }
}

# ============================================================================
# Print method
# ============================================================================

#' Print method for ggml_functional_model
#' @param x A ggml_functional_model object
#' @param ... Additional arguments (ignored)
#' @return The model object (invisibly).
#' @export
print.ggml_functional_model <- function(x, ...) {
  model <- x
  cat("ggmlR Functional Model\n")
  cat(paste(rep("=", 60), collapse = ""), "\n")
  cat(sprintf("Inputs:   %d\n", length(model$inputs)))
  cat(sprintf("Outputs:  %d\n", length(model$outputs)))
  cat(sprintf("Compiled: %s\n", if (model$compiled) "yes" else "no"))

  nodes <- nn_topo_sort(model$outputs)
  total_params <- 0L
  cat(sprintf("\n%-20s %-15s\n", "Layer (type)", "Node type"))
  cat(paste(rep("-", 40), collapse = ""), "\n")
  for (node in nodes) {
    n_params <- switch(node$node_type,
      "dense" = {
        # approximate: fan_in * units + units
        0L  # shape not available here without building
      },
      0L
    )
    nm <- if (!is.null(node$config$name)) node$config$name else node$id
    cat(sprintf("%-20s %-15s\n", nm, node$node_type))
  }
  cat(paste(rep("=", 60), collapse = ""), "\n")
  invisible(x)
}
