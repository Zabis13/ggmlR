# Graph-level tensor parallelism through ggml's meta backend.
#
# ggml_vulkan_split_mul_mat() (tensor_parallel.R) splits ONE matrix and gathers
# right away. The meta backend splits a whole graph: every weight is laid out
# across the devices by a rule, every op runs on all devices on its own share,
# and the meta backend inserts the cross-device reduction only where a result
# has to be combined (a column-parallel layer followed by a row-parallel one
# needs no traffic in between). Without a backend-specific all-reduce (Vulkan
# has none) the meta backend falls back to tensor copies between the devices --
# host staging, the same transport as the rest of the multi-GPU API.

# ggml_backend_meta_split_axis (ggml-backend.h)
.ggmlr_meta_axis <- c(cols = 0L, rows = 1L, axis2 = 2L, axis3 = 3L,
                      mirror = 10L, partial = 11L)

#' Split-axis codes for the meta backend
#'
#' Integer codes of \code{ggml_backend_meta_split_axis}, for a raw
#' \code{split_fn} passed to \code{\link{ggml_backend_meta_device}}.
#' \code{\link{ggml_meta_backend}} takes the names instead
#' (\code{"cols"}, \code{"rows"}, \code{"mirror"}, ...).
#'
#' @format Integer scalars.
#' @name GGML_SPLIT_AXIS
#' @family meta backend
#' @export
GGML_SPLIT_AXIS_0 <- 0L
#' @rdname GGML_SPLIT_AXIS
#' @export
GGML_SPLIT_AXIS_1 <- 1L
#' @rdname GGML_SPLIT_AXIS
#' @export
GGML_SPLIT_AXIS_2 <- 2L
#' @rdname GGML_SPLIT_AXIS
#' @export
GGML_SPLIT_AXIS_3 <- 3L
#' @rdname GGML_SPLIT_AXIS
#' @export
GGML_SPLIT_AXIS_MIRRORED <- 10L
#' @rdname GGML_SPLIT_AXIS
#' @export
GGML_SPLIT_AXIS_PARTIAL <- 11L

# devices -> list of ggml_backend_dev_t external pointers.
.ggmlr_meta_devices <- function(devices) {
  if (is.list(devices)) {
    ok <- vapply(devices, function(d) inherits(d, "externalptr"), logical(1))
    if (!all(ok)) stop("devices: a list must hold ggml_backend_dev_t pointers")
    return(devices)
  }
  if (is.character(devices)) {
    return(lapply(devices, function(nm) {
      d <- if (identical(tolower(nm), "cpu")) {
        ggml_backend_dev_by_type(ggml_backend_device_type_cpu())
      } else {
        ggml_backend_dev_by_name(nm)
      }
      if (is.null(d)) stop(sprintf("devices: no device named '%s'", nm))
      d
    }))
  }
  if (is.numeric(devices)) {
    if (anyNA(devices) || any(devices != as.integer(devices)) || any(devices < 0)) {
      stop("devices: Vulkan device ids must be non-negative integers (0-based)")
    }
    n_vk <- ggml_vulkan_device_count()
    if (any(devices >= n_vk)) {
      stop(sprintf("devices: Vulkan device id out of range (have %d device(s))", n_vk))
    }
    return(lapply(as.integer(devices), function(i) {
      d <- ggml_backend_dev_by_name(paste0("Vulkan", i))
      if (is.null(d)) stop(sprintf("devices: Vulkan%d is not registered", i))
      d
    }))
  }
  stop("devices must be Vulkan ids (0-based integers), device names, or a list of device pointers")
}

# Per-device sizes of n along a split axis: proportional to `weights`, each a
# multiple of `gran`, summing to n exactly (the meta backend asserts both).
.ggmlr_meta_shares <- function(n, n_devs, weights, gran) {
  if (n %% gran != 0) {
    stop(sprintf("axis length %d is not a multiple of the block size %d", n, gran))
  }
  units <- n %/% gran
  w <- if (is.null(weights)) rep(1, n_devs) else weights
  raw <- units * w / sum(w)
  out <- floor(raw)
  # hand the leftover units to the largest remainders
  left <- units - sum(out)
  if (left > 0) {
    idx <- order(raw - out, decreasing = TRUE)[seq_len(left)]
    out[idx] <- out[idx] + 1
  }
  as.numeric(out * gran)
}

# One rule (character / integer / list) -> list(axis, weights).
.ggmlr_meta_rule <- function(rule, what) {
  weights <- NULL
  if (is.list(rule)) {
    weights <- rule$weights
    rule <- rule$axis
  }
  if (is.character(rule) && length(rule) == 1L) {
    axis <- .ggmlr_meta_axis[rule]
    if (is.na(axis)) {
      stop(sprintf("%s: unknown split '%s' (use %s)", what, rule,
                   paste0("\"", names(.ggmlr_meta_axis), "\"", collapse = ", ")))
    }
  } else if (is.numeric(rule) && length(rule) == 1L && rule %in% .ggmlr_meta_axis) {
    axis <- as.integer(rule)
  } else {
    stop(sprintf("%s: a split rule is a name, an axis code or list(axis =, weights =)", what))
  }
  list(axis = unname(axis), weights = weights)
}

#' Tensor-parallel backend over several devices (meta backend)
#'
#' Wraps several devices into one ggml "meta" backend. Weights allocated in it
#' are laid out across the devices by \code{split}; a graph computed on it runs
#' on every device at once, each on its own share, and the meta backend
#' inserts the cross-device reductions itself. That is graph-level tensor
#' parallelism: unlike \code{\link{ggml_vulkan_split_mul_mat}}, which splits a
#' single matrix and gathers immediately, a column-split layer followed by a
#' row-split one exchanges nothing in between.
#'
#' Split rules, by tensor name (\code{\link{ggml_set_name}}):
#' \describe{
#'   \item{\code{"mirror"}}{a full copy on every device (the default).}
#'   \item{\code{"rows"}}{split along ggml axis 1 (\code{ne[1]}): for a
#'     weight \code{[n_in, n_out]} the output features -- column parallel.}
#'   \item{\code{"cols"}}{split along ggml axis 0 (\code{ne[0]}): the input
#'     features -- row parallel; the partial sums are reduced.}
#'   \item{\code{"axis2"}, \code{"axis3"}}{split along \code{ne[2]} /
#'     \code{ne[3]}.}
#'   \item{\code{list(axis = , weights = )}}{any of the above with uneven
#'     shares, e.g. \code{list(axis = "rows", weights = c(0.7, 0.3))}.}
#' }
#' \code{split} may also be a \code{function(name, ne)} returning a rule (or
#' \code{NULL} for the default).
#'
#' Only tensors in buffers allocated on the returned backend are asked for a
#' rule (weights and inputs); intermediate results get their layout from the
#' op that produces them. Use the backend through a scheduler whose last
#' backend is the CPU, e.g.
#' \code{ggml_backend_sched_new(list(meta, ggml_backend_cpu_init()))}.
#'
#' A rule that cannot be applied (the axis is shorter than the number of
#' devices, or not a multiple of the type's block size) falls back to
#' \code{"mirror"} for that tensor with a warning. Rules for names that never
#' show up are not an error; see \code{attr(backend, "meta_split_log")}, an
#' environment whose \code{applied} element records, by name, the rule each
#' tensor received, and whose \code{warnings} element collects the fallback
#' messages. The rules are evaluated from C while the buffer is allocated, in
#' a context of its own: those warnings reach the console, but not a
#' \code{withCallingHandlers()} / \code{tryCatch()} around the allocation --
#' check \code{warnings} instead.
#'
#' A tensor that ends up holding per-device partial sums (a matrix product
#' whose inputs are both split along the reduced axis, e.g. \code{"cols"} on
#' the weight and on its input) is reduced before its first consumer. It cannot
#' be read itself, and a graph must not end in one: read a tensor computed from
#' it. \code{\link{ggml_tp_forward}} takes care of the latter.
#'
#' The data path between devices is host staging (no Vulkan backend offers a
#' device-to-device all-reduce), so this pays off only when a model does not
#' fit on one device -- see \code{vignette("multi-gpu")}.
#'
#' @param devices Vulkan device ids (0-based integers), device names
#'   (\code{"Vulkan0"}, \code{"CPU"}), or a list of \code{ggml_backend_dev_t}
#'   pointers. The same device may be listed twice (useful for testing on one
#'   GPU or on the CPU).
#' @param split Named list of rules, or \code{function(name, ne)}; see Details.
#' @param default Rule for tensors \code{split} does not mention.
#' @return A \code{ggml_backend_t} external pointer. Free it with
#'   \code{\link{ggml_backend_free}}. It keeps the meta device alive.
#' @family meta backend
#' @seealso \code{\link{ggml_backend_meta_device}} for the raw callback API,
#'   \code{\link{ggml_vulkan_split_mul_mat}}, \code{\link{ggml_pp_forward}}
#' @examples
#' \donttest{
#' cpu <- "CPU"
#' meta <- ggml_meta_backend(c(cpu, cpu), split = list(w = "rows"))
#' ggml_backend_free(meta)
#' }
#' @export
ggml_meta_backend <- function(devices, split = list(), default = "mirror") {
  devs <- .ggmlr_meta_devices(devices)
  n_devs <- length(devs)
  if (n_devs < 1L) stop("devices: need at least one device")

  if (!is.function(split)) {
    if (!is.list(split) || (length(split) > 0L &&
        (is.null(names(split)) || any(names(split) == "")))) {
      stop("split must be a named list of rules or a function(name, ne)")
    }
    # validate every rule now, not inside the allocation callback
    for (nm in names(split)) .ggmlr_meta_rule(split[[nm]], sprintf("split[[\"%s\"]]", nm))
  }
  default_rule <- .ggmlr_meta_rule(default, "default")

  log <- new.env(parent = emptyenv())
  log$applied <- list()
  log$warnings <- character()
  log$emit <- TRUE   # ggml_tp_forward turns this off and re-signals them itself

  split_fn <- function(tensor_info, n_devs) {
    name <- tensor_info$name
    ne <- tensor_info$ne
    mirrored <- list(axis = 10L, ne = rep(0, n_devs), n_segments = 1L)
    fallback <- function(why) {
      msg <- sprintf("ggml_meta_backend: '%s': %s; mirrored", name, why)
      log$warnings <- c(log$warnings, msg)
      if (isTRUE(log$emit)) warning(msg, call. = FALSE)
      log$applied[[name]] <- "mirror"
      mirrored
    }
    # An error escaping this function would make the C side mirror EVERY
    # tensor from here on (sticky fallback); keep a bad rule local to its tensor.
    r <- tryCatch({
      rule <- if (is.function(split)) split(name, ne) else split[[name]]
      if (is.null(rule)) default_rule else .ggmlr_meta_rule(rule, "rule")
    }, error = function(e) conditionMessage(e))
    if (is.character(r)) return(fallback(r))

    if (!(r$axis %in% 0:3)) {
      log$applied[[name]] <- names(.ggmlr_meta_axis)[match(r$axis, .ggmlr_meta_axis)]
      return(list(axis = r$axis, ne = rep(0, n_devs), n_segments = 1L))
    }
    if (!is.null(r$weights) && length(r$weights) != n_devs) {
      return(fallback(sprintf("weights has %d entries for %d devices", length(r$weights), n_devs)))
    }
    n <- ne[r$axis + 1L]
    gran <- if (r$axis == 0L) ggml_blck_size(tensor_info$type) else 1L
    if (n < n_devs * gran || n %% gran != 0) {
      return(fallback(sprintf("axis %d has length %d, cannot split over %d devices", r$axis, n, n_devs)))
    }
    log$applied[[name]] <- names(.ggmlr_meta_axis)[match(r$axis, .ggmlr_meta_axis)]
    list(axis = r$axis, ne = .ggmlr_meta_shares(n, n_devs, r$weights, gran), n_segments = 1L)
  }

  meta_dev <- ggml_backend_meta_device(devs, split_fn)
  backend <- ggml_backend_dev_init(meta_dev)
  if (is.null(backend)) stop("ggml_meta_backend: could not initialise the meta backend")
  # The device XPtr owns the callback's state (freed by its finalizer); the
  # backend calls back into it, so it must not be collected first.
  attr(backend, "meta_device") <- meta_dev
  attr(backend, "meta_split_log") <- log
  backend
}

#' Tensor-parallel forward pass of a whole graph (meta backend)
#'
#' Runs \code{forward} once over several devices with graph-level tensor
#' parallelism: the weights are laid out across the devices by \code{split}
#' (see \code{\link{ggml_meta_backend}}), every op runs on all devices on its
#' own share, and the reductions between devices are inserted automatically.
#' The counterpart of \code{\link{ggml_pp_forward}} (split by layers) and
#' \code{\link{ggml_vulkan_split_mul_mat}} (one matrix at a time).
#'
#' Shapes follow ggml: an R array's \code{dim()} is the tensor's \code{ne},
#' fastest dimension first -- the layout \code{\link{ggml_backend_tensor_set_data}}
#' uses. A weight \code{W} with \code{dim(W) = c(n_in, n_out)} and an input with
#' \code{dim(x) = c(n_in, batch)} give \code{ggml_mul_mat(ctx, w$W, input)} of
#' shape \code{c(n_out, batch)}, i.e. \code{t(W) \%*\% x} in R terms.
#'
#' The weights get the names of the \code{weights} list, so \code{split} rules
#' refer to them directly; the input tensor is named \code{"input"} (mirrored
#' unless \code{split} says otherwise). Intermediate tensors take their layout
#' from the ops. The classic MLP pattern is \code{split = list(w1 = "rows",
#' w2 = "cols")}: the hidden layer stays split, and the only reduction is
#' after \code{w2}.
#'
#' @param weights Named list of numeric arrays (or vectors, as 1-D tensors),
#'   at most 4 dimensions each; \code{dim()} is the ggml \code{ne}.
#' @param forward \code{function(ctx, input, w)} building the graph with
#'   \code{ggml_*} ops and returning the output tensor; \code{w} is the named
#'   list of weight tensors.
#' @param x Numeric array (or vector) of input data; \code{dim()} is the ggml
#'   \code{ne} of the \code{"input"} tensor.
#' @param devices,split,default See \code{\link{ggml_meta_backend}}.
#' @param type Storage type of the weights: \code{GGML_TYPE_F32} (default) or
#'   \code{GGML_TYPE_F16}. The input is always F32.
#' @param mem_size Bytes of ggml context metadata for the weights and for the
#'   graph (each); raise it for graphs with very many ops.
#' @return Numeric array with the output; \code{dim()} is the output tensor's
#'   \code{ne} without trailing unit dimensions (a plain vector for 1-D).
#' @family meta backend
#' @seealso \code{\link{ggml_meta_backend}}, \code{\link{ggml_pp_forward}},
#'   \code{\link{ggml_vulkan_split_mul_mat}}
#' @examples
#' \donttest{
#' n_in <- 16L; n_hid <- 32L; n_out <- 8L; batch <- 4L
#' W1 <- array(rnorm(n_in * n_hid), c(n_in, n_hid))
#' W2 <- array(rnorm(n_hid * n_out), c(n_hid, n_out))
#' X  <- array(rnorm(n_in * batch), c(n_in, batch))
#' Y <- ggml_tp_forward(
#'   weights = list(w1 = W1, w2 = W2),
#'   forward = function(ctx, input, w)
#'     ggml_mul_mat(ctx, w$w2, ggml_relu(ctx, ggml_mul_mat(ctx, w$w1, input))),
#'   x = X, devices = c("CPU", "CPU"),
#'   split = list(w1 = "rows", w2 = "cols"))
#' max(abs(Y - t(W2) %*% pmax(t(W1) %*% X, 0)))
#' }
#' @export
ggml_tp_forward <- function(weights, forward, x, devices, split = list(),
                            default = "mirror", type = GGML_TYPE_F32,
                            mem_size = 16L * 1024L * 1024L) {
  if (!is.list(weights) || length(weights) < 1L || is.null(names(weights)) ||
      any(names(weights) == "") || anyDuplicated(names(weights))) {
    stop("weights must be a non-empty list with unique names")
  }
  if ("input" %in% names(weights)) {
    stop("weights: the name 'input' is reserved for the input tensor")
  }
  if (!is.function(forward)) stop("forward must be a function(ctx, input, w)")
  if (!(type %in% c(GGML_TYPE_F32, GGML_TYPE_F16))) {
    stop("type must be GGML_TYPE_F32 or GGML_TYPE_F16")
  }
  shape_of <- function(a, what) {
    if (!is.numeric(a) && !is.logical(a)) stop(sprintf("%s must be numeric", what))
    d <- if (is.null(dim(a))) length(a) else dim(a)
    if (length(d) > 4L) stop(sprintf("%s has %d dimensions; at most 4", what, length(d)))
    if (any(d < 1L)) stop(sprintf("%s has an empty dimension", what))
    as.numeric(d)
  }
  w_shapes <- Map(shape_of, weights, sprintf("weights$%s", names(weights)))
  x_shape <- shape_of(x, "x")

  meta <- ggml_meta_backend(devices, split = split, default = default)
  on.exit(ggml_backend_free(meta), add = TRUE, after = FALSE)
  split_log <- attr(meta, "meta_split_log")
  split_log$emit <- FALSE
  cpu <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(cpu), add = TRUE, after = FALSE)

  # Weights and input in their own context: only tensors in a buffer allocated
  # on the meta backend are given a split rule; the graph's intermediates must
  # live in the scheduler's compute buffers and take their layout from the ops.
  ctx_w <- ggml_init(mem_size, no_alloc = TRUE)
  on.exit(ggml_free(ctx_w), add = TRUE, after = FALSE)
  new_t <- function(name, ne, ttype) {
    t <- ggml_new_tensor(ctx_w, ttype, length(ne), ne)
    ggml_set_name(t, name)
    t
  }
  w <- Map(new_t, names(weights), w_shapes, MoreArgs = list(ttype = type))
  input <- new_t("input", x_shape, GGML_TYPE_F32)

  buf <- ggml_backend_alloc_ctx_tensors(ctx_w, meta)
  if (is.null(buf)) stop("ggml_tp_forward: could not allocate the weights (out of memory?)")
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  # The split rules ran from C during the allocation, out of reach of the
  # caller's handlers; signal their fallbacks here, where they can be caught.
  for (msg in unique(split_log$warnings)) warning(msg, call. = FALSE)
  for (nm in names(weights)) ggml_backend_tensor_set_data(w[[nm]], as.numeric(weights[[nm]]))
  ggml_backend_tensor_set_data(input, as.numeric(x))

  ctx_g <- ggml_init(mem_size, no_alloc = TRUE)
  on.exit(ggml_free(ctx_g), add = TRUE, after = FALSE)
  out <- forward(ctx_g, input, w)
  if (!inherits(out, "externalptr")) stop("forward must return a ggml tensor")
  # The meta backend reduces partial sums only before a consumer, never after
  # the graph's last node, and refuses to read them. A row-parallel last layer
  # (e.g. split = list(w2 = "cols")) leaves exactly that, so give the output a
  # consumer: x * 1 sees the reduced input and is mirrored or split, readable
  # either way.
  out <- ggml_scale(ctx_g, out, 1)
  ggml_set_name(out, "tp_output")
  ggml_set_output(out)
  gf <- ggml_build_forward_expand(ctx_g, out)

  sched <- ggml_backend_sched_new(list(meta, cpu), parallel = FALSE)
  on.exit(ggml_backend_sched_free(sched), add = TRUE, after = FALSE)
  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    stop("ggml_tp_forward: could not allocate the graph")
  }
  status <- ggml_backend_sched_graph_compute(sched, gf)
  if (!identical(as.integer(status), 0L)) {
    stop(sprintf("ggml_tp_forward: compute failed (ggml_status %d)", as.integer(status)))
  }

  y <- ggml_backend_tensor_get_data(out)
  nd <- max(1L, ggml_n_dims(out))
  if (nd == 1L) return(y)
  array(y, dim = ggml_tensor_shape(out)[seq_len(nd)])
}
