# Recurrent layers for the autograd engine: LSTM and GRU cells built from
# existing ag_* ops (one tape node per op, so gradients, the GPU and the graph
# backward come for free), and sequence layers that loop a cell over time.
# Gate order and the weight layout are PyTorch's, so weights move one to one.

#' LSTM and GRU cells
#'
#' One time step of an LSTM or a GRU, composed of \code{ag_*} ops. Inputs are
#' \code{[in, B]} (features by batch, as everywhere in the autograd engine);
#' the state is a list: \code{list(h, c)} for the LSTM, \code{list(h)} for the
#' GRU, each \code{[hidden, B]}.
#'
#' Weight layout and gate order follow the PyTorch documentation, so a model
#' moves from \code{torch.nn.LSTM} / \code{torch.nn.GRU} (or the R torch
#' package) by copying \code{weight_ih_l0}, \code{weight_hh_l0},
#' \code{bias_ih_l0}, \code{bias_hh_l0} into \code{weight_ih}, \code{weight_hh},
#' \code{bias_ih}, \code{bias_hh}:
#' \itemize{
#'   \item LSTM, gates \code{i, f, g, o} stacked by rows (\code{[4 hidden, in]}):
#'     \eqn{c' = \sigma(f) c + \sigma(i) \tanh(g)},
#'     \eqn{h' = \sigma(o) \tanh(c')};
#'   \item GRU, gates \code{r, z, n} (\code{[3 hidden, in]}):
#'     \eqn{n = \tanh(W_{in} x + b_{in} + r (W_{hn} h + b_{hn}))},
#'     \eqn{h' = (1 - z) n + z h}. The two biases are not interchangeable here.
#' }
#'
#' \code{init}:
#' \describe{
#'   \item{\code{"rl"} (default)}{\code{weight_ih} Glorot uniform,
#'     \code{weight_hh} orthogonal per gate block (each \code{[hidden, hidden]}
#'     map is orthogonal), biases 0 except the LSTM forget gate, 1 (Gers et
#'     al. 2000), so the cell starts by keeping its memory. This deliberately
#'     differs from PyTorch. For the GRU, \code{z} starts at
#'     \eqn{\sigma(0) = 0.5}: half memory, half new; there is no forget-bias
#'     analogue on purpose.}
#'   \item{\code{"torch"}}{everything \eqn{U(-1/\sqrt{hidden}, 1/\sqrt{hidden})},
#'     as PyTorch.}
#' }
#'
#' \code{starts} marks the copies whose episode begins at this step: their
#' incoming state is zeroed before the step (SB3 RecurrentPPO's
#' \code{episode_starts}, i.e. the \code{done} of the previous step under
#' automatic resets, when the observation is already the new episode's).
#' It is a length-B 0/1 (or logical) vector, checked; or an ag_tensor
#' \code{[1, B]} (an input of \code{\link{ag_capture_step}}, where host data
#' inside the recording is refused), applied as \eqn{s - s \cdot starts} on the
#' device. A tensor's values are NOT checked: they must be 0 or 1, a fraction
#' resets the state partly.
#'
#' @param input_size Number of input features.
#' @param hidden_size Size of the state.
#' @param init \code{"rl"} or \code{"torch"}, see Details.
#' @return An \code{ag_lstm_cell} / \code{ag_gru_cell} environment with
#'   \code{forward(x, state = NULL, starts = NULL)} returning the new state
#'   (\code{state = NULL}: zeros; \code{starts}: length-B 0/1 or logical),
#'   \code{parameters()} (\code{weight_ih}, \code{weight_hh}, \code{bias_ih},
#'   \code{bias_hh}; biases are \code{[gates * hidden, 1]}) and the four
#'   parameters as fields.
#' @seealso \code{\link{ag_lstm}}, \code{\link{ag_gru}} for whole sequences.
#' @export
#' @examples
#' \donttest{
#' cell <- ag_lstm_cell(4L, 8L)
#' s <- NULL
#' for (t in 1:3) s <- cell$forward(ag_tensor(matrix(runif(4 * 2), 4, 2)), s,
#'                                 starts = c(t == 1, FALSE))
#' dim(as.matrix(s$h))
#' }
ag_lstm_cell <- function(input_size, hidden_size, init = c("rl", "torch")) {
  .ag_rnn_cell("lstm", input_size, hidden_size, match.arg(init))
}

#' @rdname ag_lstm_cell
#' @export
ag_gru_cell <- function(input_size, hidden_size, init = c("rl", "torch")) {
  .ag_rnn_cell("gru", input_size, hidden_size, match.arg(init))
}

#' LSTM and GRU over a sequence
#'
#' Loops an \code{\link{ag_lstm_cell}} / \code{\link{ag_gru_cell}} over a
#' list of time steps; the layer owns the cell, so \code{layer$cell$forward}
#' runs single steps (rollouts) with the same weights the sequence trains.
#' Each step adds its ops to the tape, so the recording grows with \code{T};
#' sequences of 16..128 steps, as in recurrent PPO, are the intended use.
#'
#' @inheritParams ag_lstm_cell
#' @return An \code{ag_lstm} / \code{ag_gru} environment with
#'   \code{forward(xs, state = NULL, starts = NULL)}: \code{xs} a list of
#'   \code{T} inputs \code{[in, B]}, \code{starts} \code{NULL}, a
#'   \code{[T, B]} 0/1 (logical) matrix or a \code{[T, B]} ag_tensor of 0/1
#'   (see \code{\link{ag_lstm_cell}}), row \code{t} applied before step
#'   \code{t}. Returns \code{list(outputs, state)}: the \code{T} hidden states
#'   \code{h} and the final state. Plus \code{parameters()} (the cell's) and
#'   \code{cell}.
#' @export
#' @examples
#' \donttest{
#' rnn <- ag_gru(3L, 5L)
#' xs  <- lapply(1:4, function(t) ag_tensor(matrix(rnorm(3 * 2), 3, 2)))
#' r   <- rnn$forward(xs)
#' length(r$outputs)
#' }
ag_lstm <- function(input_size, hidden_size, init = c("rl", "torch")) {
  .ag_rnn_layer(ag_lstm_cell(input_size, hidden_size, match.arg(init)))
}

#' @rdname ag_lstm
#' @export
ag_gru <- function(input_size, hidden_size, init = c("rl", "torch")) {
  .ag_rnn_layer(ag_gru_cell(input_size, hidden_size, match.arg(init)))
}

.ag_rnn_cell <- function(kind, input_size, hidden_size, init) {
  ok <- function(v) is.numeric(v) && length(v) == 1L && !is.na(v) && v >= 1 && v == round(v)
  if (!ok(input_size) || !ok(hidden_size))
    stop(sprintf("ag_%s_cell: input_size and hidden_size must be positive whole numbers", kind),
         call. = FALSE)
  inp <- as.integer(input_size); H <- as.integer(hidden_size)
  G <- if (kind == "lstm") 4L else 3L
  w <- .ag_rnn_init(init, kind, G, H, inp)

  env <- new.env(parent = emptyenv())
  env$kind        <- kind
  env$input_size  <- inp
  env$hidden_size <- H
  env$init        <- init
  env$training    <- TRUE
  env$weight_ih <- ag_param(w$weight_ih)
  env$weight_hh <- ag_param(w$weight_hh)
  env$bias_ih   <- ag_param(w$bias_ih)
  env$bias_hh   <- ag_param(w$bias_hh)

  env$forward <- function(x, state = NULL, starts = NULL) {
    if (!is_ag_tensor(x)) x <- ag_tensor(as.matrix(x))
    xd <- if (x$device == "gpu") .ag_operand(x) else .ag_data(x)
    if (.ag_nrow(xd) != inp)
      stop(sprintf("ag_%s_cell: x has %d rows, expected input_size = %d",
                   kind, .ag_nrow(xd), inp), call. = FALSE)
    B <- .ag_ncol(xd)
    state <- .ag_rnn_state(kind, state, starts, H, B, x$device)
    if (kind == "lstm") .ag_lstm_step(env, x, state, H) else .ag_gru_step(env, x, state, H)
  }

  env$parameters <- function()
    list(weight_ih = env$weight_ih, weight_hh = env$weight_hh,
         bias_ih = env$bias_ih, bias_hh = env$bias_hh)

  class(env) <- c(paste0("ag_", kind, "_cell"), "ag_layer")
  env
}

.ag_rnn_layer <- function(cell) {
  env <- new.env(parent = emptyenv())
  env$cell        <- cell
  env$kind        <- cell$kind
  env$input_size  <- cell$input_size
  env$hidden_size <- cell$hidden_size
  env$training    <- TRUE
  # the same ag_param objects as the cell's: one set of weights
  for (nm in c("weight_ih", "weight_hh", "bias_ih", "bias_hh")) env[[nm]] <- cell[[nm]]

  env$forward <- function(xs, state = NULL, starts = NULL) {
    if (is_ag_tensor(xs) || !is.list(xs) || length(xs) < 1L)
      stop(sprintf("ag_%s: xs must be a non-empty list of [in, B] inputs, one per step",
                   env$kind), call. = FALSE)
    n_steps <- length(xs)
    st_tensor <- is_ag_tensor(starts)
    if (!is.null(starts)) {
      if (!st_tensor) starts <- as.matrix(starts)
      nr <- if (st_tensor) .ag_nrow(if (starts$device == "gpu") .ag_operand(starts)
                                   else .ag_data(starts)) else nrow(starts)
      if (nr != n_steps)
        stop(sprintf("ag_%s: starts must have one row per step (%d), got %d",
                     env$kind, n_steps, nr), call. = FALSE)
    }
    outputs <- vector("list", n_steps)
    for (t in seq_len(n_steps)) {
      st <- if (is.null(starts)) NULL
            else if (st_tensor) ag_slice_rows(starts, t - 1L, 1L)   # row t, on the device
            else starts[t, ]
      state <- cell$forward(xs[[t]], state, st)
      outputs[[t]] <- state$h
    }
    list(outputs = outputs, state = state)
  }

  env$parameters <- function() cell$parameters()

  class(env) <- c(paste0("ag_", cell$kind), "ag_layer")
  env
}

# Initial weights. "rl": Glorot input part, orthogonal recurrent block per gate,
# LSTM forget bias 1 (on bias_ih; bias_hh 0). "torch": U(-1/sqrt(H), 1/sqrt(H)).
.ag_rnn_init <- function(init, kind, G, H, inp) {
  if (init == "torch") {
    k <- 1 / sqrt(H)
    u <- function(r, c) matrix(stats::runif(r * c, -k, k), r, c)
    return(list(weight_ih = u(G * H, inp), weight_hh = u(G * H, H),
                bias_ih = u(G * H, 1L), bias_hh = u(G * H, 1L)))
  }
  limit <- sqrt(6 / (inp + G * H))
  w_ih <- matrix(stats::runif(G * H * inp, -limit, limit), G * H, inp)
  w_hh <- do.call(rbind, lapply(seq_len(G), function(g) .ag_orthogonal(H, H)))
  b_ih <- matrix(0, G * H, 1L)
  if (kind == "lstm") b_ih[H + seq_len(H), 1L] <- 1      # forget gate
  list(weight_ih = w_ih, weight_hh = w_hh, bias_ih = b_ih, bias_hh = matrix(0, G * H, 1L))
}

# The incoming state: zeros when NULL, then the copies in `starts` zeroed.
.ag_rnn_state <- function(kind, state, starts, H, B, device) {
  names_ <- if (kind == "lstm") c("h", "c") else "h"
  if (is.null(state)) {
    # On the device of the input, B from the input (so it follows a batch size
    # that changes between calls). On the GPU the zeros are made there (one
    # ggml_fill node): no upload, so an ag_capture recording needs no h0.
    return(stats::setNames(lapply(names_, function(n)
      if (device == "gpu") .ag_wrap_result(.ag_gpu_zeros(H, B), device)
      else ag_tensor(matrix(0, H, B), device = device)), names_))
  }
  if (!is.list(state) || !all(names_ %in% names(state)))
    stop(sprintf("ag_%s_cell: state must be list(%s)", kind,
                 paste(names_, collapse = ", ")), call. = FALSE)
  state <- lapply(state[names_], function(s)
    if (is_ag_tensor(s)) s else ag_tensor(as.matrix(s), device = device))
  if (is_ag_tensor(starts)) {
    # a [1, B] tensor (e.g. an input of ag_capture_step): s - s * starts, no
    # host constant in the graph. The values are not checked on the device:
    # they must be 0/1 (a fraction would reset the state partly).
    sd <- .ag_dim(if (starts$device == "gpu") .ag_operand(starts) else .ag_data(starts))
    if (!identical(as.integer(sd), c(1L, as.integer(B))))
      stop(sprintf("ag_%s_cell: a starts tensor must be [1, %d], got [%s]", kind, B,
                   paste(sd, collapse = ", ")), call. = FALSE)
    return(lapply(state, function(s) ag_sub(s, ag_mul(s, starts))))
  }
  if (!is.null(starts)) {
    st <- as.numeric(starts)
    if (length(st) != B || anyNA(st) || any(st != 0 & st != 1))
      stop(sprintf("ag_%s_cell: starts must be %d values 0/1 (one per column)", kind, B),
           call. = FALSE)
    if (any(st == 1)) {
      keep <- ag_tensor(matrix(rep(1 - st, each = H), H, B), device = device)
      state <- lapply(state, function(s) ag_mul(s, keep))
    }
  }
  state
}

.ag_lstm_step <- function(env, x, state, H) {
  g <- ag_add(ag_add(ag_matmul(env$weight_ih, x), env$bias_ih),
              ag_add(ag_matmul(env$weight_hh, state$h), env$bias_hh))
  i  <- ag_sigmoid(ag_slice_rows(g, 0L, H))
  f  <- ag_sigmoid(ag_slice_rows(g, H, H))
  gg <- ag_tanh(ag_slice_rows(g, 2L * H, H))
  o  <- ag_sigmoid(ag_slice_rows(g, 3L * H, H))
  c_new <- ag_add(ag_mul(f, state$c), ag_mul(i, gg))
  list(h = ag_mul(o, ag_tanh(c_new)), c = c_new)
}

.ag_gru_step <- function(env, x, state, H) {
  gi <- ag_add(ag_matmul(env$weight_ih, x), env$bias_ih)
  gh <- ag_add(ag_matmul(env$weight_hh, state$h), env$bias_hh)
  part <- function(m, k) ag_slice_rows(m, k * H, H)
  r <- ag_sigmoid(ag_add(part(gi, 0L), part(gh, 0L)))
  z <- ag_sigmoid(ag_add(part(gi, 1L), part(gh, 1L)))
  n <- ag_tanh(ag_add(part(gi, 2L), ag_mul(r, part(gh, 2L))))
  # (1 - z) n + z h = n + z (h - n)
  list(h = ag_add(n, ag_mul(z, ag_sub(state$h, n))))
}

#' @export
print.ag_lstm_cell <- function(x, ...) {
  cat(sprintf("ag_lstm_cell(%d -> %d) | init=%s\n", x$input_size, x$hidden_size, x$init))
  invisible(x)
}

#' @export
print.ag_gru_cell <- function(x, ...) {
  cat(sprintf("ag_gru_cell(%d -> %d) | init=%s\n", x$input_size, x$hidden_size, x$init))
  invisible(x)
}

#' @export
print.ag_lstm <- function(x, ...) {
  cat(sprintf("ag_lstm(%d -> %d) | init=%s\n", x$input_size, x$hidden_size, x$cell$init))
  invisible(x)
}

#' @export
print.ag_gru <- function(x, ...) {
  cat(sprintf("ag_gru(%d -> %d) | init=%s\n", x$input_size, x$hidden_size, x$cell$init))
  invisible(x)
}
