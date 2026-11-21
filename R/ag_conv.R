# 2-D convolution for the autograd engine, as one tape op ("conv2d").
#
# Layout. Autograd tensors are 2-D [features, batch], so an image is one column
# of C*H*W values ordered as ggml's ne = [W, H, C]: W fastest, then H, then C --
# R's array(x, c(W, H, C)), which is also as.vector() of a torch [C, H, W]
# image after aperm(c(3, 2, 1)). The output uses the same order,
# [OC*OH*OW, B], so an ag_linear after the convolution sees one flat feature
# vector per example. The weight is [OC, C*KH*KW] with the columns in the
# row-major order of a PyTorch kernel [OC, C, KH, KW] (KW fastest), which is
# also the row order of ggml's im2col columns -- one matmul, no reordering.
#
# Device path: im2col (F32) -> mul_mat -> permute, plus the per-channel bias, in
# one graph. Host path (CPU tensors): an index gather and rowsum(); it is the
# reference, not a fast path.

#' 2-D convolution
#'
#' Convolves a batch of images stored as columns. \code{x} is
#' \code{[C*H*W, B]}, each column an image flattened with W fastest, then H,
#' then C (\code{as.vector(array(img, c(W, H, C)))}); the result is
#' \code{[OC*OH*OW, B]} in the same order, ready for \code{\link{ag_linear}}.
#' The output size per axis is
#' \eqn{\lfloor (n + 2p - d(k - 1) - 1) / s \rfloor + 1}{floor((n + 2p - d(k - 1) - 1) / s) + 1}.
#'
#' The weight is \code{[OC, C*KH*KW]}, its columns in the row-major order of
#' a PyTorch kernel \code{[OC, C, KH, KW]}: a torch weight \code{w} (an R
#' array \code{[OC, C, KH, KW]}) becomes \code{matrix(aperm(w, c(1, 4, 3, 2)), OC)}.
#'
#' On the GPU the convolution is one graph (im2col, a matmul and a permute,
#' all F32; an f16/bf16 operand is cast first). On the CPU it is computed in R
#' by an index gather: a reference path for checking, not for speed. Scaling
#' the input (e.g. dividing pixels by 255) is the caller's.
#'
#' @param x ag_tensor (or matrix) \code{[C*H*W, B]}.
#' @param weight ag_tensor \code{[OC, C*KH*KW]}.
#' @param bias \code{NULL} or ag_tensor \code{[OC, 1]}, one per output channel.
#' @param in_shape \code{c(C, H, W)} of one image.
#' @param kernel_size \code{c(KH, KW)} or one number for both.
#' @param stride,padding,dilation \code{c(H, W)} or one number for both;
#'   padding is with zeros.
#' @return ag_tensor \code{[OC*OH*OW, B]}.
#' @seealso \code{\link{ag_conv2d_layer}}
#' @export
#' @examples
#' x <- ag_tensor(matrix(runif(3 * 8 * 8 * 2), 3 * 8 * 8, 2))   # 2 RGB 8x8 images
#' w <- ag_param(matrix(rnorm(4 * 3 * 3 * 3, sd = 0.1), 4))      # 4 filters 3x3
#' y <- ag_conv2d(x, w, in_shape = c(3, 8, 8), kernel_size = 3, padding = 1)
#' dim(as.matrix(y))                                             # 4*8*8 x 2
ag_conv2d <- function(x, weight, bias = NULL, in_shape, kernel_size,
                      stride = 1L, padding = 0L, dilation = 1L) {
  device <- .ag_result_device(x, weight)
  w_data <- if (device == "gpu") .ag_operand(weight) else .ag_data(weight)
  cf <- .ag_conv_conf(in_shape, kernel_size, stride, padding, dilation, .ag_nrow(w_data))
  if (!is_ag_tensor(x)) x <- ag_tensor(as.matrix(x), device = device)
  x_data <- if (device == "gpu") .ag_operand(x) else .ag_data(x)
  if (!.ag_is_handle(x_data) && !is.matrix(x_data)) x_data <- matrix(x_data, ncol = 1L)
  if (.ag_nrow(x_data) != cf$IC * cf$IH * cf$IW)
    stop(sprintf("ag_conv2d: x has %d rows, in_shape c(%d, %d, %d) needs %d",
                 .ag_nrow(x_data), cf$IC, cf$IH, cf$IW, cf$IC * cf$IH * cf$IW), call. = FALSE)
  if (.ag_ncol(w_data) != cf$K)
    stop(sprintf("ag_conv2d: weight has %d columns, C*KH*KW = %d", .ag_ncol(w_data), cf$K),
         call. = FALSE)
  b_data <- NULL
  if (!is.null(bias)) {
    b_data <- if (!is_ag_tensor(bias)) as.matrix(bias)
              else if (device == "gpu") .ag_operand(bias)
              else as.matrix(.ag_data(bias))
    if (!identical(as.integer(.ag_dim(b_data)), c(cf$OC, 1L)))
      stop(sprintf("ag_conv2d: bias must be [%d, 1]", cf$OC), call. = FALSE)
  }
  B <- .ag_ncol(x_data)

  if (device == "gpu") {
    out <- .ag_wrap_result(.ag_gpu_conv2d(x_data, w_data, b_data, cf), device)
  } else {
    out <- ag_tensor(.ag_conv2d_host(x_data, w_data, b_data, cf), device = device)
  }

  inputs <- list(x = x, weight = weight)
  if (!is.null(bias)) inputs$bias <- bias
  out$requires_grad <- any(vapply(inputs, function(i)
    is_ag_tensor(i) && isTRUE(i$requires_grad), logical(1L)))
  if (out$requires_grad) {
    x_snap <- x_data; w_snap <- w_data
    has_bias <- !is.null(bias)
    grad_fn <- function(grad_out) {
      g <- .ag_conv2d_host_grads(.ag_as_matrix(grad_out), .ag_as_matrix(x_snap),
                                 .ag_as_matrix(w_snap), cf, B)
      res <- list(x = g$x, weight = g$weight)
      if (has_bias) res$bias <- g$bias
      res
    }
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, inputs, op = "conv2d", x_snap = x_snap, w_snap = w_snap,
              conf = cf, batch = B, has_bias = has_bias)
  }
  out
}

#' Convolution layer
#'
#' A module around \code{\link{ag_conv2d}} for a fixed image size: the output
#' size is computed once and the layer refuses inputs of another size.
#'
#' The backward runs as a graph on the GPU (\code{im2col} is recomputed from
#' the input instead of being kept from the forward), so a training step with
#' convolutions can be recorded by \code{\link{ag_capture_step}}. There, as for
#' every input, the batch size is fixed by the first call: another size is an
#' error, not a re-recording (pad a short last minibatch).
#'
#' \code{init}:
#' \describe{
#'   \item{\code{"orthogonal"} (default)}{the weight \code{[OC, C*KH*KW]}
#'     orthogonal (as CleanRL / SB3 for Atari) times \code{gain}
#'     (\code{sqrt(2)} for ReLU), bias 0.}
#'   \item{\code{"torch"}}{weight and bias \eqn{U(-1/\sqrt{fan_{in}}, 1/\sqrt{fan_{in}})},
#'     \eqn{fan_{in} = C \cdot KH \cdot KW}, as PyTorch's default
#'     (\code{kaiming_uniform_(a = sqrt(5))}).}
#' }
#'
#' @param in_channels,out_channels Channels in and out.
#' @param kernel_size,stride,padding,dilation See \code{\link{ag_conv2d}}.
#' @param in_hw \code{c(H, W)} of the input images.
#' @param bias Use a per-channel bias (default \code{TRUE}).
#' @param init \code{"orthogonal"} or \code{"torch"}.
#' @param gain Scale of the orthogonal initialisation (ignored by \code{"torch"}).
#' @return An \code{ag_conv2d_layer} environment with \code{forward(x)},
#'   \code{parameters()} (\code{weight}, \code{bias}), \code{in_shape}
#'   \code{c(C, H, W)}, \code{out_shape} \code{c(OC, OH, OW)} and
#'   \code{out_features} (\code{OC*OH*OW}, the input size of the next linear
#'   layer).
#' @export
#' @examples
#' \donttest{
#' # the first layer of the Nature-DQN network on 4 stacked 84x84 frames
#' conv <- ag_conv2d_layer(4L, 32L, kernel_size = 8L, in_hw = c(84L, 84L), stride = 4L)
#' conv$out_shape        # 32 20 20
#' }
ag_conv2d_layer <- function(in_channels, out_channels, kernel_size, in_hw,
                            stride = 1L, padding = 0L, dilation = 1L, bias = TRUE,
                            init = c("orthogonal", "torch"), gain = sqrt(2)) {
  init <- match.arg(init)
  pos <- function(v) is.numeric(v) && length(v) == 1L && !is.na(v) && v >= 1 && v == round(v)
  if (!pos(in_channels) || !pos(out_channels))
    stop("ag_conv2d_layer: in_channels and out_channels must be positive whole numbers",
         call. = FALSE)
  if (!is.numeric(in_hw) || length(in_hw) != 2L)
    stop("ag_conv2d_layer: in_hw must be c(H, W)", call. = FALSE)
  in_shape <- c(as.integer(in_channels), as.integer(in_hw))
  OC <- as.integer(out_channels)
  cf <- .ag_conv_conf(in_shape, kernel_size, stride, padding, dilation, OC)
  if (!is.numeric(gain) || length(gain) != 1L || !is.finite(gain))
    stop("ag_conv2d_layer: gain must be a single finite number", call. = FALSE)

  if (init == "torch") {
    k <- 1 / sqrt(cf$K)
    w <- matrix(stats::runif(OC * cf$K, -k, k), OC, cf$K)
    b <- matrix(stats::runif(OC, -k, k), OC, 1L)
  } else {
    w <- gain * .ag_orthogonal(OC, cf$K)
    b <- matrix(0, OC, 1L)
  }

  env <- new.env(parent = emptyenv())
  env$in_shape     <- in_shape
  env$out_shape    <- c(OC, cf$OH, cf$OW)
  env$out_features <- OC * cf$OH * cf$OW
  env$kernel_size  <- c(cf$KH, cf$KW)
  env$stride       <- c(cf$sH, cf$sW)
  env$padding      <- c(cf$pH, cf$pW)
  env$dilation     <- c(cf$dH, cf$dW)
  env$init         <- init
  env$training     <- TRUE
  env$weight <- ag_param(w)
  env$bias   <- if (isTRUE(bias)) ag_param(b) else NULL

  env$forward <- function(x)
    ag_conv2d(x, env$weight, env$bias, in_shape = env$in_shape,
              kernel_size = env$kernel_size, stride = env$stride,
              padding = env$padding, dilation = env$dilation)

  env$parameters <- function()
    if (is.null(env$bias)) list(weight = env$weight)
    else list(weight = env$weight, bias = env$bias)

  class(env) <- c("ag_conv2d_layer", "ag_layer")
  env
}

#' @export
print.ag_conv2d_layer <- function(x, ...) {
  cat(sprintf("ag_conv2d_layer(%d -> %d, kernel %dx%d, stride %dx%d, padding %dx%d) | %s -> %s | init=%s\n",
              x$in_shape[1L], x$out_shape[1L], x$kernel_size[1L], x$kernel_size[2L],
              x$stride[1L], x$stride[2L], x$padding[1L], x$padding[2L],
              paste(x$in_shape, collapse = "x"), paste(x$out_shape, collapse = "x"), x$init))
  invisible(x)
}

# The geometry of one convolution, checked; H/W pairs in torch order (H first).
.ag_conv_conf <- function(in_shape, kernel_size, stride, padding, dilation, OC) {
  whole <- function(v, nm, min) {
    if (!is.numeric(v) || !(length(v) %in% 1:2) || anyNA(v) || any(v != round(v)) ||
        any(v < min))
      stop(sprintf("ag_conv2d: %s must be one or two whole numbers >= %d", nm, min),
           call. = FALSE)
    as.integer(rep_len(v, 2L))
  }
  if (!is.numeric(in_shape) || length(in_shape) != 3L || anyNA(in_shape) ||
      any(in_shape < 1) || any(in_shape != round(in_shape)))
    stop("ag_conv2d: in_shape must be c(C, H, W), positive whole numbers", call. = FALSE)
  k <- whole(kernel_size, "kernel_size", 1L)
  s <- whole(stride, "stride", 1L)
  p <- whole(padding, "padding", 0L)
  d <- whole(dilation, "dilation", 1L)
  in_shape <- as.integer(in_shape)
  osz <- function(n, k, s, p, d) (n + 2L * p - d * (k - 1L) - 1L) %/% s + 1L
  OH <- osz(in_shape[2L], k[1L], s[1L], p[1L], d[1L])
  OW <- osz(in_shape[3L], k[2L], s[2L], p[2L], d[2L])
  if (OH < 1L || OW < 1L)
    stop(sprintf(paste0("ag_conv2d: the output would be %d x %d: an input of %d x %d is ",
                        "smaller than the dilated kernel %d x %d plus padding"),
                 OH, OW, in_shape[2L], in_shape[3L], d[1L] * (k[1L] - 1L) + 1L,
                 d[2L] * (k[2L] - 1L) + 1L), call. = FALSE)
  list(IC = in_shape[1L], IH = in_shape[2L], IW = in_shape[3L], OC = as.integer(OC),
       KH = k[1L], KW = k[2L], sH = s[1L], sW = s[2L], pH = p[1L], pW = p[2L],
       dH = d[1L], dW = d[2L], OH = OH, OW = OW, K = in_shape[1L] * k[1L] * k[2L])
}

# ---- host (reference) path --------------------------------------------------

# Row of the padded image column (C*H*W + 1 = the zero) that each im2col entry
# reads, in column order: (kw, kh, ic) fastest to slowest, then (ow, oh).
# Cached per geometry.
.ag_conv_idx_cache <- new.env(parent = emptyenv())
.ag_conv_index <- function(cf) {
  key <- paste(unlist(cf), collapse = ",")
  idx <- .ag_conv_idx_cache[[key]]
  if (!is.null(idx)) return(idx)
  g <- expand.grid(kw = 0:(cf$KW - 1L), kh = 0:(cf$KH - 1L), ic = 0:(cf$IC - 1L),
                   ow = 0:(cf$OW - 1L), oh = 0:(cf$OH - 1L))
  ix <- g$ow * cf$sW - cf$pW + g$kw * cf$dW
  iy <- g$oh * cf$sH - cf$pH + g$kh * cf$dH
  inside <- ix >= 0L & ix < cf$IW & iy >= 0L & iy < cf$IH
  n_img <- cf$IC * cf$IH * cf$IW
  idx <- ifelse(inside, 1L + ix + cf$IW * (iy + cf$IH * g$ic), n_img + 1L)
  if (length(ls(.ag_conv_idx_cache)) > 64L) rm(list = ls(.ag_conv_idx_cache), envir = .ag_conv_idx_cache)
  assign(key, as.integer(idx), envir = .ag_conv_idx_cache)
  idx
}

# im2col of a batch: [K, OW*OH*B]
.ag_conv_cols <- function(x, cf) {
  idx <- .ag_conv_index(cf)
  xp <- rbind(x, 0)
  matrix(xp[idx, , drop = FALSE], cf$K)
}

.ag_conv2d_host <- function(x, w, b, cf) {
  B <- ncol(x)
  y <- w %*% .ag_conv_cols(x, cf)                                  # [OC, OW*OH*B]
  y <- matrix(aperm(array(y, c(cf$OC, cf$OW * cf$OH, B)), c(2L, 1L, 3L)),
              cf$OW * cf$OH * cf$OC, B)
  if (!is.null(b)) y <- y + rep(as.numeric(b), each = cf$OW * cf$OH)
  y
}

.ag_conv2d_host_grads <- function(g, x, w, cf, B) {
  P <- cf$OW * cf$OH
  gt <- matrix(aperm(array(g, c(P, cf$OC, B)), c(2L, 1L, 3L)), cf$OC, P * B)  # [OC, P*B]
  cols <- .ag_conv_cols(x, cf)
  dcols <- crossprod(w, gt)                                         # [K, P*B]
  n_img <- cf$IC * cf$IH * cf$IW
  grp <- rep(.ag_conv_index(cf), B) + rep((seq_len(B) - 1L) * (n_img + 1L), each = cf$K * P)
  s <- rowsum(as.numeric(dcols), grp)
  dxp <- numeric((n_img + 1L) * B)
  dxp[as.integer(rownames(s))] <- s[, 1L]
  list(x = matrix(dxp, n_img + 1L, B)[seq_len(n_img), , drop = FALSE],
       weight = tcrossprod(gt, cols),
       bias = matrix(rowSums(gt), cf$OC, 1L))
}

# ---- device path ------------------------------------------------------------

# Forward graph: x [C*H*W, B], w [OC, K], b [OC, 1] or NULL -> [OC*OH*OW, B].
.ag_conv2d_graph <- function(ctx, x, w, b, cf, B) {
  x4 <- ggml_reshape_4d(ctx, .ag_f32(ctx, x), cf$IW, cf$IH, cf$IC, B)
  # im2col reads only the kernel's KW, KH (and checks IC): a shape template
  kshape <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cf$KW, cf$KH, cf$IC, 1L)
  cols <- ggml_im2col(ctx, kshape, x4, cf$sW, cf$sH, cf$pW, cf$pH, cf$dW, cf$dH,
                      TRUE, GGML_TYPE_F32)                          # [K, OW, OH, B]
  cols <- ggml_reshape_2d(ctx, cols, cf$K, cf$OW * cf$OH * B)
  wt <- ggml_cont(ctx, ggml_transpose(ctx, .ag_f32(ctx, w)))       # [K, OC]
  y <- .ag_mul_mat(ctx, wt, cols)                                  # [OC, OW*OH*B]
  y <- ggml_reshape_4d(ctx, y, cf$OC, cf$OW, cf$OH, B)
  y <- ggml_cont(ctx, ggml_permute(ctx, y, 2L, 0L, 1L, 3L))         # [OW, OH, OC, B]
  if (!is.null(b))
    y <- ggml_add(ctx, y, ggml_reshape_4d(ctx, .ag_f32(ctx, b), 1L, 1L, cf$OC, 1L))
  ggml_reshape_2d(ctx, y, cf$OW * cf$OH * cf$OC, B)
}

.ag_gpu_conv2d <- function(x_data, w_data, b_data, cf) {
  B <- .ag_ncol(x_data)
  inputs <- list(x_data, w_data)
  if (!is.null(b_data)) inputs[[3L]] <- b_data
  # the graph above is ~16 tensors; .ag_run_op only reserves inputs + 4
  .ag_ctx_ensure(length(inputs) + 24L)
  .ag_run_op(
    op_fn = function(ctx, ptrs)
      .ag_conv2d_graph(ctx, ptrs[[1L]], ptrs[[2L]],
                       if (length(ptrs) > 2L) ptrs[[3L]], cf, B),
    inputs    = inputs,
    out_shape = c(cf$OC * cf$OH * cf$OW, B),
    resident  = .ag_is_handle(x_data) || .ag_is_handle(w_data)
  )
}

# ---- transposed convolution -------------------------------------------------
#
# A transposed convolution is the input gradient of a convolution. With the
# DUAL convolution cd -- input = the transpose's output [OC, OH, OW], output =
# the transpose's input [IC, IH, IW], weight [IC, OC*KH*KW] -- the three passes
# swap roles and reuse the conv2d pieces, no new kernels:
#   forward      y  = dx of cd applied to x        (mul_mat, im2col_back)
#   input grad   dx = forward of cd applied to g   (im2col, mul_mat)
#   weight grad  dW = x_t %*% t(im2col(g))         (out_prod)

#' 2-D transposed convolution
#'
#' The transposed ("fractionally strided") convolution of PyTorch's
#' \code{nn.ConvTranspose2d}: the input gradient of \code{\link{ag_conv2d}},
#' used to upsample in decoders. Images are columns as in
#' \code{\link{ag_conv2d}}: \code{x} is \code{[C*H*W, B]}, each column W
#' fastest, then H, then C; the result is \code{[OC*OH*OW, B]} in the same
#' order. The output size per axis is PyTorch's
#' \deqn{(n - 1) s - 2p + d (k - 1) + output\_padding + 1}{(n - 1) * s - 2 * p + d * (k - 1) + output_padding + 1}
#' so kernel 4, stride 2, padding 1 (or kernel 3, stride 2, padding 1,
#' output_padding 1) doubles H and W.
#'
#' The weight is \code{[C, OC*KH*KW]}, the layout of a PyTorch
#' \code{ConvTranspose2d} weight \code{[C, OC, KH, KW]} with its columns in
#' row-major order: a torch weight \code{w} (an R array \code{[C, OC, KH, KW]})
#' becomes \code{matrix(aperm(w, c(1, 4, 3, 2)), C)}. The number of output
#' channels is \code{ncol(weight) / (KH*KW)}.
#'
#' Differences from PyTorch, deliberate: \code{groups} must be 1 (grouped
#' transposed convolutions are an error), and \code{output_padding} must be
#' smaller than \code{stride} (PyTorch also allows values below
#' \code{dilation}).
#'
#' On the GPU the forward is one graph (a matmul and \code{im2col_back}) and
#' the backward runs in the graph backward, so the op can be part of an
#' \code{\link{ag_capture_step}} recording. On the CPU it is computed in R: a
#' reference path for checking, not for speed.
#'
#' @param x ag_tensor (or matrix) \code{[C*H*W, B]}.
#' @param weight ag_tensor \code{[C, OC*KH*KW]}.
#' @param bias \code{NULL} or ag_tensor \code{[OC, 1]}.
#' @param in_shape \code{c(C, H, W)} of one input image.
#' @param kernel_size \code{c(KH, KW)} or one number for both.
#' @param stride,padding,dilation \code{c(H, W)} or one number for both.
#' @param output_padding \code{c(H, W)} or one number: extra rows / columns
#'   at the bottom / right of the output, to pick one of the input sizes a
#'   strided convolution maps to the same output size. Smaller than
#'   \code{stride}.
#' @param groups Must be 1.
#' @return ag_tensor \code{[OC*OH*OW, B]}.
#' @seealso \code{\link{ag_conv_transpose2d_layer}}, \code{\link{ag_conv2d}}
#' @export
#' @examples
#' x <- ag_tensor(matrix(runif(8 * 4 * 4 * 2), 8 * 4 * 4, 2))   # 2 images 8x4x4
#' w <- ag_param(matrix(rnorm(8 * 3 * 4 * 4, sd = 0.1), 8))     # 8 -> 3 channels, 4x4
#' y <- ag_conv_transpose2d(x, w, in_shape = c(8, 4, 4), kernel_size = 4,
#'                          stride = 2, padding = 1)
#' dim(as.matrix(y))                                            # 3*8*8 x 2
ag_conv_transpose2d <- function(x, weight, bias = NULL, in_shape, kernel_size,
                                stride = 1L, padding = 0L, output_padding = 0L,
                                dilation = 1L, groups = 1L) {
  if (!identical(as.numeric(groups), 1))
    stop("ag_conv_transpose2d: only groups = 1 is supported", call. = FALSE)
  device <- .ag_result_device(x, weight)
  w_data <- if (device == "gpu") .ag_operand(weight) else .ag_data(weight)
  cd <- .ag_convT_conf(in_shape, kernel_size, stride, padding, output_padding,
                       dilation, .ag_nrow(w_data), .ag_ncol(w_data))
  if (!is_ag_tensor(x)) x <- ag_tensor(as.matrix(x), device = device)
  x_data <- if (device == "gpu") .ag_operand(x) else .ag_data(x)
  if (!.ag_is_handle(x_data) && !is.matrix(x_data)) x_data <- matrix(x_data, ncol = 1L)
  n_in <- cd$OC * cd$OH * cd$OW
  if (.ag_nrow(x_data) != n_in)
    stop(sprintf("ag_conv_transpose2d: x has %d rows, in_shape c(%d, %d, %d) needs %d",
                 .ag_nrow(x_data), cd$OC, cd$OH, cd$OW, n_in), call. = FALSE)
  b_data <- NULL
  if (!is.null(bias)) {
    b_data <- if (!is_ag_tensor(bias)) as.matrix(bias)
              else if (device == "gpu") .ag_operand(bias)
              else as.matrix(.ag_data(bias))
    if (!identical(as.integer(.ag_dim(b_data)), c(cd$IC, 1L)))
      stop(sprintf("ag_conv_transpose2d: bias must be [%d, 1]", cd$IC), call. = FALSE)
  }
  B <- .ag_ncol(x_data)

  if (device == "gpu") {
    out <- .ag_wrap_result(.ag_gpu_conv_transpose2d(x_data, w_data, b_data, cd), device)
  } else {
    out <- ag_tensor(.ag_convT_host(x_data, w_data, b_data, cd), device = device)
  }

  inputs <- list(x = x, weight = weight)
  if (!is.null(bias)) inputs$bias <- bias
  out$requires_grad <- any(vapply(inputs, function(i)
    is_ag_tensor(i) && isTRUE(i$requires_grad), logical(1L)))
  if (out$requires_grad) {
    x_snap <- x_data; w_snap <- w_data
    has_bias <- !is.null(bias)
    grad_fn <- function(grad_out) {
      g <- .ag_convT_host_grads(.ag_as_matrix(grad_out), .ag_as_matrix(x_snap),
                                .ag_as_matrix(w_snap), cd, B)
      res <- list(x = g$x, weight = g$weight)
      if (has_bias) res$bias <- g$bias
      res
    }
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, inputs, op = "conv_transpose2d", x_snap = x_snap,
              w_snap = w_snap, conf = cd, batch = B, has_bias = has_bias)
  }
  out
}

#' Transposed convolution layer
#'
#' A module around \code{\link{ag_conv_transpose2d}} for a fixed image size,
#' the counterpart of \code{\link{ag_conv2d_layer}}.
#'
#' \code{init}:
#' \describe{
#'   \item{\code{"orthogonal"} (default)}{the weight \code{[C, OC*KH*KW]}
#'     orthogonal times \code{gain}, bias 0.}
#'   \item{\code{"torch"}}{weight and bias \eqn{U(-1/\sqrt{fan_{in}}, 1/\sqrt{fan_{in}})}
#'     with PyTorch's \eqn{fan_{in}} for \code{ConvTranspose2d}, taken from the
#'     SECOND axis of its weight \code{[C, OC, KH, KW]}:
#'     \eqn{fan_{in} = OC \cdot KH \cdot KW}.}
#' }
#'
#' @param in_channels,out_channels Channels in and out.
#' @param kernel_size,stride,padding,output_padding,dilation See
#'   \code{\link{ag_conv_transpose2d}}.
#' @param in_hw \code{c(H, W)} of the input images.
#' @param bias Use a per-channel bias (default \code{TRUE}).
#' @param init \code{"orthogonal"} or \code{"torch"}.
#' @param gain Scale of the orthogonal initialisation (ignored by \code{"torch"}).
#' @return An \code{ag_conv_transpose2d_layer} environment with
#'   \code{forward(x)}, \code{parameters()} (\code{weight}, \code{bias}),
#'   \code{in_shape} \code{c(C, H, W)}, \code{out_shape} \code{c(OC, OH, OW)}
#'   and \code{out_features} (\code{OC*OH*OW}).
#' @export
#' @examples
#' \donttest{
#' # a decoder step: 32 x 8 x 8 -> 16 x 16 x 16
#' up <- ag_conv_transpose2d_layer(32L, 16L, kernel_size = 4L, in_hw = c(8L, 8L),
#'                                 stride = 2L, padding = 1L)
#' up$out_shape        # 16 16 16
#' }
ag_conv_transpose2d_layer <- function(in_channels, out_channels, kernel_size, in_hw,
                                      stride = 1L, padding = 0L, output_padding = 0L,
                                      dilation = 1L, bias = TRUE,
                                      init = c("orthogonal", "torch"), gain = 1) {
  init <- match.arg(init)
  pos <- function(v) is.numeric(v) && length(v) == 1L && !is.na(v) && v >= 1 && v == round(v)
  if (!pos(in_channels) || !pos(out_channels))
    stop("ag_conv_transpose2d_layer: in_channels and out_channels must be positive whole numbers",
         call. = FALSE)
  if (!is.numeric(in_hw) || length(in_hw) != 2L)
    stop("ag_conv_transpose2d_layer: in_hw must be c(H, W)", call. = FALSE)
  if (!is.numeric(gain) || length(gain) != 1L || !is.finite(gain))
    stop("ag_conv_transpose2d_layer: gain must be a single finite number", call. = FALSE)
  IC <- as.integer(in_channels); OC <- as.integer(out_channels)
  in_shape <- c(IC, as.integer(in_hw))
  k  <- as.integer(rep_len(kernel_size, 2L))
  cd <- .ag_convT_conf(in_shape, kernel_size, stride, padding, output_padding,
                       dilation, IC, OC * k[1L] * k[2L])

  if (init == "torch") {
    b <- 1 / sqrt(cd$K)                       # fan_in = OC*KH*KW (weight axis 2)
    w <- matrix(stats::runif(IC * cd$K, -b, b), IC, cd$K)
    bs <- matrix(stats::runif(OC, -b, b), OC, 1L)
  } else {
    w <- gain * .ag_orthogonal(IC, cd$K)
    bs <- matrix(0, OC, 1L)
  }

  env <- new.env(parent = emptyenv())
  env$in_shape       <- in_shape
  env$out_shape      <- c(OC, cd$IH, cd$IW)
  env$out_features   <- OC * cd$IH * cd$IW
  env$kernel_size    <- c(cd$KH, cd$KW)
  env$stride         <- c(cd$sH, cd$sW)
  env$padding        <- c(cd$pH, cd$pW)
  env$output_padding <- cd$out_pad
  env$dilation       <- c(cd$dH, cd$dW)
  env$init           <- init
  env$training       <- TRUE
  env$weight <- ag_param(w)
  env$bias   <- if (isTRUE(bias)) ag_param(bs) else NULL

  env$forward <- function(x)
    ag_conv_transpose2d(x, env$weight, env$bias, in_shape = env$in_shape,
                        kernel_size = env$kernel_size, stride = env$stride,
                        padding = env$padding, output_padding = env$output_padding,
                        dilation = env$dilation)

  env$parameters <- function()
    if (is.null(env$bias)) list(weight = env$weight)
    else list(weight = env$weight, bias = env$bias)

  class(env) <- c("ag_conv_transpose2d_layer", "ag_layer")
  env
}

#' @export
print.ag_conv_transpose2d_layer <- function(x, ...) {
  cat(sprintf("ag_conv_transpose2d_layer(%d -> %d, kernel %dx%d, stride %dx%d, padding %dx%d) | %s -> %s | init=%s\n",
              x$in_shape[1L], x$out_shape[1L], x$kernel_size[1L], x$kernel_size[2L],
              x$stride[1L], x$stride[2L], x$padding[1L], x$padding[2L],
              paste(x$in_shape, collapse = "x"), paste(x$out_shape, collapse = "x"), x$init))
  invisible(x)
}

# The DUAL conv2d of a transposed convolution, checked. Its IC/IH/IW are the
# transpose's output, its OC/OH/OW the transpose's input, K = OC_t*KH*KW.
.ag_convT_conf <- function(in_shape, kernel_size, stride, padding, output_padding,
                           dilation, w_rows, w_cols) {
  whole <- function(v, nm, min) {
    if (!is.numeric(v) || !(length(v) %in% 1:2) || anyNA(v) || any(v != round(v)) ||
        any(v < min))
      stop(sprintf("ag_conv_transpose2d: %s must be one or two whole numbers >= %d", nm, min),
           call. = FALSE)
    as.integer(rep_len(v, 2L))
  }
  if (!is.numeric(in_shape) || length(in_shape) != 3L || anyNA(in_shape) ||
      any(in_shape < 1) || any(in_shape != round(in_shape)))
    stop("ag_conv_transpose2d: in_shape must be c(C, H, W), positive whole numbers",
         call. = FALSE)
  in_shape <- as.integer(in_shape)
  k  <- whole(kernel_size, "kernel_size", 1L)
  s  <- whole(stride, "stride", 1L)
  p  <- whole(padding, "padding", 0L)
  op <- whole(output_padding, "output_padding", 0L)
  d  <- whole(dilation, "dilation", 1L)
  if (any(op >= s))
    stop("ag_conv_transpose2d: output_padding must be smaller than stride", call. = FALSE)
  if (w_rows != in_shape[1L])
    stop(sprintf("ag_conv_transpose2d: weight has %d rows, in_shape has %d channels",
                 w_rows, in_shape[1L]), call. = FALSE)
  if (w_cols %% (k[1L] * k[2L]) != 0L)
    stop(sprintf("ag_conv_transpose2d: weight has %d columns, not a multiple of KH*KW = %d",
                 w_cols, k[1L] * k[2L]), call. = FALSE)
  OC <- as.integer(w_cols %/% (k[1L] * k[2L]))
  osz <- function(n, k, s, p, d, op) (n - 1L) * s - 2L * p + d * (k - 1L) + op + 1L
  OH <- osz(in_shape[2L], k[1L], s[1L], p[1L], d[1L], op[1L])
  OW <- osz(in_shape[3L], k[2L], s[2L], p[2L], d[2L], op[2L])
  if (OH < 1L || OW < 1L)
    stop(sprintf("ag_conv_transpose2d: the output would be %d x %d (padding too large)",
                 OH, OW), call. = FALSE)
  cd <- .ag_conv_conf(c(OC, OH, OW), k, s, p, d, in_shape[1L])
  # op < s makes the dual map OH back onto exactly IH
  stopifnot(cd$OH == in_shape[2L], cd$OW == in_shape[3L])
  cd$out_pad <- op
  cd
}

# im2col_back on the host: dcols [K, P*B] summed into images [C*H*W, B] of cd's
# input geometry.
.ag_conv_scatter <- function(dcols, cd, B) {
  P <- cd$OW * cd$OH
  n_img <- cd$IC * cd$IH * cd$IW
  grp <- rep(.ag_conv_index(cd), B) + rep((seq_len(B) - 1L) * (n_img + 1L), each = cd$K * P)
  s <- rowsum(as.numeric(dcols), grp)
  dxp <- numeric((n_img + 1L) * B)
  dxp[as.integer(rownames(s))] <- s[, 1L]
  matrix(dxp, n_img + 1L, B)[seq_len(n_img), , drop = FALSE]
}

# [C*H*W, B] images (cd's output geometry) -> [C, P*B], P = OW*OH
.ag_convT_channels <- function(x, cd, B) {
  P <- cd$OW * cd$OH
  matrix(aperm(array(x, c(P, cd$OC, B)), c(2L, 1L, 3L)), cd$OC, P * B)
}

.ag_convT_host <- function(x, w, b, cd) {
  B <- ncol(x)
  y <- .ag_conv_scatter(crossprod(w, .ag_convT_channels(x, cd, B)), cd, B)
  if (!is.null(b)) y <- y + rep(as.numeric(b), each = cd$IH * cd$IW)
  y
}

.ag_convT_host_grads <- function(g, x, w, cd, B) {
  cols <- .ag_conv_cols(g, cd)                                      # [K, P*B]
  list(x = .ag_conv2d_host(g, w, NULL, cd),
       weight = tcrossprod(.ag_convT_channels(x, cd, B), cols),
       bias = matrix(rowSums(matrix(colSums(matrix(g, cd$IH * cd$IW)), cd$IC)),
                     cd$IC, 1L))
}

# Forward graph: x [IC_t*IH*IW, B], w [IC_t, K], b [OC_t, 1] or NULL
# -> [OC_t*OH*OW, B].
.ag_convT_graph <- function(ctx, x, w, b, cd, B) {
  P  <- cd$OW * cd$OH
  x4 <- ggml_reshape_4d(ctx, .ag_f32(ctx, x), cd$OW, cd$OH, cd$OC, B)
  xt <- ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, x4, 1L, 2L, 0L, 3L)),
                        cd$OC, P * B)                              # [IC_t, P*B]
  dcols <- .ag_mul_mat(ctx, .ag_f32(ctx, w), xt)                   # [K, P*B]
  kshape <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cd$KW, cd$KH, cd$IC, 1L)
  y <- ggml_im2col_back(ctx, ggml_reshape_4d(ctx, dcols, cd$K, cd$OW, cd$OH, B),
                        kshape, c(cd$IW, cd$IH, cd$IC, B),
                        cd$sW, cd$sH, cd$pW, cd$pH, cd$dW, cd$dH)  # [OW_t, OH_t, OC_t, B]
  if (!is.null(b))
    y <- ggml_add(ctx, y, ggml_reshape_4d(ctx, .ag_f32(ctx, b), 1L, 1L, cd$IC, 1L))
  ggml_reshape_2d(ctx, y, cd$IW * cd$IH * cd$IC, B)
}

.ag_gpu_conv_transpose2d <- function(x_data, w_data, b_data, cd) {
  B <- .ag_ncol(x_data)
  inputs <- list(x_data, w_data)
  if (!is.null(b_data)) inputs[[3L]] <- b_data
  .ag_ctx_ensure(length(inputs) + 24L)
  .ag_run_op(
    op_fn = function(ctx, ptrs)
      .ag_convT_graph(ctx, ptrs[[1L]], ptrs[[2L]],
                      if (length(ptrs) > 2L) ptrs[[3L]], cd, B),
    inputs    = inputs,
    out_shape = c(cd$IC * cd$IH * cd$IW, B),
    resident  = .ag_is_handle(x_data) || .ag_is_handle(w_data)
  )
}
