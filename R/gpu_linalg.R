# GPU linear algebra as a drop-in for R's %*% / crossprod / tcrossprod ---------
#
# Goal: let a user accelerate an ordinary R matrix multiply on the Vulkan GPU
# without rewriting their code. `ggml_matmul(A, B)` takes R matrices and returns
# an R matrix (GPU inside, CPU fallback outside); `as_gpu_matrix()` wraps a
# matrix so that `%*%`, `crossprod()` and `tcrossprod()` dispatch to the GPU.
#
# Precision: R's %*% accumulates in f64; a GPU offers f32 at best, so the GPU
# path is a fast *approximate* multiply, never bit-for-bit with %*%. prec = "f32"
# (the default) requests the f32 accumulation kernel (GGML_PREC_F32), but some
# Vulkan drivers accumulate mul_mat in f16 regardless (e.g. RADV / Mesa), landing
# at ~1e-3 relative error either way; drivers with true f32 accumulation do
# better. prec = "f16" only lowers precision further, for speed. The gap is the
# unavoidable cost of the GPU and is documented on each function.
#
# Dispatch: device = "auto" (default) sends a multiply to the GPU only when a
# Vulkan device is present AND the problem is large enough to amortise the
# host<->VRAM copy (small multiplies are faster on the CPU); otherwise it runs on
# the CPU. device = "gpu" forces the GPU (still falls back to CPU if none), and
# device = "cpu" forces the CPU. Falling back is always silent — a missing or
# failing GPU degrades to the correct CPU result, never an error.

# Below this many fused multiply-adds (m * n * k) a multiply stays on the CPU
# under device = "auto": the GPU's fixed host<->VRAM transfer cost dominates for
# small problems. Chosen conservatively; override per call with device =.
.GGMLR_GPU_MATMUL_MIN_FLOP <- 2e7   # ~ 270^3, a few hundred square

# internal: is a live Vulkan GPU usable right now? (compiled in + a device)
.ggmlr_gpu_usable <- function() {
  isTRUE(tryCatch(
    ggml_vulkan_available() && ggml_vulkan_device_count() > 0L,
    error = function(e) FALSE))
}

# internal: decide the concrete backend for one multiply given its FLOP count.
# want is "auto" | "gpu" | "cpu"; returns "gpu" or "cpu".
.ggmlr_linalg_backend <- function(want, flop) {
  want <- match.arg(want, c("auto", "gpu", "cpu"))
  if (want == "cpu") return("cpu")
  if (!.ggmlr_gpu_usable()) return("cpu")           # no GPU -> CPU (silent)
  if (want == "gpu") return("gpu")
  if (flop >= .GGMLR_GPU_MATMUL_MIN_FLOP) "gpu" else "cpu"   # auto: size gate
}

# internal: run C = t(src0) %*% src1 on the GPU, where src0 and src1 share their
# first (contraction) dimension. This is the raw ggml_mul_mat contract; the
# public ops below arrange their operands into this shape with the fewest
# transposes. prec = "f32" forces the f32 accumulation kernel (default); "f16"
# leaves the faster, lower-precision default kernel. Returns an R matrix.
.ggmlr_gpu_mul_mat <- function(src0, src1, out_shape, prec = "f32") {
  # Restore the caller's device: ag_device() is process-global state, so leaking
  # "gpu" here silently reroutes every later ag_* op (and its dtype) for the
  # rest of the session.
  orig_device <- ag_default_device()
  on.exit(tryCatch(ag_device(orig_device), error = function(e) NULL), add = TRUE)
  ag_device("gpu")
  hook <- if (identical(prec, "f32"))
    function(node) .Call("R_ggml_mul_mat_set_prec", node, GGML_PREC_F32,
                         PACKAGE = "ggmlR") else NULL
  .ag_run_op(
    op_fn     = function(ctx, ptrs) ggml_mul_mat(ctx, ptrs[[1L]], ptrs[[2L]]),
    inputs    = list(src0, src1),
    out_shape = out_shape,
    dtype     = "f32",
    node_hook = hook
  )
}

# internal: honest-f64 C = A %*% B on the GPU via the matmul_f64.comp kernel
# (dispatched directly, not through the ggml graph — ggml has no fp64 mul_mat).
# Uploads/downloads double with no float conversion, so the result matches CPU
# double to machine precision (~1e-15). Returns an R matrix, or NULL to fall back
# (no live Vulkan backend, or the device lacks the shaderFloat64 feature so the
# pipeline was never created and the dispatch returns failure).
.ggmlr_gpu_matmul_f64 <- function(A, B) {
  orig_device <- ag_default_device()
  on.exit(tryCatch(ag_device(orig_device), error = function(e) NULL), add = TRUE)
  ok <- tryCatch({ ag_device("gpu"); TRUE }, error = function(e) FALSE)
  if (!ok) return(NULL)
  backend <- .ag_device_state$backend
  if (is.null(backend) || !ggml_vulkan_is_backend(backend)) return(NULL)

  M <- nrow(A); K <- ncol(A); N <- ncol(B)
  # the kernel is row-major; R matrices are column-major, so t(A)/t(B) give the
  # row-major streams. The M*N result comes back row-major -> read as [N,M] and
  # transpose to the [M,N] R matrix.
  cvec <- tryCatch(
    .Call("R_ggml_matmul_f64", backend, as.double(t(A)), as.double(t(B)),
          as.integer(M), as.integer(N), as.integer(K), PACKAGE = "ggmlR"),
    error = function(e) NULL)
  if (is.null(cvec)) return(NULL)
  t(matrix(cvec, N, M))
}

# ============================================================================
# Functional API — R matrix in, R matrix out
# ============================================================================

#' GPU matrix multiply (drop-in for \code{\%*\%})
#'
#' Computes \code{A \%*\% B} on the Vulkan GPU and returns an ordinary R matrix,
#' with a transparent CPU fallback. A drop-in accelerator for a large matrix
#' multiply: no autograd, no wrapper types required — plain matrices in and out.
#'
#' Precision: R multiplies in double precision (f64); a GPU offers f32 at best,
#' so the GPU result never matches R to full double precision. \code{prec = "f32"}
#' (the default) requests the f32 accumulation kernel, but how close the result
#' actually lands depends on the Vulkan driver: some accumulate \code{mul_mat} in
#' f16 regardless (e.g. RADV / Mesa), giving a relative error around
#' \code{1e-3} either way. Treat the GPU path as a fast, approximate multiply —
#' typically \code{1e-3}, better on drivers with true f32 accumulation — not a
#' bit-for-bit replacement for \code{\%*\%}. \code{prec = "f16"} only ever lowers
#' precision; use it when speed matters more than the last digits.
#'
#' @param A,B Numeric matrices with \code{ncol(A) == nrow(B)}.
#' @param device \code{"auto"} (default: GPU when present and the multiply is
#'   large enough to amortise the transfer, else CPU), \code{"gpu"} (force GPU,
#'   still falls back to CPU if none), or \code{"cpu"}.
#' @param prec \code{"f32"} (default; requests f32 accumulation, ~\code{1e-3}
#'   relative error or better depending on the driver) or \code{"f16"} (faster,
#'   never more precise). Only affects the GPU path.
#' @return The product \code{A \%*\% B} as a numeric \code{matrix}.
#' @seealso \code{\link{ggml_crossprod}}, \code{\link{ggml_tcrossprod}},
#'   \code{\link{as_gpu_matrix}}
#' @examples
#' A <- matrix(rnorm(4), 2); B <- matrix(rnorm(4), 2)
#' ggml_matmul(A, B, device = "cpu")
#' @export
ggml_matmul <- function(A, B, device = "auto", prec = "f32") {
  A <- .ggmlr_as_dmat(A); B <- .ggmlr_as_dmat(B)
  if (ncol(A) != nrow(B))
    stop(sprintf("non-conformable: ncol(A)=%d != nrow(B)=%d", ncol(A), nrow(B)),
         call. = FALSE)
  m <- nrow(A); k <- ncol(A); n <- ncol(B)
  backend <- .ggmlr_linalg_backend(device, as.double(m) * n * k)

  out <- if (backend == "gpu")
    # A %*% B = t(t(A)) %*% B: src0 = t(A) [k,m], src1 = B [k,n] -> [m,n]
    tryCatch(.ggmlr_gpu_mul_mat(t(A), B, c(m, n), prec),
             error = function(e) NULL)
  else NULL
  if (is.null(out)) out <- A %*% B                  # CPU (fallback or requested)

  .ggmlr_set_dimnames(out, rownames(A), colnames(B))
}

#' GPU double-precision matrix multiply
#'
#' Computes \code{A \%*\% B} on the Vulkan GPU in \strong{full double precision}
#' (fp64), returning an ordinary R matrix, with a transparent CPU fallback.
#' Unlike \code{\link{ggml_matmul}} (which is f32/f16 and approximate), this
#' matches R's \code{\%*\%} to machine precision (~\code{1e-15}), because the
#' kernel accumulates in \code{double} throughout.
#'
#' fp64 on the GPU is only worthwhile on hardware with fast double throughput:
#' data-centre cards (NVIDIA Tesla P100/V100, AMD Instinct) run fp64 at 1:2 of
#' fp32, whereas consumer GPUs cripple it (RDNA ~1:16, GeForce ~1:64), where the
#' CPU (multithreaded double BLAS) is usually faster. Use this when you need
#' bit-accurate double results on a capable GPU, or for numerically sensitive
#' work where the f32 path's error is unacceptable; otherwise prefer
#' \code{ggml_matmul}. Requires the \code{shaderFloat64} device feature — without
#' it (or without a GPU), the multiply silently falls back to the CPU.
#'
#' @param A,B Numeric matrices with \code{ncol(A) == nrow(B)}.
#' @param device \code{"auto"} (default: GPU when present, fp64-capable and the
#'   multiply is large enough to amortise the transfer, else CPU), \code{"gpu"}
#'   (force GPU, still falls back to CPU if fp64 is unavailable), or \code{"cpu"}.
#' @return The product \code{A \%*\% B} as a numeric \code{matrix}, accurate to
#'   double precision.
#' @seealso \code{\link{ggml_matmul}} (faster f32/f16 approximate path).
#' @examples
#' A <- matrix(rnorm(4), 2); B <- matrix(rnorm(4), 2)
#' ggml_matmul_f64(A, B, device = "cpu")
#' @export
ggml_matmul_f64 <- function(A, B, device = "auto") {
  A <- .ggmlr_as_dmat(A); B <- .ggmlr_as_dmat(B)
  if (ncol(A) != nrow(B))
    stop(sprintf("non-conformable: ncol(A)=%d != nrow(B)=%d", ncol(A), nrow(B)),
         call. = FALSE)
  m <- nrow(A); k <- ncol(A); n <- ncol(B)
  backend <- .ggmlr_linalg_backend(device, as.double(m) * n * k)

  out <- if (backend == "gpu")
    .ggmlr_gpu_matmul_f64(A, B)                     # NULL if no GPU / no fp64
  else NULL
  if (is.null(out)) out <- A %*% B                  # CPU (fallback or requested)

  .ggmlr_set_dimnames(out, rownames(A), colnames(B))
}

#' GPU cross product (drop-in for \code{crossprod})
#'
#' Computes \code{t(A) \%*\% B} (or \code{t(A) \%*\% A} when \code{B} is
#' \code{NULL}) on the Vulkan GPU, returning an R matrix, with a CPU fallback.
#' See \code{\link{ggml_matmul}} for the precision and dispatch notes.
#'
#' @param A A numeric matrix.
#' @param B A numeric matrix with \code{nrow(B) == nrow(A)}, or \code{NULL}
#'   (default) for \code{t(A) \%*\% A}.
#' @param device,prec See \code{\link{ggml_matmul}}.
#' @return \code{t(A) \%*\% B} as a numeric \code{matrix}.
#' @export
ggml_crossprod <- function(A, B = NULL, device = "auto", prec = "f32") {
  A <- .ggmlr_as_dmat(A)
  B <- if (is.null(B)) A else .ggmlr_as_dmat(B)
  if (nrow(A) != nrow(B))
    stop(sprintf("non-conformable: nrow(A)=%d != nrow(B)=%d", nrow(A), nrow(B)),
         call. = FALSE)
  # t(A) %*% B: src0 = A [m,ka], src1 = B [m,kb] share ne[0]=m -> [ka,kb].
  ka <- ncol(A); kb <- ncol(B); m <- nrow(A)
  backend <- .ggmlr_linalg_backend(device, as.double(ka) * kb * m)

  out <- if (backend == "gpu")
    tryCatch(.ggmlr_gpu_mul_mat(A, B, c(ka, kb), prec), error = function(e) NULL)
  else NULL
  if (is.null(out)) out <- crossprod(A, B)

  .ggmlr_set_dimnames(out, colnames(A), colnames(B))
}

#' GPU transposed cross product (drop-in for \code{tcrossprod})
#'
#' Computes \code{A \%*\% t(B)} (or \code{A \%*\% t(A)} when \code{B} is
#' \code{NULL}) on the Vulkan GPU, returning an R matrix, with a CPU fallback.
#' See \code{\link{ggml_matmul}} for the precision and dispatch notes.
#'
#' @param A A numeric matrix.
#' @param B A numeric matrix with \code{ncol(B) == ncol(A)}, or \code{NULL}
#'   (default) for \code{A \%*\% t(A)}.
#' @param device,prec See \code{\link{ggml_matmul}}.
#' @return \code{A \%*\% t(B)} as a numeric \code{matrix}.
#' @export
ggml_tcrossprod <- function(A, B = NULL, device = "auto", prec = "f32") {
  A <- .ggmlr_as_dmat(A)
  B <- if (is.null(B)) A else .ggmlr_as_dmat(B)
  if (ncol(A) != ncol(B))
    stop(sprintf("non-conformable: ncol(A)=%d != ncol(B)=%d", ncol(A), ncol(B)),
         call. = FALSE)
  # A %*% t(B) = t(t(A)) %*% t(B): src0 = t(A) [k,ma], src1 = t(B) [k,mb] -> [ma,mb]
  ma <- nrow(A); mb <- nrow(B); k <- ncol(A)
  backend <- .ggmlr_linalg_backend(device, as.double(ma) * mb * k)

  out <- if (backend == "gpu")
    tryCatch(.ggmlr_gpu_mul_mat(t(A), t(B), c(ma, mb), prec),
             error = function(e) NULL)
  else NULL
  if (is.null(out)) out <- tcrossprod(A, B)

  .ggmlr_set_dimnames(out, rownames(A), rownames(B))
}

# internal: set result dimnames only when at least one side is non-NULL, so an
# unnamed A %*% B returns a matrix with no dimnames (absent), exactly as base R
# does — attaching list(NULL, NULL) would spuriously differ from the reference.
.ggmlr_set_dimnames <- function(out, rn, cn) {
  if (!is.null(rn) || !is.null(cn)) dimnames(out) <- list(rn, cn)
  out
}

# internal: coerce input to a plain double matrix (accepts ggml_matrix, dense,
# or a vector treated as a 1-column matrix like base R does).
.ggmlr_as_dmat <- function(x) {
  if (methods::is(x, "ggml_matrix")) x <- x@data
  if (!is.matrix(x)) x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

# ============================================================================
# S4 wrapper — as_gpu_matrix(A) %*% B routes the multiply to the GPU
# ============================================================================
#
# A thin tag over a plain matrix. It does NOT override base::`%*%` globally
# (that would be surprising and fragile); instead, wrapping one operand opts that
# expression into the GPU path via S4 dispatch, so `as_gpu_matrix(A) %*% B`
# accelerates without touching any other code.

#' A GPU-backed matrix wrapper
#'
#' An S4 class holding an ordinary numeric matrix, tagged so that \code{\%*\%},
#' \code{crossprod()} and \code{tcrossprod()} run on the GPU (see
#' \code{\link{ggml_matmul}}). Wrap a matrix with \code{\link{as_gpu_matrix}}.
#'
#' @slot data The underlying numeric matrix.
#' @slot device Dispatch preference: \code{"auto"}, \code{"gpu"} or \code{"cpu"}.
#' @slot prec Precision of the GPU path: \code{"f32"} or \code{"f16"}.
#' @param x,y Operands; at least one is a \code{ggml_matrix} (the other is
#'   coerced with \code{as.matrix}). For \code{crossprod}/\code{tcrossprod},
#'   \code{y} may be missing.
#' @param object A \code{ggml_matrix} (for the \code{show} method).
#' @return \code{\%*\%}, \code{crossprod} and \code{tcrossprod} return a plain
#'   numeric matrix (the GPU result); \code{dim} an integer vector.
#' @name ggml_matrix-class
#' @rdname ggml_matrix-class
#' @exportClass ggml_matrix
methods::setClass("ggml_matrix",
  representation(data = "matrix", device = "character", prec = "character"))

#' Wrap a matrix so multiplies run on the GPU
#'
#' Tags a matrix as GPU-backed. In an expression like
#' \code{as_gpu_matrix(A) \%*\% B}, the multiply dispatches to
#' \code{\link{ggml_matmul}} (Vulkan GPU with CPU fallback) instead of R's
#' built-in. Only one operand needs wrapping. \code{crossprod()} and
#' \code{tcrossprod()} on a wrapped matrix likewise use the GPU.
#'
#' @param x A numeric matrix (or something coercible with \code{as.matrix}).
#' @param device \code{"auto"} (default), \code{"gpu"} or \code{"cpu"}; see
#'   \code{\link{ggml_matmul}}.
#' @param prec \code{"f32"} (default) or \code{"f16"}; see \code{\link{ggml_matmul}}.
#' @return A \code{\linkS4class{ggml_matrix}}.
#' @examples
#' A <- matrix(rnorm(9), 3)
#' B <- matrix(rnorm(9), 3)
#' as_gpu_matrix(A, device = "cpu") %*% B
#' @export
as_gpu_matrix <- function(x, device = "auto", prec = "f32") {
  x <- .ggmlr_as_dmat(x)
  device <- match.arg(device, c("auto", "gpu", "cpu"))
  prec   <- match.arg(prec, c("f32", "f16"))
  methods::new("ggml_matrix", data = x, device = device, prec = prec)
}

#' Extract the underlying matrix from a ggml_matrix
#' @param x A \code{\linkS4class{ggml_matrix}}.
#' @param ... Ignored; present for S3 \code{as.matrix} compatibility.
#' @return The plain numeric matrix.
#' @export
as.matrix.ggml_matrix <- function(x, ...) x@data

# resolve the dispatch preference / precision from whichever operand carries it.
.ggmlr_mm_opts <- function(a, b) {
  gm <- if (methods::is(a, "ggml_matrix")) a else b
  list(device = gm@device, prec = gm@prec)
}

#' @rdname ggml_matrix-class
#' @export
methods::setMethod("%*%", signature("ggml_matrix", "ANY"), function(x, y) {
  o <- .ggmlr_mm_opts(x, y)
  ggml_matmul(x@data, .ggmlr_as_dmat(y), device = o$device, prec = o$prec)
})

#' @rdname ggml_matrix-class
#' @export
methods::setMethod("%*%", signature("ANY", "ggml_matrix"), function(x, y) {
  o <- .ggmlr_mm_opts(x, y)
  ggml_matmul(.ggmlr_as_dmat(x), y@data, device = o$device, prec = o$prec)
})

#' @rdname ggml_matrix-class
#' @export
methods::setMethod("%*%", signature("ggml_matrix", "ggml_matrix"), function(x, y) {
  o <- .ggmlr_mm_opts(x, y)
  ggml_matmul(x@data, y@data, device = o$device, prec = o$prec)
})

# `crossprod`/`tcrossprod` are plain functions in base R until R 4.4.0 (only
# `%*%` was an implicit generic before that). Calling setMethod() on a plain
# function makes methods promote it to a generic and print an informational
# "Creating a new generic function ..." message at install time. Doing the
# setGeneric() explicitly ourselves is that same promotion, minus the message,
# and works identically on R 4.3/4.4/4.5: setGeneric() on an existing function
# captures base::crossprod as the default method, so every non-ggml_matrix call
# still runs the original base implementation. See ?setGeneric and ?groupGeneric.
#' @importFrom methods setGeneric
methods::setGeneric("crossprod")
methods::setGeneric("tcrossprod")

# The alias below is spelled out because these two generics are promoted by our
# own setGeneric() above (base's crossprod/tcrossprod are plain functions before
# R 4.4.0), unlike `%*%` which base already exports as an implicit generic. For
# a promoted generic R CMD check looks up the method under the signature as
# *written* ("ggml_matrix"), while roxygen emits the completed one
# ("ggml_matrix,ANY") -- so without this the check reports the method as
# undocumented even though the topic itself is documented.
#' @rdname ggml_matrix-class
#' @aliases crossprod,ggml_matrix-method
#' @export
methods::setMethod("crossprod", signature("ggml_matrix"), function(x, y) {
  b <- if (missing(y) || is.null(y)) NULL else .ggmlr_as_dmat(y)
  ggml_crossprod(x@data, b, device = x@device, prec = x@prec)
})

#' @rdname ggml_matrix-class
#' @aliases tcrossprod,ggml_matrix-method
#' @export
methods::setMethod("tcrossprod", signature("ggml_matrix"), function(x, y) {
  b <- if (missing(y) || is.null(y)) NULL else .ggmlr_as_dmat(y)
  ggml_tcrossprod(x@data, b, device = x@device, prec = x@prec)
})

#' @rdname ggml_matrix-class
#' @export
methods::setMethod("dim", "ggml_matrix", function(x) dim(x@data))

#' @rdname ggml_matrix-class
#' @importFrom methods show
#' @export
methods::setMethod("show", "ggml_matrix", function(object) {
  d <- dim(object@data)
  cat(sprintf("<ggml_matrix> %d x %d  device=%s prec=%s\n",
              d[1L], d[2L], object@device, object@prec))
  invisible(object)
})
