# ============================================================================
# Marshal helpers for mlr3 integration
#
# Container format (version 1):
#   list(
#     format        = "ggmlR.marshal",
#     version       = 1L,
#     api           = "sequential" | "functional",
#     ggmlR_version = <package_version>,
#     R_version     = <R_version>,
#     created       = <POSIXct>,
#     sha256        = <hex string>,
#     payload       = <raw vector: bytes of the RDS produced by ggml_save_model>
#   )
#
# Only sequential and functional models are supported. Autograd modules return
# NULL from ggml_marshal_model() and learners expose marshaled = FALSE for them.
# ============================================================================

GGML_MARSHAL_FORMAT  <- "ggmlR.marshal"
GGML_MARSHAL_VERSION <- 1L

#' Marshal a ggmlR model to an in-memory container
#'
#' Serializes a trained sequential or functional ggmlR model into a
#' self-describing raw container suitable for transport between R sessions or
#' parallel workers (e.g. for \pkg{mlr3} parallel resampling and tuning).
#'
#' The container wraps the bytes produced by \code{\link{ggml_save_model}}
#' together with a format tag, schema version, package/R versions, a SHA-256
#' integrity checksum, and a timestamp. Autograd modules are \strong{not}
#' supported in this version and cause the function to signal an error; the
#' mlr3 learners catch this and fall back to \code{marshaled = FALSE}.
#'
#' @param model A compiled \code{ggml_sequential_model} or
#'   \code{ggml_functional_model}.
#' @return A named list with class \code{"ggmlR_marshaled"} containing the
#'   serialized payload and metadata. Pass it to
#'   \code{\link{ggml_unmarshal_model}} to reconstruct the model.
#' @seealso \code{\link{ggml_unmarshal_model}}, \code{\link{ggml_save_model}}
#' @export
ggml_marshal_model <- function(model) {
  api <- if (inherits(model, "ggml_sequential_model")) {
    "sequential"
  } else if (inherits(model, "ggml_functional_model")) {
    "functional"
  } else {
    stop("ggml_marshal_model(): unsupported model class '",
         paste(class(model), collapse = "/"),
         "'. Only sequential and functional models can be marshaled.")
  }

  backend_str <- if (!is.null(model$compilation$cpu_backend)) "gpu" else "cpu"

  # Unique per-call tmpdir to avoid collisions in parallel workers
  dir <- tempfile(pattern = "ggmlR_marshal_")
  dir.create(dir, recursive = TRUE, mode = "0700")
  on.exit(unlink(dir, recursive = TRUE, force = TRUE), add = TRUE)

  file <- file.path(dir, "model.rds")
  ggml_save_model(model, file)

  payload <- readBin(file, what = "raw", n = file.info(file)$size)

  sha <- if (requireNamespace("digest", quietly = TRUE)) {
    digest::digest(payload, algo = "sha256", serialize = FALSE)
  } else {
    NA_character_
  }

  out <- list(
    format        = GGML_MARSHAL_FORMAT,
    version       = GGML_MARSHAL_VERSION,
    api           = api,
    backend       = backend_str,
    ggmlR_version = utils::packageVersion("ggmlR"),
    R_version     = getRversion(),
    created       = Sys.time(),
    sha256        = sha,
    payload       = payload
  )
  class(out) <- "ggmlR_marshaled"
  out
}

#' Unmarshal a ggmlR model from an in-memory container
#'
#' Reconstructs a ggmlR model previously produced by
#' \code{\link{ggml_marshal_model}}. Validates the container's format tag,
#' schema version, and (if \pkg{digest} is installed) the SHA-256 checksum of
#' the payload before deserializing.
#'
#' @param x A \code{"ggmlR_marshaled"} container.
#' @param backend Backend selection passed through to
#'   \code{\link{ggml_load_model}}. Default \code{"auto"}.
#' @return A compiled ggmlR model object (sequential or functional).
#' @seealso \code{\link{ggml_marshal_model}}, \code{\link{ggml_load_model}}
#' @export
ggml_unmarshal_model <- function(x, backend = NULL) {
  if (!is.list(x) || !identical(x$format, GGML_MARSHAL_FORMAT)) {
    stop("ggml_unmarshal_model(): input is not a ggmlR marshaled container.")
  }
  if (!identical(x$version, GGML_MARSHAL_VERSION)) {
    stop("ggml_unmarshal_model(): unsupported container version ", x$version,
         " (this ggmlR supports version ", GGML_MARSHAL_VERSION, ").")
  }
  if (!is.raw(x$payload) || length(x$payload) == 0L) {
    stop("ggml_unmarshal_model(): container payload is empty or not raw.")
  }

  if (!is.na(x$sha256) && requireNamespace("digest", quietly = TRUE)) {
    got <- digest::digest(x$payload, algo = "sha256", serialize = FALSE)
    if (!identical(got, x$sha256)) {
      stop("ggml_unmarshal_model(): SHA-256 checksum mismatch - ",
           "container payload is corrupted.")
    }
  }

  resolved_backend <- backend %||% x$backend %||% "auto"
  if (identical(resolved_backend, "gpu")) resolved_backend <- "vulkan"

  dir <- tempfile(pattern = "ggmlR_unmarshal_")
  dir.create(dir, recursive = TRUE, mode = "0700")
  on.exit(unlink(dir, recursive = TRUE, force = TRUE), add = TRUE)

  file <- file.path(dir, "model.rds")
  writeBin(x$payload, file)

  ggml_load_model(file, backend = resolved_backend)
}

# ---------------------------------------------------------------------------
# S3 methods for mlr3's marshal_model / unmarshal_model generics
#
# These are registered lazily in .onLoad() via registerS3method(), so that
# ggmlR does not need to import mlr3. The generics themselves live in mlr3
# and are only visible when mlr3 is loaded.
#
# Object shape expected by the classif/regr methods:
#   self$model in the learner is a list with class "classif_ggml_model" or
#   "regr_ggml_model" containing:
#     - model:         the compiled ggml sequential/functional model
#     - class_names:   (classif only) character vector of class levels
#     - n_features:    integer
#     - feature_names: character vector
# ---------------------------------------------------------------------------

#' @noRd
marshal_model.classif_ggml_model <- function(model, inplace = FALSE, ...) {
  inner <- model$model
  if (!inherits(inner, c("ggml_sequential_model", "ggml_functional_model"))) {
    stop("LearnerClassifGGML: cannot marshal a model of class '",
         paste(class(inner), collapse = "/"),
         "'. Only sequential and functional ggmlR models are supported ",
         "(autograd models cannot be transported to parallel workers in v1).",
         call. = FALSE)
  }
  marshaled <- ggml_marshal_model(inner)
  payload <- list(
    marshaled     = marshaled,
    class_names   = model$class_names,
    n_features    = model$n_features,
    feature_names = model$feature_names
  )
  structure(
    list(marshaled = payload, packages = "ggmlR"),
    class = c("classif_ggml_model_marshaled", "list_marshaled", "marshaled")
  )
}

#' @noRd
unmarshal_model.classif_ggml_model_marshaled <- function(model, inplace = FALSE, ...) {
  payload <- model$marshaled
  inner <- ggml_unmarshal_model(payload$marshaled)
  out <- list(
    model         = inner,
    class_names   = payload$class_names,
    n_features    = payload$n_features,
    feature_names = payload$feature_names
  )
  class(out) <- c("classif_ggml_model", "list")
  out
}

#' @noRd
marshal_model.regr_ggml_model <- function(model, inplace = FALSE, ...) {
  inner <- model$model
  if (!inherits(inner, c("ggml_sequential_model", "ggml_functional_model"))) {
    stop("LearnerRegrGGML: cannot marshal a model of class '",
         paste(class(inner), collapse = "/"),
         "'. Only sequential and functional ggmlR models are supported ",
         "(autograd models cannot be transported to parallel workers in v1).",
         call. = FALSE)
  }
  marshaled <- ggml_marshal_model(inner)
  payload <- list(
    marshaled     = marshaled,
    n_features    = model$n_features,
    feature_names = model$feature_names
  )
  structure(
    list(marshaled = payload, packages = "ggmlR"),
    class = c("regr_ggml_model_marshaled", "list_marshaled", "marshaled")
  )
}

#' @noRd
unmarshal_model.regr_ggml_model_marshaled <- function(model, inplace = FALSE, ...) {
  payload <- model$marshaled
  inner <- ggml_unmarshal_model(payload$marshaled)
  out <- list(
    model         = inner,
    n_features    = payload$n_features,
    feature_names = payload$feature_names
  )
  class(out) <- c("regr_ggml_model", "list")
  out
}

#' @export
print.ggmlR_marshaled <- function(x, ...) {
  cat("<ggmlR marshaled model>\n")
  cat("  api:           ", x$api, "\n", sep = "")
  cat("  backend:       ", x$backend %||% "unknown", "\n", sep = "")
  cat("  format:        ", x$format, " v", x$version, "\n", sep = "")
  cat("  ggmlR version: ", format(x$ggmlR_version), "\n", sep = "")
  cat("  R version:     ", format(x$R_version), "\n", sep = "")
  cat("  created:       ", format(x$created), "\n", sep = "")
  cat("  payload size:  ", length(x$payload), " bytes\n", sep = "")
  if (!is.na(x$sha256)) {
    cat("  sha256:        ", substr(x$sha256, 1L, 16L), "...\n", sep = "")
  }
  invisible(x)
}
