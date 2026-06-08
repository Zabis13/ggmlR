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
# The ggml_marshal_model()/ggml_unmarshal_model() helpers below cover sequential
# and functional models. Autograd (ag_sequential) learner models are marshaled
# separately via the M2 state-dict path (ag_save_model/ag_load_model); see the
# marshal_inner()/unmarshal_inner() helpers further down.
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
#     - model:         the trained model — sequential, functional, OR
#                      ag_sequential (autograd)
#     - class_names:   (classif only) character vector of class levels
#     - n_features:    integer
#     - feature_names: character vector
#     - ag_rebuild_fn: (autograd only) zero-arg closure rebuilding the module
#                      architecture; captured by the learner at fit time, used by
#                      the M2 state-dict marshal path (ag_save_model/ag_load_model)
#
# Sequential/functional models marshal via ggml_marshal_model(); autograd models
# marshal via ag_save_model() (state dict). See marshal_inner()/unmarshal_inner().
# ---------------------------------------------------------------------------

# Marshal the inner model, dispatching on its API. Returns a tagged payload that
# unmarshal_inner() can reverse. Autograd (ag_sequential) modules use the M2
# state-dict path (ag_save_model) and require a zero-arg rebuild closure that the
# learner captured at fit time in `model$ag_rebuild_fn`.
marshal_inner <- function(model, learner_label) {
  inner <- model$model
  if (inherits(inner, c("ggml_sequential_model", "ggml_functional_model"))) {
    return(list(api = "seq", payload = ggml_marshal_model(inner)))
  }
  if (inherits(inner, "ag_sequential")) {
    rebuild_fn <- model$ag_rebuild_fn
    if (is.null(rebuild_fn) || !is.function(rebuild_fn)) {
      stop(learner_label, ": cannot marshal this autograd model because no ",
           "rebuild function was captured at fit time (this can happen if the ",
           "model was constructed outside the learner's `.train()`).",
           call. = FALSE)
    }
    dir  <- tempfile(pattern = "ggmlR_ag_marshal_")
    dir.create(dir, recursive = TRUE, mode = "0700")
    on.exit(unlink(dir, recursive = TRUE, force = TRUE), add = TRUE)
    file <- file.path(dir, "model.rds")
    ag_save_model(inner, file, model_fn = rebuild_fn)
    bytes <- readBin(file, what = "raw", n = file.info(file)$size)
    return(list(api = "autograd", payload = bytes))
  }
  stop(learner_label, ": cannot marshal a model of class '",
       paste(class(inner), collapse = "/"),
       "'. Supported: sequential, functional, autograd (ag_sequential).",
       call. = FALSE)
}

# Reverse marshal_inner(): reconstruct the inner model from a tagged payload.
unmarshal_inner <- function(tagged) {
  if (identical(tagged$api, "autograd")) {
    dir  <- tempfile(pattern = "ggmlR_ag_unmarshal_")
    dir.create(dir, recursive = TRUE, mode = "0700")
    on.exit(unlink(dir, recursive = TRUE, force = TRUE), add = TRUE)
    file <- file.path(dir, "model.rds")
    writeBin(tagged$payload, file)
    return(ag_load_model(file))
  }
  # sequential / functional
  ggml_unmarshal_model(tagged$payload)
}

#' @noRd
marshal_model.classif_ggml_model <- function(model, inplace = FALSE, ...) {
  payload <- list(
    inner         = marshal_inner(model, "LearnerClassifGGML"),
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
  out <- list(
    model         = unmarshal_inner(payload$inner),
    class_names   = payload$class_names,
    n_features    = payload$n_features,
    feature_names = payload$feature_names
  )
  class(out) <- c("classif_ggml_model", "list")
  out
}

#' @noRd
marshal_model.regr_ggml_model <- function(model, inplace = FALSE, ...) {
  payload <- list(
    inner         = marshal_inner(model, "LearnerRegrGGML"),
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
  out <- list(
    model         = unmarshal_inner(payload$inner),
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
