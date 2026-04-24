# Package-level state
.ggmlr_state <- new.env(parent = emptyenv())

# Silence R CMD check NOTEs about rlang::expr() placeholders in parsnip::set_pred() args
utils::globalVariables(c("object", "new_data", "self", "super", "private"))

.register_mlr3 <- function(...) {
  # Ignore arguments: when invoked via setHook(packageEvent(..., "onLoad")),
  # R passes (pkgname, pkgpath) where pkgpath is a full filesystem path —
  # using it as a namespace name would crash asNamespace().
  if (!requireNamespace("mlr3",    quietly = TRUE) ||
      !requireNamespace("paradox", quietly = TRUE) ||
      !requireNamespace("R6",      quietly = TRUE)) {
    return(invisible(FALSE))
  }

  ns <- asNamespace("ggmlR")

  # Build R6 classes and store in package state (not namespace bindings).
  .ggmlr_state$LearnerClassifGGML <- .make_LearnerClassifGGML()
  .ggmlr_state$LearnerRegrGGML    <- .make_LearnerRegrGGML()

  # S3 methods for mlr3's marshal_model / unmarshal_model generics.
  registerS3method("marshal_model",   "classif_ggml_model",
                   get("marshal_model.classif_ggml_model", envir = ns),
                   envir = asNamespace("mlr3"))
  registerS3method("unmarshal_model", "classif_ggml_model_marshaled",
                   get("unmarshal_model.classif_ggml_model_marshaled", envir = ns),
                   envir = asNamespace("mlr3"))
  registerS3method("marshal_model",   "regr_ggml_model",
                   get("marshal_model.regr_ggml_model", envir = ns),
                   envir = asNamespace("mlr3"))
  registerS3method("unmarshal_model", "regr_ggml_model_marshaled",
                   get("unmarshal_model.regr_ggml_model_marshaled", envir = ns),
                   envir = asNamespace("mlr3"))

  # Register learners idempotently in mlr3's dictionary.
  learners <- utils::getFromNamespace("mlr_learners", ns = "mlr3")
  if (!learners$has("classif.ggml")) {
    learners$add("classif.ggml", .ggmlr_state$LearnerClassifGGML)
  }
  if (!learners$has("regr.ggml")) {
    learners$add("regr.ggml", .ggmlr_state$LearnerRegrGGML)
  }

  invisible(TRUE)
}

.register_parsnip <- function(...) {
  if (!requireNamespace("parsnip", quietly = TRUE)) return(invisible(FALSE))
  tryCatch(make_mlp_ggml(),
           error = function(e) {
             message("ggmlR: could not register parsnip engine (",
                     conditionMessage(e), ")")
           })
  invisible(TRUE)
}

.onLoad <- function(libname, pkgname) {
  # Redirect GGML log messages through R's logging system.
  # The R callback suppresses DEBUG-level messages (scheduler realloc,
  # graph allocation internals) while forwarding INFO/WARN/ERROR.
  ggml_log_set_r()
  ggml_set_abort_callback_r()

  # Track whether backend message has been shown
  .ggmlr_state$backend_msg_shown <- FALSE

  # mlr3 / parsnip integration.
  # If already loaded: register immediately.
  # Otherwise: setHook fires when the package loads or attaches later.
  # .register_mlr3() / .register_parsnip() each guard with requireNamespace()
  # internally, so stale hook invocations are safe.
  if (isNamespaceLoaded("mlr3"))    .register_mlr3()
  if (isNamespaceLoaded("parsnip")) .register_parsnip()

  setHook(packageEvent("mlr3",    "onLoad"), function(...) .register_mlr3())
  setHook(packageEvent("mlr3",    "attach"), function(...) .register_mlr3())
  setHook(packageEvent("parsnip", "onLoad"), function(...) .register_parsnip())
  setHook(packageEvent("parsnip", "attach"), function(...) .register_parsnip())
}

.onUnload <- function(libpath) {
  if (requireNamespace("mlr3", quietly = TRUE)) {
    learners <- utils::getFromNamespace("mlr_learners", ns = "mlr3")
    if (learners$has("classif.ggml")) learners$remove("classif.ggml")
    if (learners$has("regr.ggml"))    learners$remove("regr.ggml")
  }
}
