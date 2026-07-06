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

  # Tear down Vulkan explicitly at process exit, while the loader/ICD .so files
  # are still mapped, to avoid the flaky exit-time segfault from the loader's
  # static-destruction order (see R_ggml_vk_shutdown / ggml_backend_vk_shutdown).
  # .onUnload alone is NOT enough: a plain `Rscript -e '...'` exits via an
  # implicit q() WITHOUT unloading the namespace, so .onUnload never runs — which
  # is exactly the crashing case. A finalizer with onexit = TRUE fires on normal
  # process exit too. shutdown is idempotent, so running via both paths is safe.
  reg.finalizer(.ggmlr_state, function(e) {
    try(.Call("R_ggml_vk_shutdown", PACKAGE = "ggmlR"), silent = TRUE)
  }, onexit = TRUE)

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

  # Free the process-lifetime meta buffer-type cache while the DLL is still
  # loaded, so valgrind doesn't report its contexts as leaked at exit.
  try(.Call("R_ggml_backend_meta_free_cached_bufts", PACKAGE = "ggmlR"),
      silent = TRUE)

  # Tear down the Vulkan instance + devices explicitly while the DLL and the
  # Vulkan loader / Mesa ICD .so files are still mapped. Otherwise the static
  # vk_instance is destroyed at process exit in an order the C runtime does not
  # coordinate with the loader's own static destructors, giving a flaky segfault
  # in unmapped memory *after* results are already returned. Doing it here removes
  # that race. Safe: devices are shared_ptr, so a device still held by a live
  # backend (e.g. in llamaR/sd2R) is not destroyed until that backend is freed.
  try(.Call("R_ggml_vk_shutdown", PACKAGE = "ggmlR"), silent = TRUE)
}
