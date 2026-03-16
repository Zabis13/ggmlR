# Limit threads during CRAN checks to avoid NOTE about CPU time >> elapsed time
if (identical(Sys.getenv("_R_CHECK_LIMIT_CORES_"), "TRUE")) {
  ggml_set_n_threads(2L)
}
