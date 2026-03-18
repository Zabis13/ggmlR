library(testthat)
library(ggmlR)

heavy <- c(
  "normalization",
  "sequence-ops",
  "binary-operations",
  "glu",
  "nn-sequential",
  "matrix-ops",
  "callbacks",
  "onnx-quant",
  "cpu-backend",
  "onnx-elementwise",
  "onnx-norm",
  "onnx-misc",
  "onnx-logic",
  "onnx-gemm",
  "onnx-resize",
  "onnx-nn",
  "nn-functional",
  "activations-extended",
  "onnx-ops",
  "transformer-ops",
  "onnx-shape",
  "math-operations",
  "activations",
  "backend-extended",
  "helpers",
  "onnx-reduce"
)

on_cran <- !identical(Sys.getenv("NOT_CRAN"), "true")

test_dir <- if (dir.exists("testthat")) "testthat" else "tests/testthat"

if (on_cran) {
  message("--- RUNNING LIGHT TESTS ONLY ---")

  all_tests <- list.files(test_dir, pattern = "^test-.*\\.R$")
  all_names <- sub("^test-(.*)\\.R$", "\\1", all_tests)

  light_tests <- setdiff(all_names, heavy)
  message("Tests to run: ", paste(light_tests, collapse = ", "))

  if (length(light_tests) == 0) {
    # fallback
    test_check("ggmlR")
  } else {
    # grepl(filter, names)
    filter_regex <- paste(light_tests, collapse = "|")
    test_check("ggmlR", filter = filter_regex)
  }
} else {
  test_check("ggmlR")
}
