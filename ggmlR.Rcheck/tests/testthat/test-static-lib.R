# Tests for static library (libggml.a) used by downstream packages via LinkingTo

test_that("libggml.a exists in installed package", {
  lib_dir <- system.file("lib", package = "ggmlR")
  expect_true(nzchar(lib_dir), info = "inst/lib directory should exist")

  lib_path <- file.path(lib_dir, "libggml.a")
  expect_true(file.exists(lib_path), info = "libggml.a should exist")
  expect_gt(file.size(lib_path), 0, label = "libggml.a size")
})
