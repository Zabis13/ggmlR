# Tests that inst/include/ headers are in sync with src/ headers
# This catches ABI-breaking changes (e.g. GGML_MAX_DIMS) that would
# cause segfaults in downstream packages (sd2R etc.)

.ggmlr_src_dir <- function() {
  # tests/testthat/ -> tests/ -> package root -> src/
  pkg_root <- normalizePath(file.path(
    getSrcDirectory(function() {}), "..", ".."
  ), mustWork = FALSE)
  file.path(pkg_root, "src")
}

test_that("inst/include headers match src headers", {
  pkg_src     <- .ggmlr_src_dir()
  pkg_include <- system.file("include", package = "ggmlR")

  skip_if(!dir.exists(pkg_src),     "src/ directory not found")
  skip_if(!dir.exists(pkg_include), "inst/include/ directory not found")

  headers <- c(
    "ggml.h", "ggml-alloc.h", "ggml-backend.h", "ggml-backend-impl.h",
    "ggml-common.h", "ggml-cpp.h", "ggml-cpu.h", "ggml-impl.h",
    "ggml-opt.h", "ggml-quants.h", "ggml-threading.h",
    "ggml-vulkan.h", "gguf.h", "r_ggml_compat.h"
  )

  for (h in headers) {
    src_path <- file.path(pkg_src, h)
    inc_path <- file.path(pkg_include, h)
    if (!file.exists(src_path)) next
    expect_true(file.exists(inc_path),
                label = paste0("inst/include/", h, " exists"))
    expect_identical(readLines(src_path), readLines(inc_path),
                     label = paste0(h, " is in sync"))
  }
})

test_that("GGML_MAX_DIMS is consistent between src and installed headers", {
  pkg_src     <- file.path(.ggmlr_src_dir(), "ggml.h")
  pkg_include <- system.file("include/ggml.h", package = "ggmlR")

  skip_if(!file.exists(pkg_src), "src/ggml.h not found")
  skip_if(!nzchar(pkg_include),  "installed ggml.h not found")

  extract_max_dims <- function(path) {
    lines <- readLines(path)
    hits <- grep("^#define GGML_MAX_DIMS", lines, value = TRUE)
    m <- regmatches(hits, regexpr("\\d+$", hits))
    as.integer(m[1])
  }

  src_val <- extract_max_dims(pkg_src)
  inc_val <- extract_max_dims(pkg_include)

  expect_false(is.na(src_val), label = "GGML_MAX_DIMS found in src/ggml.h")
  expect_false(is.na(inc_val), label = "GGML_MAX_DIMS found in installed ggml.h")
  expect_equal(src_val, inc_val,
               label = "GGML_MAX_DIMS matches between src and installed headers")
})
