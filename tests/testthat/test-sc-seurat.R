# Single-cell adapter (Seurat / Bioconductor-style) tests.
#
# The matrix / dgCMatrix / registry paths run everywhere (no Seurat needed).
# Seurat object paths skip when Seurat is not installed. CPU backend is used for
# the numeric checks so they are deterministic and pass without a GPU; one extra
# test exercises the Vulkan path and skips when no GPU is present.

# --- small deterministic fixture: 40 genes x 150 cells ----------------------
sc_fixture <- function(seed = 11L) {
  set.seed(seed)
  X <- matrix(rnorm(40 * 150), nrow = 40)
  rownames(X) <- paste0("Gene", seq_len(40))
  colnames(X) <- paste0("Cell", seq_len(150))
  X
}

# engines flip the global device to "gpu"; restore it when this file ends so the
# state does not leak into later test files (see helper-device.R).
withr::defer(ag_device("cpu"), teardown_env())

# ---------------------------------------------------------------------------
# Contracts + registry
# ---------------------------------------------------------------------------
test_that("ggml_ops_registry declares the embed op with its required params", {
  reg <- ggml_ops_registry()
  expect_true("embed" %in% names(reg))
  entry <- ggml_ops_registry("embed")
  expect_equal(entry$op, "embed")
  expect_true("n_components" %in% entry$params)
  expect_null(ggml_ops_registry("does_not_exist"))
})

test_that("ggml_task validates inputs and prints", {
  X <- sc_fixture()
  task <- ggml_task("embed", X, params = list(n_components = 5))
  expect_s3_class(task, "ggml_task")
  expect_equal(task$op, "embed")
  expect_output(print(task), "ggml_task")

  expect_error(ggml_task("embed", "not a matrix"), "dense matrix")
  expect_error(ggml_task(c("a", "b"), X), "single operation")
})

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
test_that("ggml_extract.matrix returns a dense double matrix and subsets", {
  X <- sc_fixture()
  m <- ggml_extract(X)
  expect_true(is.matrix(m))
  expect_identical(storage.mode(m), "double")
  expect_equal(dim(m), c(40L, 150L))

  sub <- ggml_extract(X, genes = paste0("Gene", 1:10), cells = paste0("Cell", 1:20))
  expect_equal(dim(sub), c(10L, 20L))
})

test_that("ggml_extract.dgCMatrix materialises to dense only for the subset", {
  skip_if_not_installed("Matrix")
  X <- sc_fixture()
  S <- methods::as(X, "dgCMatrix")
  m <- ggml_extract(S, genes = paste0("Gene", 1:15))
  expect_true(is.matrix(m))
  expect_equal(dim(m), c(15L, 150L))
  expect_equal(unname(m), unname(X[1:15, ]))
})

# ---------------------------------------------------------------------------
# Dispatch + PCA engine (CPU = deterministic, compared against prcomp)
# ---------------------------------------------------------------------------
test_that("ggml_run embed on CPU matches base prcomp", {
  X <- sc_fixture()
  task <- ggml_task("embed", X, params = list(n_components = 5), device = "cpu")
  res  <- ggml_run(task)

  expect_s3_class(res, "ggml_result")
  expect_equal(dim(res$embedding), c(150L, 5L))
  expect_equal(res$metadata$backend, "cpu")

  pc <- prcomp(t(X), center = TRUE, scale. = FALSE)
  expect_equal(res$metadata$stdev, pc$sdev[1:5], tolerance = 1e-6)

  # scores match prcomp up to per-component sign
  s_ours <- res$embedding
  s_ref  <- pc$x[, 1:5]
  for (j in 1:5) if (sum(s_ours[, j] * s_ref[, j]) < 0) s_ours[, j] <- -s_ours[, j]
  expect_equal(unname(s_ours), unname(s_ref), tolerance = 1e-5)
})

test_that("ggml_run rejects unknown op and missing required params", {
  X <- sc_fixture()
  expect_error(ggml_run(ggml_task("embed", X)), "n_components")

  bad <- ggml_task("embed", X, params = list(n_components = 2))
  bad$op <- "no_such_op"
  expect_error(ggml_run(bad), "Unsupported op")

  expect_error(ggml_run("not a task"), "ggml_task")
})

test_that("device auto/cpu resolution is sane", {
  expect_equal(ggmlR:::.ggmlr_resolve_backend("cpu"), "cpu")
  resolved <- ggmlR:::.ggmlr_resolve_backend("auto")
  expect_true(resolved %in% c("cpu", "vulkan"))
})

test_that("CPU fallback: vulkan requested but no GPU -> cpu with a message", {
  # Pretend there is no GPU: the GPU-first promise is that we silently degrade.
  testthat::with_mocked_bindings(
    ggml_vulkan_available = function() FALSE,
    {
      expect_equal(ggmlR:::.ggmlr_resolve_backend("auto"), "cpu")
      expect_message(
        expect_equal(ggmlR:::.ggmlr_resolve_backend("vulkan"), "cpu"),
        "falling back to CPU"
      )
      # full dispatch still produces a valid result on the CPU
      X   <- sc_fixture()
      res <- suppressMessages(
        ggml_run(ggml_task("embed", X, params = list(n_components = 4),
                            device = "vulkan"))
      )
      expect_equal(res$metadata$backend, "cpu")
      expect_equal(dim(res$embedding), c(150L, 4L))
    },
    .package = "ggmlR"
  )
})

test_that("auto resolves to vulkan when a GPU is present", {
  testthat::with_mocked_bindings(
    ggml_vulkan_available = function() TRUE,
    {
      expect_equal(ggmlR:::.ggmlr_resolve_backend("auto"), "vulkan")
      expect_equal(ggmlR:::.ggmlr_resolve_backend("vulkan"), "vulkan")
    },
    .package = "ggmlR"
  )
})

# ---------------------------------------------------------------------------
# Vulkan path (skips without a GPU)
# ---------------------------------------------------------------------------
test_that("ggml_run embed on Vulkan agrees with CPU", {
  skip_if_not(isTRUE(tryCatch(ggml_vulkan_available(), error = function(e) FALSE)),
              "No Vulkan GPU")
  X   <- sc_fixture()
  gpu <- ggml_run(ggml_task("embed", X, params = list(n_components = 5), device = "vulkan"))
  cpu <- ggml_run(ggml_task("embed", X, params = list(n_components = 5), device = "cpu"))
  expect_equal(gpu$metadata$backend, "vulkan")
  expect_equal(gpu$metadata$stdev, cpu$metadata$stdev, tolerance = 1e-3)
})

# ---------------------------------------------------------------------------
# Seurat object path (skips without Seurat)
# ---------------------------------------------------------------------------

# shared Seurat fixture: 50 genes x 100 cells, normalized (data layer present)
seurat_fixture <- function(seed = 3L) {
  set.seed(seed)
  counts <- matrix(rpois(50 * 100, lambda = 5), nrow = 50)
  rownames(counts) <- paste0("Gene", seq_len(50))
  colnames(counts) <- paste0("Cell", seq_len(100))
  # Build the counts as a sparse dgCMatrix (the native Seurat format).
  # Passing a dense matrix makes CreateSeuratObject emit "Data is of class
  # matrix. Coercing to dgCMatrix." on every fixture build, spamming the test
  # log; coercing up front silences it and matches how Seurat stores counts.
  counts <- methods::as(counts, "dgCMatrix")
  so <- SeuratObject::CreateSeuratObject(counts = counts)
  Seurat::NormalizeData(so, verbose = FALSE)
}

# Same fixture forced onto the legacy v4 Assay model (not Assay5). This drives
# the GetAssayData extraction branch instead of LayerData. The caller is
# responsible for restoring options(); we set it inside and reset on exit.
seurat_fixture_v4 <- function(seed = 3L) {
  old <- options(Seurat.object.assay.version = "v3")
  on.exit(options(old))
  so <- seurat_fixture(seed)
  # guard: make sure we really got the legacy model, else the test is a no-op
  testthat::skip_if_not(inherits(so[["RNA"]], "Assay"),
                        "could not build a legacy v4 Assay")
  so
}

test_that("ggml_extract.Seurat reads the data layer and subsets (v5 LayerData)", {
  skip_if_not_installed("Seurat")
  skip_if_not_installed("SeuratObject")
  so <- seurat_fixture()

  m <- ggml_extract(so, layer = "data")
  expect_true(is.matrix(m))
  expect_identical(storage.mode(m), "double")
  expect_equal(dim(m), c(50L, 100L))          # genes x cells

  sub <- ggml_extract(so, layer = "data",
                      genes = paste0("Gene", 1:10), cells = paste0("Cell", 1:25))
  expect_equal(dim(sub), c(10L, 25L))

  # matches the layer pulled directly from the object
  ref <- as.matrix(SeuratObject::LayerData(so, assay = "RNA", layer = "data"))
  expect_equal(unname(m), unname(ref), tolerance = 1e-10)
})

test_that("ggml_extract.Seurat reads a legacy v4 Assay (GetAssayData branch)", {
  skip_if_not_installed("Seurat")
  skip_if_not_installed("SeuratObject")
  so <- seurat_fixture_v4()

  # this object must take the v4 branch, not the v5 LayerData one
  expect_false(ggmlR:::.ggmlr_object_is_v5(so, "RNA"))

  m <- ggml_extract(so, layer = "data")
  expect_true(is.matrix(m))
  expect_identical(storage.mode(m), "double")
  expect_equal(dim(m), c(50L, 100L))          # genes x cells

  # matches the layer pulled directly from the legacy assay
  ref <- as.matrix(SeuratObject::GetAssayData(so, assay = "RNA", layer = "data"))
  expect_equal(unname(m), unname(ref), tolerance = 1e-10)
})

test_that("RunGGML on a legacy v4 Seurat object matches prcomp end-to-end", {
  skip_if_not_installed("Seurat")
  skip_if_not_installed("SeuratObject")
  so <- seurat_fixture_v4()

  so2 <- RunGGML(so, op = "embed", n_components = 8L, device = "cpu",
                 reduction_name = "ggml")
  expect_true("ggml" %in% SeuratObject::Reductions(so2))
  emb <- SeuratObject::Embeddings(so2, "ggml")
  expect_equal(dim(emb), c(100L, 8L))

  # cells-as-rows PCA; PCs are sign-ambiguous so compare absolute correlation
  mat <- ggml_extract(so, layer = "data")
  ref <- prcomp(t(mat), center = TRUE, scale. = FALSE)$x[, 1:8]
  cors <- vapply(1:8, function(i) abs(cor(emb[, i], ref[, i])), numeric(1))
  expect_gt(min(cors), 0.999)
})

test_that("ggml_inject.Seurat writes a reduction and provenance metadata", {
  skip_if_not_installed("Seurat")
  skip_if_not_installed("SeuratObject")
  so <- seurat_fixture()

  mat <- ggml_extract(so, layer = "data")
  res <- ggml_run(ggml_task("embed", mat, params = list(n_components = 6),
                            device = "cpu"))
  so2 <- ggml_inject(so, res, reduction_name = "ggml")

  expect_true("ggml" %in% SeuratObject::Reductions(so2))
  emb <- SeuratObject::Embeddings(so2, "ggml")
  expect_equal(dim(emb), c(100L, 6L))
  expect_match(colnames(emb)[1], "^GGML_")
  expect_equal(SeuratObject::Misc(so2, "ggml_ggml")$backend, "cpu")

  expect_error(ggml_inject(so, "not a result"), "ggml_result")
})

test_that("RunGGML on a Seurat object writes a reduction end-to-end", {
  skip_if_not_installed("Seurat")
  skip_if_not_installed("SeuratObject")
  so <- seurat_fixture()

  so <- RunGGML(so, op = "embed", n_components = 8,
                reduction_name = "ggml", device = "cpu")

  expect_true("ggml" %in% SeuratObject::Reductions(so))
  emb <- SeuratObject::Embeddings(so, "ggml")
  expect_equal(dim(emb), c(100L, 8L))
  expect_match(colnames(emb)[1], "^GGML_")
  expect_length(SeuratObject::Stdev(so, "ggml"), 8L)

  misc <- SeuratObject::Misc(so, "ggml_ggml")
  expect_equal(misc$backend, "cpu")
})

test_that("RunGGML on a Seurat object runs on Vulkan (auto) and agrees with CPU", {
  skip_if_not_installed("Seurat")
  skip_if_not_installed("SeuratObject")
  skip_if_not(isTRUE(tryCatch(ggml_vulkan_available(), error = function(e) FALSE)),
              "No Vulkan GPU")
  so <- seurat_fixture()

  gpu <- RunGGML(so, op = "embed", n_components = 8,
                 reduction_name = "ggml", device = "auto")
  cpu <- RunGGML(so, op = "embed", n_components = 8,
                 reduction_name = "ggml", device = "cpu")

  expect_equal(SeuratObject::Misc(gpu, "ggml_ggml")$backend, "vulkan")
  expect_equal(
    SeuratObject::Stdev(gpu, "ggml"),
    SeuratObject::Stdev(cpu, "ggml"),
    tolerance = 1e-3
  )
})

# --- Seurat GPU paths (skip without a Vulkan GPU) ---------------------------

# shared guard for the Seurat + Vulkan tests below
skip_no_seurat_gpu <- function() {
  skip_if_not_installed("Seurat")
  skip_if_not_installed("SeuratObject")
  skip_if_not(isTRUE(tryCatch(ggml_vulkan_available(), error = function(e) FALSE)),
              "No Vulkan GPU")
}

test_that("RunGGML.Seurat with explicit device='vulkan' runs on the GPU", {
  skip_no_seurat_gpu()
  so <- seurat_fixture()

  so <- RunGGML(so, op = "embed", n_components = 8,
                reduction_name = "ggml", device = "vulkan")

  expect_equal(SeuratObject::Misc(so, "ggml_ggml")$backend, "vulkan")
  expect_true("ggml" %in% SeuratObject::Reductions(so))
  expect_equal(dim(SeuratObject::Embeddings(so, "ggml")), c(100L, 8L))
})

test_that("Vulkan PCA on the sparse Seurat data layer agrees with CPU", {
  skip_no_seurat_gpu()
  so <- seurat_fixture()

  # the v5 data layer is a real dgCMatrix with structural zeros
  raw <- SeuratObject::LayerData(so, assay = "RNA", layer = "data")
  expect_s4_class(raw, "dgCMatrix")

  gpu <- RunGGML(so, op = "embed", n_components = 8, device = "vulkan")
  cpu <- RunGGML(so, op = "embed", n_components = 8, device = "cpu")
  expect_equal(
    SeuratObject::Stdev(gpu, "ggml"),
    SeuratObject::Stdev(cpu, "ggml"),
    tolerance = 1e-3
  )
})

test_that("RunGGML.Seurat on Vulkan with a gene/cell subset is consistent with CPU", {
  skip_no_seurat_gpu()
  so <- seurat_fixture()

  genes <- paste0("Gene", 1:30)
  cells <- paste0("Cell", 1:60)
  gpu <- RunGGML(so, op = "embed", n_components = 5, device = "vulkan",
                 genes = genes, cells = cells)
  cpu <- RunGGML(so, op = "embed", n_components = 5, device = "cpu",
                 genes = genes, cells = cells)

  # only the selected cells are embedded
  expect_equal(nrow(SeuratObject::Embeddings(gpu, "ggml")), 60L)
  expect_equal(SeuratObject::Misc(gpu, "ggml_ggml")$backend, "vulkan")
  expect_equal(
    SeuratObject::Stdev(gpu, "ggml"),
    SeuratObject::Stdev(cpu, "ggml"),
    tolerance = 1e-3
  )
})

test_that("repeated Vulkan RunGGML is deterministic (no scheduler aliasing)", {
  skip_no_seurat_gpu()
  so <- seurat_fixture()

  a <- RunGGML(so, op = "embed", n_components = 8, device = "vulkan")
  b <- RunGGML(so, op = "embed", n_components = 8, device = "vulkan")

  ea <- SeuratObject::Embeddings(a, "ggml")
  eb <- SeuratObject::Embeddings(b, "ggml")
  expect_equal(ea, eb, tolerance = 1e-6)
})

test_that("RunGGML.default on a bare matrix returns a ggml_result", {
  X   <- sc_fixture()
  res <- RunGGML(X, op = "embed", n_components = 6, device = "cpu")
  expect_s3_class(res, "ggml_result")
  expect_equal(dim(res$embedding), c(150L, 6L))
})

# ---------------------------------------------------------------------------
# Transform ops: normalize (LogNormalize) and scale (ScaleData z-score)
# ---------------------------------------------------------------------------

# raw counts fixture (transforms read counts / data, not random gaussians)
sc_counts_fixture <- function(seed = 7L) {
  set.seed(seed)
  X <- matrix(rpois(40 * 60, lambda = 4), nrow = 40)
  rownames(X) <- paste0("Gene", seq_len(40))
  colnames(X) <- paste0("Cell", seq_len(60))
  storage.mode(X) <- "double"
  X
}

test_that("registry declares the normalize and scale transform ops", {
  reg <- names(ggml_ops_registry())
  expect_true(all(c("normalize", "scale") %in% reg))
  # transforms take no required params
  expect_length(ggml_ops_registry("normalize")$params, 0L)
  expect_length(ggml_ops_registry("scale")$params, 0L)
})

test_that("CPU normalize matches LogNormalize and is tagged as a transform", {
  X   <- sc_counts_fixture()
  res <- ggml_run(ggml_task("normalize", X, device = "cpu"))
  expect_s3_class(res, "ggml_result")
  expect_identical(res$metadata$kind, "transform")
  expect_identical(res$metadata$layer, "data")
  expect_equal(dim(res$embedding), dim(X))

  ref <- log1p(sweep(X, 2L, colSums(X), `/`) * 1e4)
  expect_equal(unname(res$embedding), unname(ref), tolerance = 1e-10)
})

test_that("CPU scale matches per-gene z-score with clamp", {
  X   <- log1p(sc_counts_fixture())          # pretend this is the data layer
  res <- ggml_run(ggml_task("scale", X, device = "cpu"))
  expect_identical(res$metadata$layer, "scale.data")
  expect_equal(dim(res$embedding), dim(X))

  mu  <- rowMeans(X)
  sdv <- apply(X, 1L, stats::sd); sdv[sdv == 0] <- 1
  ref <- pmin((X - mu) / sdv, 10)
  expect_equal(unname(res$embedding), unname(ref), tolerance = 1e-10)
  expect_true(all(res$embedding <= 10 + 1e-9))     # clamp respected
  expect_lt(max(abs(rowMeans(res$embedding))), 1e-9)  # centered
})

test_that("RunGGML normalize then scale write the assay layers (CPU)", {
  skip_if_not_installed("Seurat")
  skip_if_not_installed("SeuratObject")
  so <- seurat_fixture()        # has counts + data

  so <- RunGGML(so, op = "normalize", device = "cpu")
  gpu_data <- as.matrix(SeuratObject::LayerData(so, layer = "data"))
  raw      <- as.matrix(SeuratObject::LayerData(so, layer = "counts"))
  ref_data <- log1p(sweep(raw, 2L, colSums(raw), `/`) * 1e4)
  expect_equal(unname(gpu_data), unname(ref_data), tolerance = 1e-8)
  expect_equal(SeuratObject::Misc(so, "data_ggml")$backend, "cpu")

  so <- RunGGML(so, op = "scale", device = "cpu")
  sd_mat <- as.matrix(SeuratObject::LayerData(so, layer = "scale.data"))
  expect_equal(dim(sd_mat), dim(gpu_data))
  expect_lt(max(abs(rowMeans(sd_mat))), 1e-6)
  expect_equal(SeuratObject::Misc(so, "scale.data_ggml")$backend, "cpu")
})

test_that("Vulkan normalize and scale agree with the CPU path", {
  skip_no_seurat_gpu()
  X <- sc_counts_fixture()

  n_gpu <- ggml_run(ggml_task("normalize", X, device = "vulkan"))$embedding
  n_cpu <- ggml_run(ggml_task("normalize", X, device = "cpu"))$embedding
  expect_equal(unname(n_gpu), unname(n_cpu), tolerance = 1e-3)

  D <- log1p(X)
  s_gpu <- ggml_run(ggml_task("scale", D, device = "vulkan"))$embedding
  s_cpu <- ggml_run(ggml_task("scale", D, device = "cpu"))$embedding
  expect_equal(unname(s_gpu), unname(s_cpu), tolerance = 1e-3)
})
