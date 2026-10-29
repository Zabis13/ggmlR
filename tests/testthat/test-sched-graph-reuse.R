# The scheduler must leave the caller's graph untouched.
#
# split_graph points a node's src at the split backend's copy of an input that
# lives on another backend. Upstream leaves that rewrite in the graph, so
# running the same graph through reset -> alloc again found src already pointing
# at the stale copy: no new copy, an output placed on top of it, and the second
# run came out wrong (36 -> 138 -> 444 for the same inputs). The setup that
# triggers it: weights in a Vulkan buffer, the op pinned to the CPU.

skip_if_no_vk_sched <- function() {
  skip_on_cran()
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "Vulkan not available")
}

src_name <- function(t, j) .Call("R_ggml_tensor_src_name", t, as.integer(j), PACKAGE = "ggmlR")

# Weights on the GPU, a [K, M] and b [K, N]; the graph mm = a^T b, r = sum_rows(mm)
# runs on the CPU, so both weights reach it through scheduler copies.
K <- 4L; M <- 2L; N <- 3L
# Fixed seed so a failure reproduces; expect_equal() tolerance below is relative.
set.seed(42)
XA <- runif(K * M); XB <- runif(K * N)
REF_R  <- colSums(crossprod(matrix(XA, K), matrix(XB, K)))
REF_MM <- as.vector(crossprod(matrix(XA, K), matrix(XB, K)))

new_env <- function() {
  e <- new.env()
  e$gpu <- ggml_vulkan_init(0L)
  e$cpu <- ggml_backend_cpu_init()
  ggml_backend_cpu_set_n_threads(e$cpu, 2L)
  e$sched <- ggml_backend_sched_new(list(e$gpu, e$cpu), parallel = FALSE)
  e$ctx_w <- ggml_init(1024 * 1024, no_alloc = TRUE)
  e$a <- ggml_new_tensor_2d(e$ctx_w, GGML_TYPE_F32, K, M); ggml_set_name(e$a, "w_a")
  e$b <- ggml_new_tensor_2d(e$ctx_w, GGML_TYPE_F32, K, N); ggml_set_name(e$b, "w_b")
  e$buf <- ggml_backend_alloc_ctx_tensors(e$ctx_w, e$gpu)
  ggml_backend_tensor_set_data(e$a, XA)
  ggml_backend_tensor_set_data(e$b, XB)
  e
}

free_env <- function(e) {
  if (!is.null(e$sched)) ggml_backend_sched_free(e$sched)
  ggml_backend_buffer_free(e$buf)
  ggml_free(e$ctx_w)
  ggml_backend_free(e$cpu)
  ggml_backend_free(e$gpu)
}

# kind = "sum": r = sum_rows(a^T b); kind = "scale": r = 2 * (a^T b)
build_graph <- function(e, kind = "sum") {
  g <- list(ctx = ggml_init(1024 * 1024, no_alloc = TRUE), kind = kind)
  g$mm <- ggml_mul_mat(g$ctx, e$a, e$b); ggml_set_name(g$mm, "mm")
  g$r  <- if (kind == "sum") ggml_sum_rows(g$ctx, g$mm) else ggml_scale(g$ctx, g$mm, 2)
  g$gf <- ggml_build_forward_expand(g$ctx, g$r)
  g
}

pin_cpu <- function(e, g) {
  ggml_backend_sched_set_tensor_backend(e$sched, g$mm, e$cpu)
  ggml_backend_sched_set_tensor_backend(e$sched, g$r,  e$cpu)
}

expect_graph_clean <- function(g) {
  expect_identical(src_name(g$mm, 0L), "w_a")
  expect_identical(src_name(g$mm, 1L), "w_b")
  expect_identical(src_name(g$r,  0L), "mm")
}

expect_result <- function(g) {
  ref <- if (g$kind == "sum") REF_R else 2 * REF_MM
  expect_equal(ggml_backend_tensor_get_data(g$r), ref, tolerance = 1e-5)
}

run_once <- function(e, g, alloc = TRUE) {
  ggml_backend_sched_reset(e$sched)
  expect_graph_clean(g)
  pin_cpu(e, g)
  if (alloc) {
    expect_true(ggml_backend_sched_alloc_graph(e$sched, g$gf))
    expect_graph_clean(g)
  }
  ggml_backend_sched_graph_compute(e$sched, g$gf)
  expect_graph_clean(g)
  expect_result(g)
}

test_that("same graph through reset -> alloc -> compute stays correct", {
  skip_if_no_vk_sched()
  e <- new_env(); on.exit(free_env(e))
  g <- build_graph(e)
  expect_graph_clean(g)
  for (i in 1:3) run_once(e, g)
  ggml_free(g$ctx)
})

test_that("alloc once, compute many", {
  skip_if_no_vk_sched()
  e <- new_env(); on.exit(free_env(e))
  g <- build_graph(e)
  ggml_backend_sched_reset(e$sched)
  pin_cpu(e, g)
  expect_true(ggml_backend_sched_alloc_graph(e$sched, g$gf))
  for (i in 1:3) {
    ggml_backend_sched_graph_compute(e$sched, g$gf)
    expect_graph_clean(g)
    expect_result(g)
  }
  ggml_free(g$ctx)
})

test_that("graph rebuilt for every run", {
  skip_if_no_vk_sched()
  e <- new_env(); on.exit(free_env(e))
  for (i in 1:3) {
    g <- build_graph(e)
    run_once(e, g)
    ggml_free(g$ctx)
  }
})

test_that("compute without an explicit alloc", {
  skip_if_no_vk_sched()
  e <- new_env(); on.exit(free_env(e))
  g <- build_graph(e)
  for (i in 1:3) run_once(e, g, alloc = FALSE)
  ggml_free(g$ctx)
})

test_that("two graphs alternating on one scheduler", {
  skip_if_no_vk_sched()
  e <- new_env(); on.exit(free_env(e))
  g1 <- build_graph(e, "sum")
  g2 <- build_graph(e, "scale")
  for (i in 1:3) {
    run_once(e, g1)
    expect_graph_clean(g2)
    run_once(e, g2)
    expect_graph_clean(g1)
  }
  ggml_free(g1$ctx); ggml_free(g2$ctx)
})

test_that("graph freed before or after the scheduler", {
  skip_if_no_vk_sched()
  # graph context freed first, then the scheduler is reset, reused and freed
  e <- new_env()
  g <- build_graph(e)
  run_once(e, g)
  ggml_free(g$ctx)
  ggml_backend_sched_reset(e$sched)
  g2 <- build_graph(e)
  run_once(e, g2)
  ggml_free(g2$ctx)
  free_env(e)

  # scheduler freed first, graph afterwards
  e <- new_env()
  g <- build_graph(e)
  run_once(e, g)
  ggml_backend_sched_free(e$sched); e$sched <- NULL
  expect_graph_clean(g)
  ggml_free(g$ctx)
  free_env(e)
  succeed()
})

# An error return out of the middle of a compute must still put the original
# src pointers back. The CPU backend is made to abort after n nodes of its
# split, whose nodes run in the order s, mm, r:
#   n = 1 -- after s, which reads a CPU tensor: the copies are already in the
#            graph, but no node that reads them has run;
#   n = 2 -- after mm, the node that reads the copies: part of the work is done.
# The C++-exception path out of the Vulkan backend cannot be driven from R.

cpu_abort_after <- function(backend, n) {
  invisible(.Call("R_ggml_backend_cpu_abort_after", backend, as.integer(n), PACKAGE = "ggmlR"))
}

GGML_STATUS_ABORTED <- 1L  # enum ggml_status in ggml.h

XC <- runif(5L)

for (abort_n in 1:2) {
  local({
    n <- abort_n
    test_that(paste("compute aborted after node", n, "leaves the graph clean"), {
      skip_if_no_vk_sched()
      e <- new_env()
      ctx_c <- ggml_init(1024 * 1024, no_alloc = TRUE)
      xc <- ggml_new_tensor_1d(ctx_c, GGML_TYPE_F32, length(XC)); ggml_set_name(xc, "x_cpu")
      buf_c <- ggml_backend_alloc_ctx_tensors(ctx_c, e$cpu)
      ggml_backend_tensor_set_data(xc, XC)
      on.exit({
        cpu_abort_after(e$cpu, 0L)
        ggml_backend_buffer_free(buf_c); ggml_free(ctx_c)
        free_env(e)
      })

      g <- build_graph(e)
      s <- ggml_scale(g$ctx, xc, 3)
      gf <- ggml_build_forward_expand(g$ctx, s)   # node order: s, mm, r
      ggml_graph_expand(gf, g$r)
      g$gf <- gf

      for (i in 1:3) {
        ggml_backend_sched_reset(e$sched)
        pin_cpu(e, g)
        ggml_backend_sched_set_tensor_backend(e$sched, s, e$cpu)
        expect_true(ggml_backend_sched_alloc_graph(e$sched, g$gf))
        expect_graph_clean(g)

        cpu_abort_after(e$cpu, n)
        expect_identical(ggml_backend_sched_graph_compute(e$sched, g$gf), GGML_STATUS_ABORTED)
        expect_graph_clean(g)
        expect_identical(src_name(s, 0L), "x_cpu")

        cpu_abort_after(e$cpu, 0L)
        expect_identical(ggml_backend_sched_graph_compute(e$sched, g$gf), 0L)
        expect_graph_clean(g)
        expect_result(g)
        expect_equal(ggml_backend_tensor_get_data(s), 3 * XC, tolerance = 1e-6)
      }
      ggml_free(g$ctx)
    })
  })
}
