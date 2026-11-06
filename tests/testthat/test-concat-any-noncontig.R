# CPU CONCAT for types without a dedicated kernel (F64 here; also I64 and the
# quantized types) goes through ggml_compute_forward_concat_any. Upstream 0.27
# asserts contiguous rows there; ggmlR keeps an element-wise path for inputs
# whose rows are not contiguous (DIVERGENCE) and walks ne[4] in both paths.
# Checked against a reference built in R, bit for bit, on 2 threads so the
# i2 split across ith/nth is exercised. The Vulkan CONCAT shader (concat.comp,
# 8-byte types included) is checked against the same reference, not against CPU.
#
# F64 data goes through ggml_backend_tensor_set_data/get_data as is.

# view shape seen by CONCAT (ggml order, ne0 first); 5D so i4 is walked
VIEW_NE <- c(4L, 3L, 2L, 1L, 2L)

# concatenate R arrays x, y along axis k (1-based)
cat_along <- function(x, y, k) {
  n <- length(dim(x))
  p <- c(setdiff(seq_len(n), k), k)
  xp <- aperm(x, p)
  yp <- aperm(y, p)
  r <- array(c(xp, yp), c(dim(xp)[-n], dim(xp)[n] + dim(yp)[n]))
  aperm(r, order(p))
}

# dim: ggml concat axis (0..4); permuted: feed transposed views (rows not
# contiguous, element-wise path) or the tensors themselves (row-copy path)
run_concat_f64 <- function(backend, dim, permuted) {
  ne_a <- VIEW_NE
  ne_b <- VIEW_NE
  ne_b[dim + 1L] <- ne_b[dim + 1L] + 1L   # unequal sizes along dim: catches o[]
  # a permute(1, 0, 2, 3) view of a [ne1, ne0, ...] source has shape [ne0, ne1, ...]
  src_ne <- function(ne) if (permuted) ne[c(2L, 1L, 3L, 4L, 5L)] else ne

  set.seed(100 + dim)
  da <- runif(prod(ne_a), -1, 1)
  db <- runif(prod(ne_b), -1, 1) + 10      # disjoint ranges: a mixed-up source shows

  ctx <- ggml_init(16 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  sa <- ggml_new_tensor(ctx, GGML_TYPE_F64, 5L, src_ne(ne_a))
  sb <- ggml_new_tensor(ctx, GGML_TYPE_F64, 5L, src_ne(ne_b))
  va <- if (permuted) ggml_permute(ctx, sa, 1L, 0L, 2L, 3L) else sa
  vb <- if (permuted) ggml_permute(ctx, sb, 1L, 0L, 2L, 3L) else sb
  out <- ggml_concat(ctx, va, vb, dim)
  gf  <- ggml_build_forward_expand(ctx, out)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)

  ggml_backend_tensor_set_data(sa, da)
  ggml_backend_tensor_set_data(sb, db)
  ggml_backend_graph_compute(backend, gf)
  got <- ggml_backend_tensor_get_data(out)

  # reference: the arrays CONCAT sees (views), concatenated along dim
  view_arr <- function(d, ne) {
    s <- array(d, src_ne(ne))
    if (permuted) aperm(s, c(2L, 1L, 3L, 4L, 5L)) else s
  }
  want <- cat_along(view_arr(da, ne_a), view_arr(db, ne_b), dim + 1L)
  list(got = got, want = as.numeric(want))
}

for (perm in c(TRUE, FALSE)) {
  for (d in 0:4) {
    local({
      permuted <- perm
      dim <- d
      label <- if (permuted) "non-contiguous rows" else "contiguous rows"
      test_that(sprintf("CPU concat_any F64, %s, dim %d", label, dim), {
        backend <- ggml_backend_cpu_init()
        on.exit(ggml_backend_free(backend))
        ggml_backend_cpu_set_n_threads(backend, 2L)
        r <- run_concat_f64(backend, dim, permuted)
        expect_identical(r$got, r$want)
      })

      # concat.comp: a different kernel, same reference. It takes 8-byte types
      # (ggml_vk_concat_supported), so F64 is a plain copy here too.
      test_that(sprintf("Vulkan concat F64, %s, dim %d", label, dim), {
        skip_if_not(ggml_vulkan_available(), "no Vulkan device")
        backend <- ggml_vulkan_init(0)
        on.exit(ggml_backend_free(backend))
        r <- run_concat_f64(backend, dim, permuted)
        expect_identical(r$got, r$want)
      })
    })
  }
}
