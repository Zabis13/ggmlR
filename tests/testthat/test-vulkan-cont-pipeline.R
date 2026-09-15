# CONT on Vulkan: the copy must be correct whichever shader carries it.
#
# ggml_vk_get_cpy_pipeline picks between a 32x32 tiled "transpose" shader and a
# linear "contiguous copy" one. The choice used to hinge on nb[1] == type_size,
# which is also true of a tensor whose ne[0] is 1 -- trivially, since there is
# nothing for the stride to step over. Such a tensor is contiguous, and sending
# it to the tiled shader sized the grid from ne[0] and ne[1], both 1, so
# CEIL_DIV(1,32) = 1: a 1x1xN dispatch filling one element per 1024-element
# tile. On MaskRCNN that shape carried two thirds of all CONT time.
#
# These tests pin correctness on both routes, which is what the fix relies on: a
# misrouted copy is silent, since both shaders copy -- just at very different
# rates. Speed is deliberately not asserted here; timings belong in a benchmark,
# not a test that has to pass on any device.

if (isTRUE(tryCatch(ggml_vulkan_available(), error = function(e) FALSE))) {

  # One ggml_cont through the Vulkan backend, returned as a plain vector.
  cont_vk <- function(ne0, ne1, values, transpose = FALSE) {
    ctx <- ggml_init(mem_size = 64 * 1024 * 1024)
    ggml_set_no_alloc(ctx, TRUE)

    t   <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1)
    src <- if (transpose) ggml_transpose(ctx, t) else t
    out <- ggml_cont(ctx, src)

    backend <- ggml_vulkan_init(0)
    buffer  <- ggml_backend_alloc_ctx_tensors(ctx, backend)
    ggml_backend_tensor_set_data(t, values)

    graph <- ggml_build_forward_expand(ctx, out)
    ggml_backend_graph_compute(backend, graph)
    res <- ggml_backend_tensor_get_data(out)

    ggml_backend_buffer_free(buffer)
    ggml_vulkan_free(backend)
    ggml_free(ctx)
    res
  }

  test_that("Vulkan CONT: contiguous copy with a unit leading dim", {
    # ne[0] == 1 is the shape that used to be misrouted to the tiled shader:
    # contiguous, yet with nb[1] == type_size, which the old condition read as
    # "transposed". A plain copy must come back unchanged.
    set.seed(1)
    v <- runif(64)
    expect_equal(cont_vk(1L, 64L, v), v, tolerance = 1e-6)
  })

  test_that("Vulkan CONT: ordinary contiguous copy", {
    set.seed(3)
    v <- runif(32 * 32)
    expect_equal(cont_vk(32L, 32L, v), v, tolerance = 1e-6)
  })

  test_that("Vulkan CONT: a genuinely transposed copy still transposes", {
    # Not contiguous, and not a unit axis: this is what the tiled shader is for,
    # and it must keep taking that path and producing the transpose.
    set.seed(2)
    ne0 <- 48L; ne1 <- 80L
    v   <- runif(ne0 * ne1)
    got <- cont_vk(ne0, ne1, v, transpose = TRUE)
    # ggml is column-major: the tensor holds v as [ne0, ne1], and cont() of its
    # transpose is that matrix read the other way round.
    want <- as.numeric(t(matrix(v, nrow = ne0, ncol = ne1)))
    expect_equal(got, want, tolerance = 1e-6)
  })
}
