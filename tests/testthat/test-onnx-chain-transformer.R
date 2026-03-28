# Chain tests: Transformer / BERT patterns
# Gather(embeddings) → Add(pos_emb) → LayerNorm → MatMul(QKV) → Softmax → MatMul → Add(residual)
#
# Tests embedding lookup, shape propagation through Add+LN, attention MatMul ndims.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}


# ── Minimal (3 ops): Gather → Add → LayerNorm ────────────────

test_that("chain transformer: Gather→Add→LayerNorm (minimal)", {
  # Embedding table [4, 3], indices [2] → Gather → [2, 3]
  # Positional embedding [2, 3] (init) → Add → [2, 3]
  # LayerNorm → [2, 3]

  inp <- .onnx_value_info("I", 7L, c(2L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 3L))

  # Embedding table
  emb_data <- c(0.1, 0.2, 0.3,
                0.4, 0.5, 0.6,
                0.7, 0.8, 0.9,
                1.0, 1.1, 1.2)
  emb_raw <- unlist(lapply(emb_data, .float_bytes))
  emb_t  <- .onnx_tensor("E", c(4L, 3L), 1L, emb_raw)
  emb_vi <- .onnx_value_info("E", 1L, c(4L, 3L))

  # Positional embedding
  pos_data <- rep(0.01, 6)
  pos_raw <- unlist(lapply(pos_data, .float_bytes))
  pos_t  <- .onnx_tensor("P", c(2L, 3L), 1L, pos_raw)
  pos_vi <- .onnx_value_info("P", 1L, c(2L, 3L))

  # LN scale/bias
  sc_raw <- unlist(lapply(rep(1.0, 3), .float_bytes))
  bi_raw <- unlist(lapply(rep(0.0, 3), .float_bytes))
  sc_t <- .onnx_tensor("sc", c(3L), 1L, sc_raw)
  bi_t <- .onnx_tensor("bi", c(3L), 1L, bi_raw)
  sc_vi <- .onnx_value_info("sc", 1L, c(3L))
  bi_vi <- .onnx_value_info("bi", 1L, c(3L))

  gather_node <- .onnx_node("Gather", c("E", "I"), "emb_out",
                             attrs = list(.onnx_attr_int("axis", 0L)))
  add_node <- .onnx_node("Add", c("emb_out", "P"), "add_out")
  ln_node  <- .onnx_node("LayerNormalization", c("add_out", "sc", "bi"), "Y",
                          attrs = list(.onnx_attr_float("epsilon", 1e-5)))

  graph <- .onnx_graph("test",
                        list(gather_node, add_node, ln_node),
                        list(inp, emb_vi, pos_vi, sc_vi, bi_vi),
                        list(outp),
                        list(emb_t, pos_t, sc_t, bi_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- run_onnx(path, list(I = c(0, 2)))
  r <- as.numeric(result)
  expect_equal(length(r), 6)
  # LN normalizes each row to mean≈0
  expect_true(abs(mean(r[1:3])) < 0.01)
  expect_true(abs(mean(r[4:6])) < 0.01)
})


# ── Real (5 ops): Gather → Add → LN → MatMul → Softmax ──────

test_that("chain transformer: Gather→Add→LN→MatMul→Softmax (attention-like)", {
  # Simulates: embed tokens, add pos, normalize, project to logits, softmax
  # Embed [4, 8], indices [3] → [3, 8]
  # Add pos [3, 8] → [3, 8]
  # LN → [3, 8]
  # MatMul with W[8, 4] → [3, 4]  (project to 4-class vocab)
  # Softmax → [3, 4]

  inp <- .onnx_value_info("I", 7L, c(3L))
  outp <- .onnx_value_info("Y", 1L, c(3L, 4L))

  # Embedding table [4, 8]
  set.seed(42)
  emb_data <- rnorm(32, 0, 0.5)
  emb_raw <- unlist(lapply(emb_data, .float_bytes))
  emb_t  <- .onnx_tensor("E", c(4L, 8L), 1L, emb_raw)
  emb_vi <- .onnx_value_info("E", 1L, c(4L, 8L))

  # Positional embedding [3, 8]
  pos_data <- rnorm(24, 0, 0.1)
  pos_raw <- unlist(lapply(pos_data, .float_bytes))
  pos_t  <- .onnx_tensor("P", c(3L, 8L), 1L, pos_raw)
  pos_vi <- .onnx_value_info("P", 1L, c(3L, 8L))

  # LN params
  sc_raw <- unlist(lapply(rep(1.0, 8), .float_bytes))
  bi_raw <- unlist(lapply(rep(0.0, 8), .float_bytes))
  sc_t <- .onnx_tensor("sc", c(8L), 1L, sc_raw)
  bi_t <- .onnx_tensor("bi", c(8L), 1L, bi_raw)
  sc_vi <- .onnx_value_info("sc", 1L, c(8L))
  bi_vi <- .onnx_value_info("bi", 1L, c(8L))

  # Projection: [8, 4]
  proj_data <- rnorm(32, 0, 0.3)
  proj_raw <- unlist(lapply(proj_data, .float_bytes))
  proj_t  <- .onnx_tensor("W", c(8L, 4L), 1L, proj_raw)
  proj_vi <- .onnx_value_info("W", 1L, c(8L, 4L))

  gather_node <- .onnx_node("Gather", c("E", "I"), "emb",
                             attrs = list(.onnx_attr_int("axis", 0L)))
  add_node <- .onnx_node("Add", c("emb", "P"), "add_out")
  ln_node  <- .onnx_node("LayerNormalization", c("add_out", "sc", "bi"), "ln_out",
                          attrs = list(.onnx_attr_float("epsilon", 1e-5)))
  mm_node  <- .onnx_node("MatMul", c("ln_out", "W"), "mm_out")
  sm_node  <- .onnx_node("Softmax", "mm_out", "Y",
                          attrs = list(.onnx_attr_int("axis", 1L)))

  graph <- .onnx_graph("test",
                        list(gather_node, add_node, ln_node, mm_node, sm_node),
                        list(inp, emb_vi, pos_vi, sc_vi, bi_vi, proj_vi),
                        list(outp),
                        list(emb_t, pos_t, sc_t, bi_t, proj_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- run_onnx(path, list(I = c(0, 1, 3)))
  r <- as.numeric(result)
  expect_equal(length(r), 12)  # 3 tokens x 4 classes
  # All softmax outputs in [0,1], total sum = 3 (three rows)
  expect_true(all(r >= 0 & r <= 1))
  expect_equal(sum(r), 3.0, tolerance = 1e-3)
})


# ── Boundary: single token, dim=1 ────────────────────────────

test_that("chain transformer: single token embedding (boundary)", {
  # Embed [3, 2], index [1] → [1, 2]
  # Add pos [1, 2] → [1, 2]
  # LN → [1, 2]

  inp <- .onnx_value_info("I", 7L, c(1L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L))

  emb_data <- c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
  emb_raw <- unlist(lapply(emb_data, .float_bytes))
  emb_t  <- .onnx_tensor("E", c(3L, 2L), 1L, emb_raw)
  emb_vi <- .onnx_value_info("E", 1L, c(3L, 2L))

  pos_data <- c(0.1, 0.1)
  pos_raw <- unlist(lapply(pos_data, .float_bytes))
  pos_t  <- .onnx_tensor("P", c(1L, 2L), 1L, pos_raw)
  pos_vi <- .onnx_value_info("P", 1L, c(1L, 2L))

  sc_raw <- unlist(lapply(rep(1.0, 2), .float_bytes))
  bi_raw <- unlist(lapply(rep(0.0, 2), .float_bytes))
  sc_t <- .onnx_tensor("sc", c(2L), 1L, sc_raw)
  bi_t <- .onnx_tensor("bi", c(2L), 1L, bi_raw)
  sc_vi <- .onnx_value_info("sc", 1L, c(2L))
  bi_vi <- .onnx_value_info("bi", 1L, c(2L))

  gather_node <- .onnx_node("Gather", c("E", "I"), "emb",
                             attrs = list(.onnx_attr_int("axis", 0L)))
  add_node <- .onnx_node("Add", c("emb", "P"), "add_out")
  ln_node  <- .onnx_node("LayerNormalization", c("add_out", "sc", "bi"), "Y",
                          attrs = list(.onnx_attr_float("epsilon", 1e-5)))

  graph <- .onnx_graph("test",
                        list(gather_node, add_node, ln_node),
                        list(inp, emb_vi, pos_vi, sc_vi, bi_vi),
                        list(outp),
                        list(emb_t, pos_t, sc_t, bi_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # Token index 1 → embed = [3.0, 4.0], + pos → [3.1, 4.1]
  result <- run_onnx(path, list(I = c(1)))
  r <- as.numeric(result)
  expect_equal(length(r), 2)
  # LN of [3.1, 4.1]: mean=3.6, std=0.5 → [-1, 1]
  expect_true(abs(mean(r)) < 0.01)
})
