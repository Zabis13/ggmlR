#!/usr/bin/env Rscript
# Test load + run for all 15 ONNX Zoo models
library(ggmlR)

ONNX_DIR <- "/mnt/Data2/DS_projects/ONNX models-main"

# Each entry: file, input specs (name -> shape), device, description
# For integer inputs (token ids etc) set integer = TRUE
models <- list(
  list(file = "mnist-8.onnx",
       inputs = list(Input3 = c(1L, 1L, 28L, 28L))),

  list(file = "squeezenet1.0-8.onnx",
       inputs = list(data_0 = c(1L, 3L, 224L, 224L))),

  list(file = "adv_inception_v3_Opset17.onnx",
       inputs = list(x = c(1L, 3L, 299L, 299L))),

  list(file = "adv_inception_v3_Opset18.onnx",
       inputs = list(x = c(1L, 3L, 299L, 299L))),

  list(file = "super-resolution-10.onnx",
       inputs = list(input = c(1L, 1L, 224L, 224L))),

  list(file = "emotion-ferplus-8.onnx",
       inputs = list(Input3 = c(1L, 1L, 64L, 64L))),

  list(file = "bert_Opset17.onnx",
       inputs = list(input_ids = c(1L, 128L),
                     attention_mask = c(1L, 128L)),
       int_inputs = c("input_ids", "attention_mask")),

  list(file = "sageconv_Opset16.onnx",
       inputs = list(x = c(4L, 3L),
                     edge_index = c(2L, 5L)),
       int_inputs = c("edge_index")),

  list(file = "roberta-sequence-classification-9.onnx",
       inputs = list(input = c(1L, 128L)),
       int_inputs = c("input")),

  list(file = "bat_resnext26ts_Opset18.onnx",
       inputs = list(x = c(1L, 3L, 256L, 256L))),

  list(file = "botnet26t_256_Opset16.onnx",
       inputs = list(x = c(1L, 3L, 256L, 256L))),

  list(file = "cait_xs24_384_Opset16.onnx",
       inputs = list(x = c(1L, 3L, 384L, 384L))),

  list(file = "gptneox_Opset18.onnx",
       inputs = list(input_ids = c(1L, 128L),
                     attention_mask = c(1L, 128L)),
       int_inputs = c("input_ids", "attention_mask")),

  list(file = "MaskRCNN-12-int8.onnx",
       inputs = list(image = c(3L, 224L, 224L))),

  list(file = "xcit_tiny_12_p8_224_Opset17.onnx",
       inputs = list(x = c(1L, 3L, 224L, 224L)))
)

cat(sprintf("Testing %d ONNX models on CPU\n\n", length(models)))

pass <- 0L
fail <- 0L

for (m in models) {
  path <- file.path(ONNX_DIR, m$file)
  cat(sprintf("%-45s ", m$file))

  if (!file.exists(path)) {
    cat("SKIP (file not found)\n")
    next
  }

  tryCatch({
    # Build input_shapes for onnx_load
    input_shapes <- lapply(m$inputs, identity)

    model <- onnx_load(path, device = "cpu", input_shapes = input_shapes)

    # Generate input data
    set.seed(42)
    input_data <- list()
    for (nm in names(m$inputs)) {
      sz <- prod(m$inputs[[nm]])
      if (!is.null(m$int_inputs) && nm %in% m$int_inputs) {
        input_data[[nm]] <- rep(1, sz)
      } else {
        input_data[[nm]] <- runif(sz)
      }
    }

    out <- onnx_run(model, input_data)
    n_out <- length(out)
    out_len <- length(out[[1]])
    cat(sprintf("OK  (%d outputs, first length=%d)\n", n_out, out_len))
    pass <- pass + 1L
    rm(model, out); gc(verbose = FALSE)
  }, error = function(e) {
    cat(sprintf("FAIL: %s\n", e$message))
    fail <<- fail + 1L
  })
}

cat(sprintf("\n--- Results: %d OK, %d FAIL, %d total ---\n", pass, fail, pass + fail))
