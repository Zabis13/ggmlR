#!/usr/bin/env Rscript
# Write every model's inputs and this package's outputs as flat binary, for the
# ONNX Runtime reference runner to read back.
#
# The inputs are written rather than described because they have to be the SAME
# numbers on both sides: reproducing R's Mersenne-Twister in C++ to regenerate
# them would put a second implementation between the two results being
# compared, which is what a reference exists to avoid.
#
# Format: little-endian float32, no header, in the element order onnx_run uses.
# manifest.tsv carries everything the runner needs to rebuild the tensors.
#
# Usage: Rscript inst/reference/dump_io.R [outdir] [model-substring]

library(ggmlR)

ONNX_DIR <- "/mnt/Data2/DS_projects/ONNX models-main"
args     <- commandArgs(trailingOnly = TRUE)
OUT_DIR  <- if (length(args) >= 1 && nzchar(args[1])) args[1] else "inst/reference/data"
# The wrapper always passes a second argument, empty when no filter was given,
# so emptiness has to mean "all models" rather than being matched against.
ONLY     <- if (length(args) >= 2 && nzchar(args[2])) args[2] else NULL
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Same models, shapes and input generation as inst/examples/test_all_onnx.R —
# the point is to check what that script reports as OK.
models <- list(
  list(file = "mnist-8.onnx",            inputs = list(Input3 = c(1L,1L,28L,28L))),
  list(file = "squeezenet1.0-8.onnx",    inputs = list(data_0 = c(1L,3L,224L,224L))),
  list(file = "adv_inception_v3_Opset17.onnx", inputs = list(x = c(1L,3L,299L,299L))),
  list(file = "adv_inception_v3_Opset18.onnx", inputs = list(x = c(1L,3L,299L,299L))),
  list(file = "super-resolution-10.onnx", inputs = list(input = c(1L,1L,224L,224L))),
  list(file = "emotion-ferplus-8.onnx",  inputs = list(Input3 = c(1L,1L,64L,64L))),
  list(file = "bert_Opset17.onnx",
       inputs = list(input_ids = c(1L,128L), attention_mask = c(1L,128L)),
       int_inputs = c("input_ids","attention_mask")),
  # edge_index is [2, 2708] -- what this exported model declares.  Cora has
  # 10556 edges, and that number was used here at first, but the export fixed
  # the dimension at the node count instead; ONNX Runtime rejects anything
  # else, and ggmlR silently accepted the wrong shape.
  list(file = "sageconv_Opset16.onnx",
       inputs = list(x = c(2708L,1433L), edge_index = c(2L,2708L)),
       int_inputs = c("edge_index")),
  list(file = "roberta-sequence-classification-9.onnx",
       inputs = list(input = c(1L,128L)), int_inputs = c("input")),
  list(file = "bat_resnext26ts_Opset18.onnx", inputs = list(x = c(1L,3L,256L,256L))),
  list(file = "botnet26t_256_Opset16.onnx",   inputs = list(x = c(1L,3L,256L,256L))),
  list(file = "cait_xs24_384_Opset16.onnx",   inputs = list(x = c(1L,3L,384L,384L))),
  list(file = "gptneox_Opset18.onnx",
       inputs = list(input_ids = c(1L,128L), attention_mask = c(1L,128L)),
       int_inputs = c("input_ids","attention_mask")),
  list(file = "MaskRCNN-12-int8.onnx",   inputs = list(image = c(3L,224L,224L))),
  list(file = "xcit_tiny_12_p8_224_Opset17.onnx", inputs = list(x = c(1L,3L,224L,224L)))
)

lines <- character(0)
for (m in models) {
  tag  <- sub("\\.onnx$", "", m$file)
  if (!is.null(ONLY) && !grepl(ONLY, tag, fixed = TRUE)) next
  path <- file.path(ONNX_DIR, m$file)
  cat(sprintf("%-45s ", m$file))
  if (!file.exists(path)) { cat("SKIP (no file)\n"); next }

  # tryCatch returns this model's manifest rows, or NULL if it failed.  The
  # rows are returned rather than appended from inside the block: `<<-` into a
  # variable of the script's own top level is refused as a locked binding,
  # which turned every single model into a FAIL.
  rows <- tryCatch({
    model <- onnx_load(path, device = "cpu", input_shapes = m$inputs)

    set.seed(42)
    input_data <- list()
    for (nm in names(m$inputs)) {
      sz <- prod(m$inputs[[nm]])
      input_data[[nm]] <- if (!is.null(m$int_inputs) && nm %in% m$int_inputs)
        rep(1, sz) else runif(sz)
    }

    out <- onnx_run(model, input_data)

    r <- character(0)
    for (nm in names(input_data)) {
      writeBin(as.numeric(input_data[[nm]]),
               file.path(OUT_DIR, sprintf("%s.in.%s.bin", tag, nm)),
               size = 4, endian = "little")
      r <- c(r, sprintf("in\t%s\t%s\t%s\t%s\t%d", tag, m$file, nm,
                        paste(m$inputs[[nm]], collapse = ","),
                        as.integer(!is.null(m$int_inputs) &&
                                   nm %in% m$int_inputs)))
    }
    # Only the first output is compared: it is what test_all_onnx.R reports on,
    # and a model whose first output agrees has agreed about everything feeding
    # it, which is most of the graph.
    v <- out[[1]]
    writeBin(as.numeric(v), file.path(OUT_DIR, sprintf("%s.ggmlr.bin", tag)),
             size = 4, endian = "little")
    r <- c(r, sprintf("out\t%s\t%s\t%d", tag, m$file, length(v)))

    cat(sprintf("OK  out=%d  head=%s\n", length(v),
                paste(format(head(v, 3), digits = 6), collapse = " ")))
    rm(model, out); gc(verbose = FALSE)
    r
  }, error = function(e) {
    cat(sprintf("FAIL %s\n", conditionMessage(e)))
    NULL
  })

  if (!is.null(rows)) lines <- c(lines, rows)
}

writeLines(lines, file.path(OUT_DIR, "manifest.tsv"))
cat("\nwritten to", OUT_DIR, "\n")
