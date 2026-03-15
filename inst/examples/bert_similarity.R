#!/usr/bin/env Rscript
# ============================================================================
# BERT Semantic Similarity — ggmlR ONNX inference example
#
# Loads BERT-base from ONNX, tokenizes sentences via vocab.txt,
# and compares them by cosine similarity of pooler_output (CLS embedding).
#
# Requirements:
#   - bert_Opset17.onnx  (ONNX Zoo)
#   - vocab.txt          (BERT-base-cased, 28996 tokens)
# ============================================================================

library(ggmlR)

# --- Paths ---
ONNX_DIR  <- "/mnt/Data2/DS_projects/ONNX models-main"
MODEL     <- file.path(ONNX_DIR, "bert_Opset17.onnx")
VOCAB     <- file.path(ONNX_DIR, "vocab.txt")
SEQ_LEN   <- 128L   # sequence length (must match the model)

stopifnot(file.exists(MODEL), file.exists(VOCAB))

# ============================================================================
# 1. Simple WordPiece tokenizer
# ============================================================================

# Load vocabulary: token -> 0-based index
vocab_lines <- readLines(VOCAB, warn = FALSE)
vocab <- setNames(seq_along(vocab_lines) - 1L, vocab_lines)

CLS_ID <- vocab[["[CLS]"]]   # 101
SEP_ID <- vocab[["[SEP]"]]   # 102
PAD_ID <- vocab[["[PAD]"]]   #   0
UNK_ID <- vocab[["[UNK]"]]   # 100

# Greedy longest-match WordPiece
tokenize_word <- function(word, max_chars = 200L) {
  if (nchar(word) > max_chars) return(UNK_ID)
  tokens <- integer(0)
  start <- 1L
  while (start <= nchar(word)) {
    found <- FALSE
    for (end in nchar(word):start) {
      piece <- substring(word, start, end)
      if (start > 1L) piece <- paste0("##", piece)
      if (!is.na(vocab[piece])) {
        tokens <- c(tokens, vocab[piece])
        start <- end + 1L
        found <- TRUE
        break
      }
    }
    if (!found) {
      tokens <- c(tokens, UNK_ID)
      start <- start + 1L
    }
  }
  tokens
}

# Full encode: [CLS] tokens... [SEP] + padding to SEQ_LEN
bert_encode <- function(text) {
  text <- tolower(trimws(text))
  words <- unlist(strsplit(text, "(?=[[:punct:]])|(?<=[[:punct:]])|\\s+", perl = TRUE))
  words <- words[nchar(words) > 0]

  ids <- CLS_ID
  for (w in words) ids <- c(ids, tokenize_word(w))
  ids <- c(ids, SEP_ID)

  # Truncate
  if (length(ids) > SEQ_LEN) {
    ids <- ids[1:SEQ_LEN]
    ids[SEQ_LEN] <- SEP_ID
  }

  # Padding
  n_real <- length(ids)
  n_pad  <- SEQ_LEN - n_real
  mask   <- c(rep(1, n_real), rep(0, n_pad))
  ids    <- c(ids, rep(PAD_ID, n_pad))

  list(input_ids = as.numeric(ids), attention_mask = as.numeric(mask),
       n_tokens = n_real)
}

# ============================================================================
# 2. Load model
# ============================================================================

cat("Loading BERT-base (Opset 17)...\n")
t0 <- proc.time()
model <- onnx_load(
  MODEL,
  device       = if (ggml_vulkan_available()) "vulkan" else "cpu",
  input_shapes = list(input_ids = c(1L, SEQ_LEN), attention_mask = c(1L, SEQ_LEN))
)
load_sec <- (proc.time() - t0)[3]
cat(sprintf("  Loaded in %.2f s  (%s)\n", load_sec,
            if (ggml_vulkan_available()) "Vulkan GPU" else "CPU"))
print(model)

# ============================================================================
# 3. Inference helper
# ============================================================================

# pooler_output: [CLS] -> Linear -> Tanh -> 768-dim vector
bert_embed <- function(text) {
  enc <- bert_encode(text)
  out <- onnx_run(model, list(
    input_ids      = enc$input_ids,
    attention_mask = enc$attention_mask
  ))
  # out[[1]] = last_hidden_state [128 x 768]
  # out[[2]] = pooler_output     [768]
  as.numeric(out[[2]])
}

cosine_sim <- function(a, b) {
  sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2)))
}

# ============================================================================
# 4. Compare sentences
# ============================================================================

sentences <- c(
  "The cat sat on the mat",
  "A kitten was sitting on the rug",
  "Dogs are playing in the park",
  "The stock market crashed today",
  "Financial markets experienced a downturn"
)

cat("\n==============================================================\n")
cat("  BERT Semantic Similarity (pooler_output)\n")
cat("==============================================================\n\n")

# Tokenization check
cat("Tokenization:\n")
for (s in sentences) {
  enc <- bert_encode(s)
  cat(sprintf("  \"%s\" -> %d tokens\n", s, enc$n_tokens))
}

# Embeddings
cat("\nComputing embeddings...\n")
t0 <- proc.time()
embeddings <- lapply(sentences, bert_embed)
embed_sec <- (proc.time() - t0)[3]
cat(sprintf("  %d sentences in %.3f s (%.1f ms each)\n",
            length(sentences), embed_sec,
            embed_sec / length(sentences) * 1000))

# Similarity matrix
n <- length(sentences)
sim_matrix <- matrix(0, n, n)
for (i in seq_len(n)) {
  for (j in seq_len(n)) {
    sim_matrix[i, j] <- cosine_sim(embeddings[[i]], embeddings[[j]])
  }
}

# Output
cat("\nSentences:\n")
for (i in seq_len(n)) {
  cat(sprintf("  [%d] %s\n", i, sentences[i]))
}

cat("\nCosine similarity matrix:\n\n")
cat(sprintf("      %s\n", paste(sprintf("[%d]   ", seq_len(n)), collapse = "")))
for (i in seq_len(n)) {
  cat(sprintf(" [%d]  %s\n", i,
              paste(sprintf("%.3f  ", sim_matrix[i, ]), collapse = "")))
}

# Top pairs (excluding diagonal)
cat("\nMost similar pairs:\n")
pairs <- data.frame(i = integer(), j = integer(), sim = numeric())
for (i in 1:(n - 1)) {
  for (j in (i + 1):n) {
    pairs <- rbind(pairs, data.frame(i = i, j = j, sim = sim_matrix[i, j]))
  }
}
pairs <- pairs[order(-pairs$sim), ]
for (k in seq_len(nrow(pairs))) {
  p <- pairs[k, ]
  cat(sprintf("  %.3f  [%d] %s\n         [%d] %s\n",
              p$sim, p$i, sentences[p$i], p$j, sentences[p$j]))
}

cat("\nNote: BERT-base without fine-tuning produces generic embeddings.\n")
cat("For better similarity, use a sentence-transformer model (e.g. all-MiniLM).\n")
cat("==============================================================\n")
