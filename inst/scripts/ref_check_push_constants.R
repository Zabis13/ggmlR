#!/usr/bin/env Rscript
# Check every Vulkan shader's push constant block against the C++ struct that
# fills it.
#
# A mismatch here is silent in every other check. The shader reads whatever
# bytes sit at each field's offset, so a block that is one field short does not
# fail to compile, does not fault, and does not warn -- it computes addresses
# from the wrong numbers and writes plausible-looking garbage.
#
# ⚠️ This is what scatter_elements.comp did: it declared four ne/nb per tensor
# while vk_op_binary_push_constants has five (GGML_MAX_DIMS == 5), so 29 fields
# against 35. Everything after ne03 was shifted -- the dst strides read nb11..
# nb13 and the scatter AXIS read nb22. `total` still came out right, because it
# uses only the fields before the first missing one, so the shader ran over the
# correct element count and wrote every one of them to a garbage address. Found
# only after three rebuild cycles hunting it through a 45 MB detector; this
# script finds it in under a second, with no GPU and no build.
#
# Most shaders take their layout from a shared .glsl head and cannot drift.
# The ones that declare their own block are the ones worth checking, and they
# are found here rather than listed, so a new one is covered the day it lands.
#
# Usage:
#   Rscript inst/scripts/ref_check_push_constants.R
#
# Exit status is 1 if any shader disagrees with its struct.

SHADER_DIR <- "src/ggml-vulkan/vulkan-shaders"
CPP_FILES  <- c("src/ggml-vulkan/ggml-vulkan-device.cpp")

if (!dir.exists(SHADER_DIR)) {
  cat("run from the package root (no", SHADER_DIR, ")\n")
  quit(status = 2)
}

# ── the C++ side: struct name -> field names, in order ───────────
parse_structs <- function(paths) {
  out <- list()
  for (p in paths) {
    ln <- readLines(p, warn = FALSE)
    starts <- grep("^struct vk_op_[A-Za-z0-9_]+ \\{", ln)
    for (s in starts) {
      nm <- sub("^struct (vk_op_[A-Za-z0-9_]+) \\{.*", "\\1", ln[s])
      e <- s
      while (e < length(ln) && !grepl("^\\};", ln[e])) e <- e + 1
      body <- ln[(s + 1):(e - 1)]
      body <- body[!grepl("^\\s*(//|/\\*|\\*)", body)]   # drop comment lines
      body <- sub("//.*$", "", body)
      # One declaration may name several fields: `uint32_t nb01, nb02;` is two,
      # and reading only the first made eight agreeing structs look like
      # mismatches on the first run of this script. Split each declaration on
      # commas, then take the identifiers.
      fields <- character(0)
      for (d in unlist(strsplit(paste(body, collapse = "\n"), ";", fixed = TRUE))) {
        d <- trimws(d)
        if (!grepl("^(uint32_t|int32_t|float|uint|int)\\b", d)) next
        d <- sub("^(uint32_t|int32_t|float|uint|int)\\s+", "", d)
        for (nm2 in strsplit(d, ",", fixed = TRUE)[[1]]) {
          nm2 <- trimws(nm2)
          nm2 <- sub("\\[.*$", "", nm2)          # arrays: nb[12][4] -> nb
          if (grepl("^[A-Za-z_][A-Za-z0-9_]*$", nm2)) fields <- c(fields, nm2)
        }
      }
      if (length(fields)) out[[nm]] <- fields
    }
  }
  out
}

# ── the shader side: the fields of its push_constant block ──────
parse_shader_block <- function(path) {
  ln <- readLines(path, warn = FALSE)
  s <- grep("layout\\s*\\(\\s*push_constant\\s*\\)", ln)
  if (!length(s)) return(NULL)
  s <- s[1]
  b <- s
  while (b <= length(ln) && !grepl("\\{", ln[b])) b <- b + 1
  e <- b
  while (e < length(ln) && !grepl("^\\s*\\}\\s*[A-Za-z_]*\\s*;", ln[e])) e <- e + 1
  body <- ln[(b + 1):(e - 1)]
  # A block with #if branches has more than one layout; comparing it needs to
  # know which branch is live, which this script does not. Say so, do not guess.
  if (any(grepl("^\\s*#(if|ifdef|else|elif)", body))) return(NA)
  # Strip comments before splitting on ';'. A trailing /* 2*H-1 */ holds no
  # semicolon but does hold text that looks like a declaration once the lines
  # are joined, and a leading comment line above the block gets swept in with
  # it -- both made agreeing shaders report as mismatches.
  body <- gsub("/\\*.*?\\*/", "", paste(body, collapse = "\n"))
  body <- gsub("//[^\n]*", "", body)
  body <- strsplit(body, "\n", fixed = TRUE)[[1]]
  # Same comma rule as the struct side: `uint nb01, nb02;` is two fields.
  fields <- character(0)
  for (d in unlist(strsplit(paste(body, collapse = "\n"), ";", fixed = TRUE))) {
    d <- trimws(d)
    if (!grepl("^(uint32_t|int32_t|uint|int|float|uvec2|ivec2)\\b", d)) next
    d <- sub("^(uint32_t|int32_t|uint|int|float|uvec2|ivec2)\\s+", "", d)
    for (nm2 in strsplit(d, ",", fixed = TRUE)[[1]]) {
      nm2 <- trimws(nm2)
      nm2 <- sub("\\[.*$", "", nm2)
      if (grepl("^[A-Za-z_][A-Za-z0-9_]*$", nm2)) fields <- c(fields, nm2)
    }
  }
  fields
}

structs <- parse_structs(CPP_FILES)
cat(sprintf("parsed %d vk_op_* struct(s)\n\n", length(structs)))

# Which struct does a shader's dispatcher send?  Read it from the call rather
# than guessed from the name: ggml_vk_op_f32<vk_op_X_push_constants>(...).
disp <- readLines("src/ggml-vulkan/ggml-vulkan-elemwise.cpp", warn = FALSE)
disp <- c(disp, readLines("src/ggml-vulkan/ggml-vulkan-attn.cpp", warn = FALSE))

shaders <- list.files(SHADER_DIR, pattern = "\\.comp$", full.names = TRUE)

# A shader that includes a shared head takes its layout from there.
own_block <- Filter(function(f) {
  ln <- readLines(f, warn = FALSE)
  any(grepl("layout\\s*\\(\\s*push_constant\\s*\\)", ln))
}, shaders)

cat(sprintf("%d shader(s) declare their own push constant block\n\n",
            length(own_block)))

fails <- 0L; checked <- 0L; skipped <- character(0)

for (f in own_block) {
  base <- sub("\\.comp$", "", basename(f))
  sf <- parse_shader_block(f)
  if (is.null(sf)) next
  if (length(sf) == 1 && is.na(sf)) { skipped <- c(skipped, paste0(base, " (#if in block)")); next }

  # Find the struct sent for this op, by the shader's base name.
  hit <- names(structs)[names(structs) == paste0("vk_op_", base, "_push_constants")]
  if (!length(hit)) {
    # No struct named after the shader. Read which one its dispatcher sends
    # instead -- ggml_vk_<op>(...) { ... ggml_vk_op_f32<vk_op_X_push_constants>
    # -- because that is the fact the comparison needs, and it is the case that
    # matters most: scatter_elements sends vk_op_binary_push_constants, and
    # THAT is the pair that had drifted.
    #
    # ⚠️ Never match by field count. Tried, and it paired mul_mat_vec_nc with
    # vk_op_pool2d_push_constants because both hold 13 fields, then reported
    # the unrelated names as a mismatch.
    fn <- grep(sprintf("static void ggml_vk_%s\\(", base), disp)
    if (length(fn)) {
      # Wide enough to clear a long preamble: ggml_vk_scatter_elements copies
      # the base tensor and carries a diagnostic block before it dispatches,
      # putting its ggml_vk_op_f32 call ~100 lines below the signature.
      tail_lines <- disp[fn[1]:min(fn[1] + 160, length(disp))]
      m <- regmatches(tail_lines,
                      regexpr("vk_op_[A-Za-z0-9_]+_push_constants", tail_lines))
      m <- unlist(m)
      if (length(m)) hit <- m[1]
    }
    if (!length(hit) || !hit %in% names(structs)) {
      skipped <- c(skipped, paste0(base, " (no struct of that name)"))
      next
    }
  }
  cf <- structs[[hit]]
  checked <- checked + 1L

  # A shader may declare a PREFIX of the struct and still be correct: fields it
  # never reads can sit past the end of its block harmlessly, since every field
  # it does read is at the same offset in both. What is never safe is a field
  # MISSING FROM THE MIDDLE, which shifts everything after it -- that is the
  # scatter_elements defect.
  #
  # topk_moe is the live example: it declares the first five fields and takes
  # with_norm from a specialization constant instead of the struct's
  # gating_func..output_bias tail, deliberately (see the note in the shader).
  # Reporting it as a mismatch every run would teach the reader to ignore this
  # script's output, so a clean prefix is called out as such.
  prefix_ok <- length(sf) < length(cf) && identical(sf, cf[seq_along(sf)])

  if (identical(sf, cf)) {
    cat(sprintf("  %-24s %-34s %2d fields  OK\n", base, hit, length(sf)))
  } else if (prefix_ok) {
    cat(sprintf("  %-24s %-34s %2d of %d  OK (prefix; reads none of the tail)\n",
                base, hit, length(sf), length(cf)))
  } else {
    fails <- fails + 1L
    cat(sprintf("  %-24s %-34s shader %d vs struct %d  MISMATCH\n",
                base, hit, length(sf), length(cf)))
    n <- max(length(sf), length(cf))
    for (i in seq_len(n)) {
      a <- if (i <= length(sf)) sf[i] else "(absent)"
      b <- if (i <= length(cf)) cf[i] else "(absent)"
      if (!identical(a, b))
        cat(sprintf("      @%-4d shader %-12s struct %-12s <-- first divergence\n",
                    (i - 1) * 4, a, b))
      if (!identical(a, b)) break
    }
  }
}

if (length(skipped))
  cat(sprintf("\nnot compared: %s\n", paste(skipped, collapse = ", ")))
cat(sprintf("\n%d checked, %d mismatch(es)\n", checked, fails))
quit(status = if (fails > 0L) 1L else 0L)
