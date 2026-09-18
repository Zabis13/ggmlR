#!/usr/bin/env Rscript
# Compare two ONNX_TRACE_VALS traces node by node, and report the first
# divergence.
#
# Companion to ref_first_divergence.sh, which diffs the two traces as TEXT.
# That works only while the two runs visit nodes in the same order; on a
# segmented model they do not, and a text diff then reports hundreds of
# differences that are all the same nodes in another sequence. This compares
# VALUES, matched per node, so a reordering is not a finding and a wrong number
# is.
#
# Capture the traces with the sums on -- three leading values hide a difference
# anywhere else in the tensor:
#   ONNX_TRACE_VALS=1 ONNX_TRACE_SUM=1 Rscript inst/scripts/repeat_maskrcnn.R cpu    > cpu.txt 2>&1
#   ONNX_TRACE_VALS=1 ONNX_TRACE_SUM=1 Rscript inst/scripts/repeat_maskrcnn.R vulkan > vk.txt  2>&1
#
# Usage:
#   Rscript inst/scripts/ref_compare_traces.R <a.txt> <b.txt> [rel_tol]
#
# A crashing run is expected input, not a problem: its trace simply stops, and
# the last line of the shorter trace is where the backend died. That node --
# the one AFTER the last one both traces hold -- is the one to look at.
#
# ⚠️ Two traps, both of which produced wrong answers before this script existed:
#
#  1. Do NOT match nodes by their index in the trace. The two backends compute
#     in different orders, so equal indices are different nodes; the values
#     then look shifted and every node appears to diverge. (Seen on MaskRCNN:
#     cpu node 3's sum turned up as vulkan node 4's.)
#
#  2. Node NAMES are not unique. 'node_12' occurs many times with different
#     shapes, and ' (permuted)' occurred 125 times in one trace. Matching by
#     name alone pairs unrelated nodes. The key here is
#     name|op|shape|n plus the occurrence number, which pairs correctly --
#     verified by the fact that matched pairs then agree exactly.
#
# View ops (PERMUTE/VIEW/TRANSPOSE) carry no sum because they own no data;
# they are counted and skipped, not treated as missing values.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("usage: ref_compare_traces.R <a.txt> <b.txt> [rel_tol]\n")
  quit(status = 1)
}
A <- args[1]; B <- args[2]
TOL <- if (length(args) >= 3) as.numeric(args[3]) else 1e-4

for (f in c(A, B)) if (!file.exists(f)) { cat("missing:", f, "\n"); quit(status = 1) }

# A named numeric field, or NA when the line does not carry it. Look-behind so
# that "sum=" does not also match "n=" inside it.
num <- function(L, key) {
  m <- regexpr(paste0("(?<=", key, "=)[^ ]+"), L, perl = TRUE)
  x <- rep(NA_character_, length(L))
  ok <- m > 0
  x[ok] <- regmatches(L, m)
  suppressWarnings(as.numeric(x))
}

parse_trace <- function(f) {
  L <- grep("^\\[val\\] ", readLines(f, warn = FALSE), value = TRUE)
  if (!length(L)) return(NULL)
  data.frame(
    idx = as.integer(sub("^\\[val\\] ([0-9]+) .*", "\\1", L)),
    nm  = sub("^\\[val\\] [0-9]+ '(.*?)' op=.*", "\\1", L),
    op  = sub(".* op=([A-Za-z_0-9]+) .*", "\\1", L),
    ne  = sub(".*ne=\\[([^]]*)\\].*", "\\1", L),
    sum = num(L, "sum"), min = num(L, "min"), max = num(L, "max"), n = num(L, "n"),
    stringsAsFactors = FALSE)
}

a <- parse_trace(A); b <- parse_trace(B)
if (is.null(a) || is.null(b)) { cat("no [val] lines -- was ONNX_TRACE_VALS=1 set?\n"); quit(status = 1) }

cat(sprintf("A: %-30s %6d nodes\n", basename(A), nrow(a)))
cat(sprintf("B: %-30s %6d nodes\n", basename(B), nrow(b)))
if (nrow(a) != nrow(b)) {
  short <- if (nrow(a) < nrow(b)) "A" else "B"
  cat(sprintf("\n%s stops first, at node %d -- if that run crashed, the node it\n",
              short, min(nrow(a), nrow(b))))
  cat("died producing is the one AFTER this, in that run's own order:\n")
  s <- if (nrow(a) < nrow(b)) a else b
  o <- if (nrow(a) < nrow(b)) b else a
  cat("  last computed : ", tail(s$nm, 1), " (", tail(s$op, 1), ")\n", sep = "")
  # What the other run did next is NOT the answer: the two orders differ (see
  # the traps above), so the successor must be read from the surviving run's
  # position of that same node, not from its index.
  pos <- which(o$nm == tail(s$nm, 1) & o$op == tail(s$op, 1))
  if (length(pos) && pos[1] < nrow(o)) {
    cat("  the other run computed this next, at the SAME point in its own order:\n")
    cat("    ", o$nm[pos[1] + 1], " (", o$op[pos[1] + 1], ")",
        "  <- the node to investigate\n", sep = "")
  }
}

key <- function(d) paste(d$nm, d$op, d$ne, d$n, sep = "|")
a$k <- key(a); b$k <- key(b)
a$occ <- ave(seq_len(nrow(a)), a$k, FUN = seq_along)
b$occ <- ave(seq_len(nrow(b)), b$k, FUN = seq_along)

m <- merge(a, b, by = c("k", "occ"), suffixes = c(".a", ".b"))
m <- m[order(m$idx.a), ]

d   <- abs(m$sum.a - m$sum.b)
rel <- d / pmax(abs(m$sum.a), abs(m$sum.b), 1e-9)
bad <- which(!is.na(rel) & rel > TOL & d > TOL)

n_view <- sum(is.na(m$sum.a))
cat(sprintf("\nmatched node-instances: %d   (%d view ops without sums, skipped)\n",
            nrow(m), n_view))
cat(sprintf("divergent (rel > %g): %d\n\n", TOL, length(bad)))

if (!length(bad)) {
  cat("No value divergence on any matched node.\n")
  cat("If one run crashed, the defect is memory/scheduling, not arithmetic.\n")
  quit(status = 0)
}

show <- head(bad, 20)
out <- data.frame(
  idx.a = m$idx.a[show], idx.b = m$idx.b[show],
  node  = m$nm.a[show],  op = m$op.a[show], shape = m$ne.a[show],
  sum.a = signif(m$sum.a[show], 8), sum.b = signif(m$sum.b[show], 8),
  rel   = signif(rel[show], 3), stringsAsFactors = FALSE)
print(out, row.names = FALSE)
if (length(bad) > 20) cat(sprintf("\n... and %d more\n", length(bad) - 20))
cat("\nThe FIRST row is the node to investigate; the rest are downstream of it.\n")
cat("Confirm isolation: check that node's inputs agree, then its output does not.\n")
