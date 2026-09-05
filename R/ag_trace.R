# Diagnostics: who reads ag_tensor$data, and how often?
#
# The autograd tape keeps every tensor's value as an R matrix in $data. Making
# the tape GPU-resident means $data stops being the source of truth and reads
# have to go through .ag_data(), which materialises on demand. Before changing
# that contract it is worth knowing empirically which call sites actually read
# $data and at what frequency: a site that reads once per model is a mechanical
# substitution, a site that reads once per training step is a hot path where a
# careless .ag_data() would reintroduce exactly the per-op round trip that
# residency is meant to remove.
#
# grep finds the sites but cannot tell those two apart, so this traces reads at
# runtime. It is OFF unless GGMLR_AG_TRACE_DATA=1, and the check happens once
# per tensor at construction, never on the read path itself, so a normal session
# pays nothing for it.
#
# Usage:
#   GGMLR_AG_TRACE_DATA=1 Rscript -e 'testthat::test_local(filter = "...")'
#   ag_trace_data_report()          # ranked call sites
#
# The trace records the CALLER of each read, walking past ag_tensor internals so
# that .ag_data() shows up as its own caller rather than swallowing the site.

.ag_trace <- new.env(parent = emptyenv())
.ag_trace$enabled <- FALSE
.ag_trace$counts  <- NULL   # environment used as a hash: site -> count

#' Trace reads of `ag_tensor$data`
#'
#' Turns the tracer on or off for the current session. Enabling it affects only
#' tensors created afterwards, since the instrumentation is installed when a
#' tensor is built.
#'
#' @param on `TRUE` to trace, `FALSE` to stop, `NA` to only query the state.
#' @return The previous state, invisibly.
#' @keywords internal
ag_trace_data <- function(on = TRUE) {
  old <- .ag_trace$enabled
  if (!is.na(on)) {
    .ag_trace$enabled <- isTRUE(on)
    if (isTRUE(on) && is.null(.ag_trace$counts)) ag_trace_data_reset()
  }
  invisible(old)
}

#' @rdname ag_trace_data
#' @keywords internal
ag_trace_data_reset <- function() {
  .ag_trace$counts <- new.env(parent = emptyenv())
  invisible(NULL)
}

# Identify the code location that asked for $data.
#
# sys.calls() is walked from the top down, skipping the frames that belong to
# the tracing machinery and to ag_tensor's own accessors, so the site reported
# is the first frame that genuinely wanted the value.
.ag_trace_site <- function() {
  calls <- sys.calls()
  if (length(calls) < 2L) return("<top level>")
  skip <- c("ag_trace_data", ".ag_trace_site", ".ag_trace_record")
  for (i in rev(seq_len(length(calls) - 1L))) {
    fn <- calls[[i]][[1L]]
    nm <- if (is.name(fn)) as.character(fn) else NULL
    if (is.null(nm) || nm %in% skip) next
    src <- utils::getSrcref(calls[[i]])
    loc <- if (!is.null(src)) {
      paste0(basename(utils::getSrcFilename(calls[[i]])), ":",
             utils::getSrcLocation(calls[[i]], "line"))
    } else ""
    return(if (nzchar(loc)) paste0(nm, "  (", loc, ")") else nm)
  }
  "<top level>"
}

.ag_trace_record <- function() {
  site <- .ag_trace_site()
  cur  <- .ag_trace$counts[[site]]
  .ag_trace$counts[[site]] <- if (is.null(cur)) 1L else cur + 1L
  invisible(NULL)
}

# Install the counting binding on one tensor.
#
# $data must stay both readable and writable: ag_load_model() assigns to it
# (R/ag_save.R), and so does every op that refreshes a tensor's value. The
# active binding therefore proxies a private slot rather than replacing it, and
# behaves exactly like a plain field apart from the counter.
.ag_trace_install <- function(e, value) {
  e$.data_slot <- value
  # makeActiveBinding() refuses to shadow an existing regular binding
  # ("symbol already has a regular binding"), and the constructor has already
  # assigned e$data by this point, so the plain field is removed first.
  if (exists("data", envir = e, inherits = FALSE)) {
    rm("data", envir = e)
  }
  makeActiveBinding("data", function(v) {
    if (missing(v)) {
      .ag_trace_record()
      e$.data_slot
    } else {
      e$.data_slot <- v
    }
  }, e)
  invisible(e)
}

#' Report traced reads of `ag_tensor$data`
#'
#' @param n Maximum number of call sites to print.
#' @return A data frame of sites and counts, invisibly.
#' @keywords internal
ag_trace_data_report <- function(n = 40L) {
  if (is.null(.ag_trace$counts)) {
    message("ag_trace_data_report(): tracing was never enabled.")
    return(invisible(NULL))
  }
  sites  <- ls(.ag_trace$counts, all.names = TRUE)
  if (!length(sites)) {
    message("ag_trace_data_report(): no reads recorded.")
    return(invisible(NULL))
  }
  counts <- vapply(sites, function(s) .ag_trace$counts[[s]], integer(1))
  ord    <- order(counts, decreasing = TRUE)
  df     <- data.frame(site = sites[ord], reads = counts[ord],
                       row.names = NULL, stringsAsFactors = FALSE)

  cat("\nReads of ag_tensor$data by call site\n")
  cat(strrep("-", 72), "\n")
  for (i in seq_len(min(n, nrow(df)))) {
    cat(sprintf("%8d  %s\n", df$reads[i], df$site[i]))
  }
  cat(strrep("-", 72), "\n")
  cat(sprintf("%8d  TOTAL across %d sites\n", sum(df$reads), nrow(df)))
  invisible(df)
}
