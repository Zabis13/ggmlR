# Diagnostics: how many host<->device transfers does one training step cost?
#
# Why this exists. The whole point of full residency is that a tensor stops
# crossing the PCIe bus every step, and the way that goes wrong is silent: one
# forgotten .ag_data() somewhere puts a matrix back on the host, everything
# still computes the right answer, and the only symptom is that the speedup
# never arrives. Timings alone cannot tell "the transfer is gone" from "the
# transfer is still there but noise hid it".
#
# So this counts the crossings themselves. Path B has exactly eight sites that
# touch ggml_backend_tensor_get_data / _set_data:
#
#   R/ag_handle.R:70          download  materialise a handle
#   R/ag_device.R:605         upload    .ag_run_op operands
#   R/ag_device.R:644         download  .ag_run_op result
#   R/ag_device.R:921         upload    .ag_r_to_gpu
#   R/ag_device.R:954         upload    .ag_r_to_gpu_batch
#   R/ag_device.R:963         download  .ag_gpu_to_r
#   R/ag_backward_graph.R:522 upload    backward graph operands
#   R/ag_backward_graph.R:597 download  backward graph leaf gradients
#
# Each is wrapped by .ag_xfer_up() / .ag_xfer_down() below, which count and then
# delegate. The counters are OFF unless switched on, and when off the wrapper is
# a single field read on top of a device copy that costs microseconds -- far
# below the noise of the transfer it wraps.
#
# Usage:
#   ag_xfer_count(TRUE)       # start counting, resets the tally
#   ...run a training step...
#   ag_xfer_report()          # per-site counts and bytes
#
# What to look for. After stage 3.1 (resident weights, host-side optimizer
# step) one step should download each weight once for the optimizer's
# arithmetic and upload it once afterwards. More than that means a transfer
# nobody asked for -- and finding it is cheaper before the device-side step is
# written than after, because that change would hide it among its own traffic.

.ag_xfer <- new.env(parent = emptyenv())
.ag_xfer$enabled <- FALSE
.ag_xfer$counts  <- NULL   # environment used as a hash: "dir site" -> list(n, bytes)

#' Count host/device transfers on the ag_* path
#'
#' Switches the transfer counter on or off. Turning it on clears any previous
#' tally, so a measurement always starts from zero.
#'
#' @param on `TRUE` to count, `FALSE` to stop, `NA` to only query the state.
#' @return Invisibly, the previous state.
#' @keywords internal
ag_xfer_count <- function(on = TRUE) {
  old <- .ag_xfer$enabled
  if (!is.na(on)) {
    .ag_xfer$enabled <- isTRUE(on)
    if (isTRUE(on)) ag_xfer_reset()
  }
  invisible(old)
}

#' @rdname ag_xfer_count
#' @keywords internal
ag_xfer_reset <- function() {
  .ag_xfer$counts <- new.env(parent = emptyenv())
  invisible(NULL)
}

# Record one crossing. `site` names the call site, `dir` is "up" or "down".
.ag_xfer_record <- function(dir, site, n_elem) {
  key <- paste(dir, site)
  cur <- .ag_xfer$counts[[key]]
  if (is.null(cur)) {
    .ag_xfer$counts[[key]] <- list(dir = dir, site = site, n = 1L,
                                   elems = as.double(n_elem))
  } else {
    cur$n     <- cur$n + 1L
    cur$elems <- cur$elems + as.double(n_elem)
    .ag_xfer$counts[[key]] <- cur
  }
  invisible(NULL)
}

# Upload wrapper: count, then do the transfer.
#
# The count happens even when the value is short -- a partial write is still a
# crossing, and the point here is the number of trips, not only their size.
.ag_xfer_up <- function(ptr, data, site) {
  if (.ag_xfer$enabled) .ag_xfer_record("up", site, length(data))
  ggml_backend_tensor_set_data(ptr, data)
}

# Download wrapper: count, then do the transfer.
.ag_xfer_down <- function(ptr, site) {
  raw <- ggml_backend_tensor_get_data(ptr)
  if (.ag_xfer$enabled) .ag_xfer_record("down", site, length(raw))
  raw
}

#' Report host/device transfers recorded on the ag_* path
#'
#' @param n Maximum number of sites to print.
#' @return A data frame of sites, directions, counts and bytes, invisibly.
#' @keywords internal
ag_xfer_report <- function(n = 40L) {
  if (is.null(.ag_xfer$counts)) {
    message("ag_xfer_report(): counting was never enabled.")
    return(invisible(NULL))
  }
  keys <- ls(.ag_xfer$counts, all.names = TRUE)
  if (!length(keys)) {
    message("ag_xfer_report(): no transfers recorded.")
    return(invisible(NULL))
  }
  rec <- lapply(keys, function(k) .ag_xfer$counts[[k]])
  df  <- data.frame(
    dir   = vapply(rec, function(r) r$dir,  character(1)),
    site  = vapply(rec, function(r) r$site, character(1)),
    n     = vapply(rec, function(r) r$n,    integer(1)),
    # f32 on the wire: the counters see R doubles, the device stores 4 bytes.
    mb    = vapply(rec, function(r) r$elems * 4 / 1024^2, numeric(1)),
    row.names = NULL, stringsAsFactors = FALSE)
  df <- df[order(-df$n), ]

  cat("\nHost<->device transfers on the ag_* path\n")
  cat(strrep("-", 72), "\n")
  cat(sprintf("%-6s %8s %10s  %s\n", "dir", "count", "MB", "site"))
  for (i in seq_len(min(n, nrow(df)))) {
    cat(sprintf("%-6s %8d %10.2f  %s\n",
                df$dir[i], df$n[i], df$mb[i], df$site[i]))
  }
  cat(strrep("-", 72), "\n")
  up   <- df[df$dir == "up", , drop = FALSE]
  down <- df[df$dir == "down", , drop = FALSE]
  cat(sprintf("%-6s %8d %10.2f\n", "up",   sum(up$n),   sum(up$mb)))
  cat(sprintf("%-6s %8d %10.2f\n", "down", sum(down$n), sum(down$mb)))
  cat(sprintf("%-6s %8d %10.2f  TOTAL\n", "", sum(df$n), sum(df$mb)))
  invisible(df)
}
