# Vulkan GPU Backend Functions

#' Check if Vulkan support is available
#'
#' Returns TRUE if the package was compiled with Vulkan support.
#' To enable Vulkan, install libvulkan-dev and glslc, then reinstall ggmlR.
#'
#' @return Logical indicating if Vulkan is available
#' @export
#' @examples
#' ggml_vulkan_available()
ggml_vulkan_available <- function() {
  .Call("R_ggml_vulkan_is_available", PACKAGE = "ggmlR")
}

#' Get number of Vulkan devices
#'
#' Returns the number of available Vulkan-capable GPU devices.
#'
#' @return Integer count of Vulkan devices (0 if Vulkan not available)
#' @export
#' @examples
#' \donttest{
#' if (ggml_vulkan_available()) {
#'   ggml_vulkan_device_count()
#' }
#' }
ggml_vulkan_device_count <- function() {
  .Call("R_ggml_vulkan_device_count", PACKAGE = "ggmlR")
}

#' List all Vulkan devices
#'
#' Returns detailed information about all available Vulkan devices.
#'
#' @return List of device information (index, name, memory)
#' @export
#' @examples
#' \donttest{
#' if (ggml_vulkan_available() && ggml_vulkan_device_count() > 0) {
#'   devices <- ggml_vulkan_list_devices()
#'   print(devices)
#' }
#' }
ggml_vulkan_list_devices <- function() {
  .Call("R_ggml_vulkan_list_devices", PACKAGE = "ggmlR")
}

#' Get Vulkan device description
#'
#' Returns a human-readable description of the specified Vulkan device.
#'
#' @param device Device index (0-based)
#' @return Character string with device description
#' @export
#' @examples
#' \donttest{
#' if (ggml_vulkan_available() && ggml_vulkan_device_count() > 0) {
#'   ggml_vulkan_device_description(0)
#' }
#' }
ggml_vulkan_device_description <- function(device = 0L) {
  .Call("R_ggml_vulkan_device_description", as.integer(device), PACKAGE = "ggmlR")
}

#' Get Vulkan device memory
#'
#' Returns free and total memory for the specified Vulkan device.
#'
#' @param device Device index (0-based)
#' @return Named list with 'free' and 'total' memory in bytes
#' @export
#' @examples
#' \donttest{
#' if (ggml_vulkan_available() && ggml_vulkan_device_count() > 0) {
#'   mem <- ggml_vulkan_device_memory(0)
#'   cat("Free:", mem$free / 1e9, "GB\n")
#'   cat("Total:", mem$total / 1e9, "GB\n")
#' }
#' }
ggml_vulkan_device_memory <- function(device = 0L) {
  .Call("R_ggml_vulkan_device_memory", as.integer(device), PACKAGE = "ggmlR")
}

#' Probe Vulkan device groups (NVLink / multi-GPU peer access)
#'
#' Enumerates Vulkan device groups (\code{VK_KHR_device_group}, also known as
#' Linked Device Adapter / LDA) and probes whether the driver reports direct
#' peer memory access between the physical GPUs in each group.
#'
#' A device group with more than one device \emph{and} peer copy/generic memory
#' features is the prerequisite for true GPU-to-GPU transfers routed over NVLink
#' (or PCIe peer-to-peer) through a single device-group logical device — as
#' opposed to sharing memory as an opaque fd between independent devices, which
#' the driver may route through host memory.
#'
#' This is a diagnostic only: it does not create any long-lived device group.
#' On a machine with a single GPU it reports zero multi-device groups. Device
#' groups for compute are effectively an NVIDIA feature; AMD/RADV typically
#' reports only single-device groups.
#'
#' @return A named list with \code{n_groups} (integer, number of device groups
#'   reported by the driver) and \code{report} (character, a human-readable
#'   per-group diagnostic including peer memory features). Use \code{cat()} on
#'   \code{report} to read it.
#' @export
#' @examples
#' \donttest{
#' if (ggml_vulkan_available()) {
#'   g <- ggml_vulkan_device_groups()
#'   cat(g$report)
#' }
#' }
ggml_vulkan_device_groups <- function() {
  .Call("R_ggml_vulkan_device_groups", PACKAGE = "ggmlR")
}

#' Compute tensor-parallel row-split ranges
#'
#' Pure arithmetic helper for the Vulkan tensor-parallel split buffer type: given
#' a number of tensor rows and a per-device weight vector, returns the half-open
#' row range \code{[row_low, row_high)} owned by each device. Row boundaries are
#' rounded down to a fixed granularity, and the last device always covers up to
#' \code{nrows}, so the ranges are contiguous, non-overlapping and cover every
#' row exactly once.
#'
#' This touches no GPU and is exposed mainly to verify the split logic. It is the
#' math behind row-split tensor parallelism (a weight matrix distributed across
#' several GPUs), independent of any actual multi-GPU allocation.
#'
#' @param nrows Number of rows in the tensor (integer).
#' @param n_devices Number of devices to split across (integer, >= 1).
#' @param weights Optional numeric vector of length \code{n_devices} giving the
#'   relative share of rows per device. \code{NULL} (default) splits evenly.
#' @return A named list with \code{row_low} and \code{row_high}: numeric vectors
#'   of length \code{n_devices} holding 0-based, half-open row ranges.
#' @export
#' @examples
#' \donttest{
#' if (ggml_vulkan_available()) {
#'   # 4096 rows across 2 devices, evenly
#'   ggml_vulkan_split_row_ranges(4096L, 2L)
#'   # weighted 3:1
#'   ggml_vulkan_split_row_ranges(4096L, 2L, weights = c(3, 1))
#' }
#' }
ggml_vulkan_split_row_ranges <- function(nrows, n_devices, weights = NULL) {
  w <- if (is.null(weights)) NULL else as.numeric(weights)
  .Call("R_ggml_vk_split_row_ranges", as.numeric(nrows), w,
        as.integer(n_devices), PACKAGE = "ggmlR")
}

#' Opaque-fd device-to-device P2P self-test
#'
#' Exercises the \code{VK_KHR_external_memory_fd} transport used by Vulkan tensor
#' parallelism to move data between GPUs. A byte pattern is written on the source
#' device, its memory is exported as an opaque fd and imported on the destination
#' device, copied into a local buffer there, then read back and compared.
#'
#' When \code{src_device == dst_device} the test runs in \emph{loopback} mode
#' (export and import on the same GPU) — a sanity check of the fd mechanism that
#' touches no inter-device link. When the devices differ it runs
#' \emph{cross-device}: after verifying correctness it times \code{iters}
#' device-to-device copies and reports the achieved bandwidth.
#'
#' Interpreting bandwidth: a measured rate above the PCIe 3.0 x16 ceiling
#' (~16 GB/s) is \emph{empirical} evidence that a faster physical link (e.g.
#' NVLink) carried the bytes. It is \strong{not} a claim that Vulkan used an
#' NVLink API — Vulkan exposes no call to query the route, so the conclusion is
#' inferred from the rate alone, never asserted from the API.
#'
#' @param src_device Source GPU index (0-based).
#' @param dst_device Destination GPU index (0-based). Equal to \code{src_device}
#'   for a loopback sanity check.
#' @param bytes Transfer size in bytes (default 64 MiB).
#' @param iters Number of copies to time for the bandwidth measurement (default 50).
#' @param transport Cross-device transport to exercise: \code{"host-staging"}
#'   (default, portable device->host->device copy — correct on every driver,
#'   PCIe + RAM bounded), \code{"opaque-fd"} (\code{VK_KHR_external_memory_fd} P2P;
#'   works on AMD/RADV but does NOT alias memory cross-device on the NVIDIA
#'   proprietary driver — reads back as zeros), or \code{"device-group"}
#'   (experimental NVIDIA LDA). \code{host-staging} is the transport Stage E3 uses.
#' @return A named list: \code{status} (integer, 0 = data verified, <0 = failure),
#'   \code{gbps} (numeric, measured cross-device bandwidth; 0 for loopback or on
#'   failure) and \code{report} (character diagnostic, incl. the NVLink-vs-PCIe
#'   inference).
#' @export
#' @examples
#' \donttest{
#' if (ggml_vulkan_available()) {
#'   # loopback sanity check on device 0
#'   r <- ggml_vulkan_p2p_selftest(0L, 0L)
#'   cat(r$report)
#'   # cross-device P2P (requires >= 2 GPUs)
#'   if (ggml_vulkan_status()$n_devices >= 2) {
#'     r <- ggml_vulkan_p2p_selftest(0L, 1L)
#'     cat(r$report)
#'   }
#' }
#' }
ggml_vulkan_p2p_selftest <- function(src_device, dst_device,
                                     bytes = 64L * 1024L * 1024L, iters = 50L,
                                     transport = c("host-staging", "opaque-fd", "device-group")) {
  transport <- match.arg(transport)
  t_code <- switch(transport,
                   "host-staging" = 0L,
                   "opaque-fd"    = 1L,
                   "device-group" = 2L)
  .Call("R_ggml_vk_p2p_selftest",
        as.integer(src_device), as.integer(dst_device),
        as.numeric(bytes), as.integer(iters), t_code, PACKAGE = "ggmlR")
}

#' Tensor-parallel matrix multiply across Vulkan devices
#'
#' Computes \code{Y = X \%*\% t(W)} with the rows of the weight matrix \code{W}
#' split across \code{n_devices} GPUs (true tensor parallelism), the activations
#' \code{X} broadcast to every device, and the resulting column slices gathered
#' back into a single matrix. This is Stage E3 of the Vulkan tensor-parallelism
#' work: it exercises the row-split + per-device compute + cross-device gather
#' path end to end on real hardware.
#'
#' Each device owns a contiguous band of \code{W}'s rows (via the same row-split
#' math as \code{\link{ggml_vulkan_split_row_ranges}}) and produces the matching
#' band of \code{Y}'s columns; the bands are gathered through the transport (host
#' staging by default). This is a validation / correctness entry point, not a hot
#' inference path — it stands up one backend per device per call and round-trips
#' through host memory.
#'
#' @param W Weight matrix, \code{N x K} (N output features, K input features).
#'   Its rows are split across the devices.
#' @param X Activation matrix, \code{M x K} (M samples, K input features),
#'   broadcast to every device.
#' @param n_devices Number of GPUs to split across (default: all available).
#'   Ignored if \code{device_ids} is given (its length is used instead).
#' @param weights Optional numeric vector of length \code{n_devices} giving the
#'   relative share of \code{W}'s rows per device (e.g. \code{c(3, 1)} for a 3:1
#'   split). \code{NULL} (default) splits the rows evenly.
#' @param device_ids Optional integer vector of physical GPU indices (0-based) to
#'   split across, e.g. \code{c(2, 3)} for the second replica of a TPxDP layout.
#'   \code{NULL} (default) uses devices \code{0..n_devices-1}. When supplied,
#'   \code{n_devices} is taken from its length.
#' @param transport Cross-device gather transport: \code{"host-staging"} (default,
#'   portable and correct on every driver), \code{"opaque-fd"} or
#'   \code{"device-group"} (see \code{\link{ggml_vulkan_p2p_selftest}}).
#' @return The \code{M x N} result matrix, equal to \code{X \%*\% t(W)} up to the
#'   GPU's floating-point accumulation. Errors if the split compute fails; the
#'   C-level diagnostic is attached as attribute \code{"report"}.
#' @export
#' @examples
#' \donttest{
#' if (ggml_vulkan_available() && ggml_vulkan_device_count() >= 2) {
#'   W <- matrix(rnorm(8 * 4), nrow = 8)   # 8 output features x 4 inputs
#'   X <- matrix(rnorm(3 * 4), nrow = 3)   # 3 samples x 4 inputs
#'   Y <- ggml_vulkan_split_mul_mat(W, X, n_devices = 2)
#'   max(abs(Y - X %*% t(W)))              # ~1e-3 (GPU f16 accumulation)
#' }
#' }
ggml_vulkan_split_mul_mat <- function(W, X, n_devices = ggml_vulkan_device_count(),
                                      weights = NULL, device_ids = NULL,
                                      transport = c("host-staging", "opaque-fd", "device-group")) {
  transport <- match.arg(transport)
  t_code <- switch(transport,
                   "host-staging" = 0L,
                   "opaque-fd"    = 1L,
                   "device-group" = 2L)
  if (!is.null(device_ids)) {
    device_ids <- as.integer(device_ids)
    n_devices  <- length(device_ids)
  }
  W <- as.matrix(W)
  X <- as.matrix(X)
  N <- nrow(W); K <- ncol(W)
  M <- nrow(X)
  if (ncol(X) != K) {
    stop(sprintf("X has %d columns but W has %d (both are the K/input dimension)",
                 ncol(X), K))
  }
  w <- if (is.null(weights)) NULL else as.numeric(weights)

  # Pack into the column-major flat layout the C API expects:
  #   w[n*K + k] = W[n, k]  ->  as.numeric(t(W))  (row n contiguous)
  #   x[m*K + k] = X[m, k]  ->  as.numeric(t(X))
  res <- .Call("R_ggml_vk_split_mul_mat",
               as.numeric(t(W)), as.numeric(t(X)),
               as.numeric(N), as.numeric(K), as.numeric(M),
               w, as.integer(n_devices), device_ids, t_code, PACKAGE = "ggmlR")

  if (res$status != 0L || is.null(res$y)) {
    stop("ggml_vulkan_split_mul_mat failed:\n", res$report)
  }
  # y[m*N + n] = Y[m, n]; a column-major fill of an N x M matrix, transposed -> M x N.
  Y <- t(matrix(res$y, nrow = N, ncol = M))
  attr(Y, "report") <- res$report
  Y
}

#' Explicitly tear down the Vulkan instance and devices
#'
#' Waits for all Vulkan devices to go idle, then destroys the devices and the
#' Vulkan instance. Call this from a long-running script \emph{before} the process
#' exits to run the teardown while the Vulkan loader / ICD libraries are still
#' mapped. Idempotent and a no-op if Vulkan was never initialized. After this,
#' the next Vulkan operation re-initializes the instance from scratch.
#'
#' @return Invisibly \code{NULL}.
#' @export
ggml_vulkan_shutdown <- function() {
  invisible(.Call("R_ggml_vk_shutdown", PACKAGE = "ggmlR"))
}

#' Create a Vulkan tensor-split buffer type
#'
#' Builds (or fetches from a cache) a \code{ggml} buffer type that row-splits a
#' weight tensor across \code{n_devices} GPUs for tensor parallelism. This is the
#' factory the graph allocator talks to (Stage E4): a weight placed in this buffer
#' type has its rows distributed across the devices, each holding a contiguous
#' band (see \code{\link{ggml_vulkan_split_row_ranges}}). Non-split tensors fall
#' back to \code{main_device}.
#'
#' The buffer type is cached in C and lives for the rest of the session; the
#' returned external pointer has no finalizer and must not be freed.
#'
#' @param n_devices Number of GPUs to split across (default: all available).
#'   Ignored if \code{device_ids} is given (its length is used instead).
#' @param main_device Device index (0-based) holding non-split fallbacks (default 0).
#' @param weights Optional numeric vector of length \code{n_devices} giving the
#'   relative row share per device (e.g. \code{c(3, 1)}). \code{NULL} (default)
#'   splits evenly.
#' @param device_ids Optional integer vector of physical GPU indices (0-based) the
#'   split slots map onto, e.g. \code{c(2, 3)}. \code{NULL} (default) uses
#'   \code{0..n_devices-1}. When supplied, \code{n_devices} is its length.
#' @param transport Cross-device gather transport for the eventual compute:
#'   \code{"host-staging"} (default), \code{"opaque-fd"} or \code{"device-group"}.
#' @param probe A numeric vector \code{c(N, K)} describing a probe weight tensor
#'   (\code{N} rows, \code{K} columns) whose split allocation size to report, or
#'   \code{NULL} (default) to skip. Useful for inspecting how much total VRAM a
#'   given weight would occupy across the split.
#' @return A named list: \code{ptr} (external pointer to the cached buffer type),
#'   \code{name} (its config-derived name) and \code{alloc_size} (sum of the
#'   padded per-device slice sizes for \code{probe}, in bytes; 0 if no probe).
#' @export
#' @examples
#' \donttest{
#' if (ggml_vulkan_available() && ggml_vulkan_device_count() >= 2) {
#'   bt <- ggml_vulkan_split_buffer_type(n_devices = 2, probe = c(2048, 64))
#'   bt$name
#'   bt$alloc_size   # total bytes for a 2048x64 f32 weight across 2 devices
#' }
#' }
ggml_vulkan_split_buffer_type <- function(n_devices = ggml_vulkan_device_count(),
                                          main_device = 0L, weights = NULL,
                                          device_ids = NULL,
                                          transport = c("host-staging", "opaque-fd", "device-group"),
                                          probe = NULL) {
  transport <- match.arg(transport)
  t_code <- switch(transport,
                   "host-staging" = 0L,
                   "opaque-fd"    = 1L,
                   "device-group" = 2L)
  if (!is.null(device_ids)) {
    device_ids <- as.integer(device_ids)
    n_devices  <- length(device_ids)
  }
  w <- if (is.null(weights)) NULL else as.numeric(weights)
  probe_N <- if (is.null(probe)) 0 else as.numeric(probe[1])
  probe_K <- if (is.null(probe)) 0 else as.numeric(probe[2])
  .Call("R_ggml_vk_split_buffer_type",
        as.integer(main_device), w, as.integer(n_devices), device_ids, t_code,
        probe_N, probe_K, PACKAGE = "ggmlR")
}

#' Initialize Vulkan backend
#'
#' Creates a Vulkan backend for the specified device.
#' The backend must be freed with ggml_vulkan_free() when done.
#'
#' @param device Device index (0-based, default 0)
#' @return Vulkan backend pointer
#' @export
#' @examples
#' \donttest{
#' if (ggml_vulkan_available() && ggml_vulkan_device_count() > 0) {
#'   backend <- ggml_vulkan_init(0)
#'   print(ggml_vulkan_backend_name(backend))
#'   ggml_vulkan_free(backend)
#' }
#' }
ggml_vulkan_init <- function(device = 0L) {
  .Call("R_ggml_vulkan_init", as.integer(device), PACKAGE = "ggmlR")
}

#' Free Vulkan backend
#'
#' Releases resources associated with the Vulkan backend.
#'
#' @param backend Vulkan backend pointer from ggml_vulkan_init()
#' @return NULL (invisible)
#' @export
#' @examples
#' \donttest{
#' if (ggml_vulkan_available() && ggml_vulkan_device_count() > 0) {
#'   backend <- ggml_vulkan_init(0)
#'   ggml_vulkan_free(backend)
#' }
#' }
ggml_vulkan_free <- function(backend) {
  invisible(.Call("R_ggml_vulkan_free", backend, PACKAGE = "ggmlR"))
}

#' Check if backend is Vulkan
#'
#' Returns TRUE if the given backend is a Vulkan backend.
#'
#' @param backend Backend pointer
#' @return Logical indicating if backend is Vulkan
#' @export
#' @examples
#' \donttest{
#' if (ggml_vulkan_available() && ggml_vulkan_device_count() > 0) {
#'   vk_backend <- ggml_vulkan_init(0)
#'   cpu_backend <- ggml_backend_cpu_init()
#'
#'   ggml_vulkan_is_backend(vk_backend)  # TRUE
#'   ggml_vulkan_is_backend(cpu_backend) # FALSE
#'
#'   ggml_vulkan_free(vk_backend)
#'   ggml_backend_free(cpu_backend)
#' }
#' }
ggml_vulkan_is_backend <- function(backend) {
  .Call("R_ggml_vulkan_is_backend", backend, PACKAGE = "ggmlR")
}

#' Get Vulkan backend name
#'
#' Returns the name of the Vulkan backend (includes device info).
#'
#' @param backend Vulkan backend pointer
#' @return Character string with backend name
#' @export
#' @examples
#' \donttest{
#' if (ggml_vulkan_available() && ggml_vulkan_device_count() > 0) {
#'   backend <- ggml_vulkan_init(0)
#'   print(ggml_vulkan_backend_name(backend))
#'   ggml_vulkan_free(backend)
#' }
#' }
ggml_vulkan_backend_name <- function(backend) {
  .Call("R_ggml_vulkan_backend_name", backend, PACKAGE = "ggmlR")
}

#' Get Vulkan device capabilities
#'
#' Returns hardware capabilities for the specified Vulkan device.
#'
#' @param device Device index (0-based, default 0)
#' @return Named list: coopmat_support, coopmat1_fa_support, fp16, subgroup_size, subgroup_no_shmem
#' @export
ggml_vulkan_device_caps <- function(device = 0L) {
  .Call("R_ggml_vulkan_device_caps", as.integer(device), PACKAGE = "ggmlR")
}

#' Print Vulkan status
#'
#' Prints information about Vulkan availability and devices.
#'
#' @return NULL (invisible), prints status to console
#' @export
#' @examples
#' ggml_vulkan_status()
ggml_vulkan_status <- function() {
  available <- ggml_vulkan_available()

  if (!available) {
    cat("Vulkan: NOT AVAILABLE\n")
    cat("  To enable: install libvulkan-dev and glslc, then reinstall ggmlR\n")
    cat("  Ubuntu/Debian: sudo apt install libvulkan-dev glslc\n")
    return(invisible(NULL))
  }

  count <- ggml_vulkan_device_count()
  cat("Vulkan: AVAILABLE\n")
  cat("  Devices:", count, "\n")

  if (count > 0) {
    devices <- ggml_vulkan_list_devices()
    for (i in seq_along(devices)) {
      dev <- devices[[i]]
      cat(sprintf("  [%d] %s\n", dev$index, dev$name))
      cat(sprintf("      Memory: %.2f GB free / %.2f GB total\n",
                  dev$free_memory / 1e9, dev$total_memory / 1e9))
    }
  }

  invisible(NULL)
}
