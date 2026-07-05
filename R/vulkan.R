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
