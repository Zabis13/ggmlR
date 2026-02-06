# Vulkan GPU Backend Functions

#' Check if Vulkan support is available
#'
#' Returns TRUE if the package was compiled with Vulkan support.
#' To enable Vulkan, reinstall with: install.packages(..., configure.args = "--with-vulkan")
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
    cat("  To enable: reinstall with configure.args = \"--with-vulkan\"\n")
    cat("  Requirements: Vulkan SDK, glslc compiler\n")
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
