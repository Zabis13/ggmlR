#' Get Tensor Overhead
#' 
#' Returns the memory overhead (metadata) for each tensor in bytes
#'
#' @return Size in bytes
#' @export
#' @examples
#' \donttest{
#' ggml_tensor_overhead()
#' }
ggml_tensor_overhead <- function() {
  .Call("R_ggml_tensor_overhead")
}

#' Get Context Memory Size
#'
#' Returns the total memory pool size of the context
#'
#' @param ctx GGML context
#' @return Total memory size in bytes
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' ggml_get_mem_size(ctx)
#' ggml_free(ctx)
#' }
ggml_get_mem_size <- function(ctx) {
  .Call("R_ggml_get_mem_size", ctx)
}

#' Get Used Memory
#'
#' Returns the amount of memory currently used in the context
#'
#' @param ctx GGML context
#' @return Used memory in bytes
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
#' ggml_used_mem(ctx)
#' ggml_free(ctx)
#' }
ggml_used_mem <- function(ctx) {
  .Call("R_ggml_used_mem", ctx)
}

#' Set No Allocation Mode
#'
#' When enabled, tensor creation will not allocate memory for data.
#' Useful for creating computation graphs without allocating storage.
#'
#' @param ctx GGML context
#' @param no_alloc Logical, TRUE to disable allocation
#' @return NULL (invisible)
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' ggml_set_no_alloc(ctx, TRUE)
#' ggml_get_no_alloc(ctx)
#' ggml_set_no_alloc(ctx, FALSE)
#' ggml_free(ctx)
#' }
ggml_set_no_alloc <- function(ctx, no_alloc) {
  invisible(.Call("R_ggml_set_no_alloc", ctx, as.logical(no_alloc)))
}

#' Get No Allocation Mode
#'
#' Check if no-allocation mode is enabled
#'
#' @param ctx GGML context
#' @return Logical indicating if no_alloc is enabled
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' ggml_get_no_alloc(ctx)
#' ggml_free(ctx)
#' }
ggml_get_no_alloc <- function(ctx) {
  .Call("R_ggml_get_no_alloc", ctx)
}

#' Get Maximum Tensor Size
#'
#' Returns the maximum tensor size that can be allocated in the context
#'
#' @param ctx GGML context
#' @return Maximum tensor size in bytes
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' ggml_get_max_tensor_size(ctx)
#' ggml_free(ctx)
#' }
ggml_get_max_tensor_size <- function(ctx) {
  .Call("R_ggml_get_max_tensor_size", ctx)
}

#' Print Objects in Context
#'
#' Debug function to print all objects (tensors) in the context
#'
#' @param ctx GGML context
#' @return NULL (invisible)
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
#' ggml_print_objects(ctx)
#' ggml_free(ctx)
#' }
ggml_print_objects <- function(ctx) {
  invisible(.Call("R_ggml_print_objects", ctx))
}

#' Estimate Required Memory
#' 
#' Helper function to estimate memory needed for a tensor
#'
#' @param type Tensor type (GGML_TYPE_F32, etc)
#' @param ne0 Size of dimension 0
#' @param ne1 Size of dimension 1 (optional)
#' @param ne2 Size of dimension 2 (optional)
#' @param ne3 Size of dimension 3 (optional)
#' @return Estimated memory in bytes
#' @export
#' @examples
#' \donttest{
#' # For 1000x1000 F32 matrix
#' ggml_estimate_memory(GGML_TYPE_F32, 1000, 1000)
#' }
ggml_estimate_memory <- function(type = GGML_TYPE_F32, ne0, ne1 = 1, ne2 = 1, ne3 = 1) {
  # Calculate manually
  n_elements <- ne0 * ne1 * ne2 * ne3
  
  # F32 = 4 bytes
  type_size <- switch(as.character(type),
                     "0" = 4,  # GGML_TYPE_F32
                     4)
  
  data_size <- n_elements * type_size
  overhead <- ggml_tensor_overhead()
  alignment <- 256
  
  total <- data_size + overhead + alignment
  return(total)
}

#' Print Context Memory Status
#'
#' Helper to print memory usage information
#'
#' @param ctx GGML context
#' @return List with total, used, free memory (invisible)
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' ggml_print_mem_status(ctx)
#' ggml_free(ctx)
#' }
ggml_print_mem_status <- function(ctx) {
  total <- ggml_get_mem_size(ctx)
  used <- ggml_used_mem(ctx)
  free <- total - used
  
  cat(sprintf("GGML Context Memory Status:\n"))
  cat(sprintf("  Total: %.2f MB\n", total / (1024*1024)))
  cat(sprintf("  Used:  %.2f MB (%.1f%%)\n", used / (1024*1024), 100 * used / total))
  cat(sprintf("  Free:  %.2f MB (%.1f%%)\n", free / (1024*1024), 100 * free / total))
  
  invisible(list(total = total, used = used, free = free))
}
