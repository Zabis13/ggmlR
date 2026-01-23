#' Set Number of Threads
#' 
#' Set the number of threads for GGML operations
#'
#' @param n_threads Number of threads to use
#' @return Number of threads set
#' @export
#' @examples
#' \dontrun{
#' # Use 4 threads
#' ggml_set_n_threads(4)
#'
#' # Use all available cores
#' ggml_set_n_threads(parallel::detectCores())
#' }
ggml_set_n_threads <- function(n_threads) {
  .Call("R_ggml_set_n_threads", as.integer(n_threads))
}

#' Get Number of Threads
#' 
#' Get the current number of threads for GGML operations
#'
#' @return Number of threads
#' @export
#' @examples
#' \dontrun{
#' ggml_get_n_threads()
#' }
ggml_get_n_threads <- function() {
  .Call("R_ggml_get_n_threads")
}
