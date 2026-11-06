args <- commandArgs(trailingOnly = TRUE)
dev <- args[1]; runs <- as.integer(args[2])
suppressMessages(library(ggmlR))

path  <- file.path(Sys.getenv("ONNX_DIR", "/mnt/Data2/DS_projects/ONNX models-main"),
                   "MaskRCNN-12-int8.onnx")
shape <- c(3L, 224L, 224L)

vm_kb <- function(key) {
  l <- grep(paste0("^", key, ":"), readLines("/proc/self/status"), value = TRUE)
  if (!length(l)) NA_real_ else as.numeric(gsub("[^0-9]", "", l))
}
t_ms <- function(f) { t0 <- Sys.time(); f(); as.numeric(Sys.time() - t0, units = "secs") * 1000 }

vk   <- dev == "vulkan" && ggml_vulkan_available()
mem0 <- if (vk) ggml_vulkan_device_memory(0)$free else NA
model <- NULL
t_load <- t_ms(function() model <<- onnx_load(path, device = dev, input_shapes = list(image = shape)))
set.seed(42)
inp <- list(image = runif(prod(shape)))
out <- NULL
t_first <- t_ms(function() out <<- onnx_run(model, inp))
det_first <- length(as.numeric(out[[1]])) %/% 4L
ts <- vapply(seq_len(runs), function(i) t_ms(function() out <<- onnx_run(model, inp)), numeric(1))
det_last <- length(as.numeric(out[[1]])) %/% 4L
mem1 <- if (vk) ggml_vulkan_device_memory(0)$free else NA

cat(sprintf("%s,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%d,%d,%.1f,%.1f,%.1f,%.1f\n",
    Sys.getenv("GGML_SCHED_MAX_SPLIT_INPUTS"), dev, t_load, t_first,
    median(ts), min(ts), max(ts), det_first, det_last,
    vm_kb("VmPeak") / 1024, vm_kb("VmSize") / 1024, vm_kb("VmHWM") / 1024,
    if (vk) (mem0 - mem1) / 2^20 else NA_real_))
