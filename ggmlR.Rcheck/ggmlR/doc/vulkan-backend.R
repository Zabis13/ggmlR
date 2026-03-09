## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = FALSE
)

## ----install, eval = FALSE----------------------------------------------------
# install.packages("ggmlR")
# # or force Vulkan:
# install.packages("ggmlR", configure.args = "--with-vulkan")

## ----check--------------------------------------------------------------------
# library(ggmlR)
# 
# ggml_vulkan_available()
# #> TRUE
# 
# ggml_vulkan_status()
# #> Vulkan: AVAILABLE
# #>   Devices: 1
# #>   [0] NVIDIA GeForce RTX 4090
# #>       Memory: 23.56 GB free / 24.00 GB total

## ----model--------------------------------------------------------------------
# model <- ggml_model_sequential(input_shape = c(28, 28, 1)) |>
#   ggml_layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") |>
#   ggml_layer_flatten() |>
#   ggml_layer_dense(units = 10, activation = "softmax")
# 
# # backend = "auto" uses GPU if available, CPU otherwise
# model <- ggml_compile(model, optimizer = "adamw", loss = "cross_entropy")

## ----backend------------------------------------------------------------------
# model <- ggml_compile(model, optimizer = "adamw",
#                       loss = "cross_entropy", backend = "vulkan")
# model <- ggml_compile(model, optimizer = "adamw",
#                       loss = "cross_entropy", backend = "cpu")

