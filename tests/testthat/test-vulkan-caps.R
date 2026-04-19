library(ggmlR)

test_that("ggml_vulkan_device_caps returns expected fields", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if_not(ggml_vulkan_device_count() > 0, "No Vulkan devices")

  caps <- ggml_vulkan_device_caps(0L)

  expect_type(caps, "list")
  expect_named(caps, c("coopmat_support", "coopmat1_fa_support", "fp16", "bf16",
                        "integer_dot_product",
                        "subgroup_size", "subgroup_min_size", "subgroup_max_size",
                        "subgroup_no_shmem", "wavefronts_per_simd", "arch",
                        "coopmat_m", "coopmat_n", "coopmat_k",
                        "supports_256_push_constants", "max_push_constants_size"))

  expect_type(caps$coopmat_support,     "logical")
  expect_type(caps$coopmat1_fa_support, "logical")
  expect_type(caps$fp16,                "logical")
  expect_type(caps$bf16,                "logical")
  expect_type(caps$integer_dot_product, "logical")
  expect_type(caps$subgroup_size,       "integer")
  expect_type(caps$subgroup_min_size,   "integer")
  expect_type(caps$subgroup_max_size,   "integer")
  expect_type(caps$subgroup_no_shmem,   "logical")
  expect_type(caps$wavefronts_per_simd, "integer")
  expect_type(caps$arch,                "character")
  expect_type(caps$coopmat_m,                    "integer")
  expect_type(caps$coopmat_n,                    "integer")
  expect_type(caps$coopmat_k,                    "integer")
  expect_type(caps$supports_256_push_constants,  "logical")
  expect_type(caps$max_push_constants_size,      "integer")
})

test_that("RX 9070 (RDNA4) has coopmat1_fa and subgroup_no_shmem enabled", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if_not(ggml_vulkan_device_count() > 0, "No Vulkan devices")

  desc <- ggml_vulkan_device_description(0L)
  skip_if_not(grepl("9070|GFX1201|RDNA4|Radeon RX 9", desc, ignore.case = TRUE),
              paste("Not an RX 9070:", desc))

  caps <- ggml_vulkan_device_caps(0L)

  expect_equal(caps$arch,                "AMD_RDNA4",  label = "arch")
  expect_true(caps$coopmat_support,      label = "coopmat_support")
  expect_true(caps$coopmat1_fa_support,  label = "coopmat1_fa_support")
  expect_true(caps$fp16,                 label = "fp16")
  skip_if_not(caps$integer_dot_product,  "integer_dot_product not supported (glslc too old)")
  expect_true(caps$subgroup_no_shmem,    label = "subgroup_no_shmem")
  expect_gte(caps$subgroup_size,         32L)
  expect_equal(caps$subgroup_min_size,   32L,         label = "subgroup_min_size")
  expect_equal(caps$subgroup_max_size,   64L,         label = "subgroup_max_size")
})
