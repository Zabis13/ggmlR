# Vulkan Device Capabilities Inspector
# Run: Rscript inst/examples/vulkan_caps.R

library(ggmlR)

cat("=== Vulkan Device Capabilities ===\n\n")

# --- Availability ---
if (!ggml_vulkan_available()) {
  cat("Vulkan: NOT COMPILED\n")
  cat("  Reinstall: R CMD INSTALL . (requires libvulkan-dev + glslc)\n")
  quit(status = 1)
}

cat("Vulkan: compiled OK\n")

n <- ggml_vulkan_device_count()
cat("Devices found:", n, "\n\n")

if (n == 0) {
  cat("No Vulkan devices. Check driver installation.\n")
  quit(status = 1)
}

# --- Per-device info ---
for (i in seq_len(n)) {
  idx <- i - 1L

  desc <- ggml_vulkan_device_description(idx)
  mem  <- ggml_vulkan_device_memory(idx)
  caps <- ggml_vulkan_device_caps(idx)

  cat(sprintf("Device [%d]: %s\n", idx, desc))
  cat(sprintf("  Memory : %.2f GB free / %.2f GB total\n",
              mem$free / 1e9, mem$total / 1e9))
  cat("\n  --- Capabilities ---\n")

  cat(sprintf("  arch               : %s\n", caps$arch))

  cat(sprintf("  fp16               : %s",  if (caps$fp16) "YES" else "NO"))
  cat("   (FP16 arithmetic, required for fast inference)\n")

  cat(sprintf("  bf16               : %s",  if (caps$bf16) "YES" else "NO"))
  cat("   (BF16 native arithmetic; Flux/SD3 weights natively BF16)\n")

  cat(sprintf("  integer_dot_product: %s",  if (caps$integer_dot_product) "YES" else "NO"))
  cat("   (4x8-bit packed dot product; accelerates Q4/Q8 GEMM)\n")

  cat(sprintf("  coopmat_support    : %s",  if (caps$coopmat_support) "YES" else "NO"))
  cat("   (VK_KHR_cooperative_matrix, enables fast GEMM kernels)\n")

  cat(sprintf("  coopmat1_fa_support: %s",  if (caps$coopmat1_fa_support) "YES" else "NO"))
  cat("   (coopmat v1 flash-attention path, RDNA4/Ampere+)\n")

  cat(sprintf("  subgroup_size      : %d", caps$subgroup_size))
  cat("   (warp size used by ggml)\n")

  cat(sprintf("  subgroup_min_size  : %d", caps$subgroup_min_size))
  cat("   (min from VK_EXT_subgroup_size_control)\n")

  cat(sprintf("  subgroup_max_size  : %d", caps$subgroup_max_size))
  cat("   (max from VK_EXT_subgroup_size_control)\n")

  cat(sprintf("  subgroup_no_shmem  : %s",  if (caps$subgroup_no_shmem) "YES" else "NO"))
  cat("   (subgroup ops without shared memory, affects shader selection)\n")

  cat(sprintf("  wavefronts_per_simd: %d", caps$wavefronts_per_simd))
  cat("   (AMD only: 16=RDNA4, 20=RDNA1, 8=RDNA2/3; 0=non-AMD)\n")

  if (caps$coopmat_support && caps$coopmat_m > 0) {
    cat(sprintf("  coopmat tile       : M=%d N=%d K=%d", caps$coopmat_m, caps$coopmat_n, caps$coopmat_k))
    cat("   (KHR coopmat subgroup tile sizes)\n")
  }

  # --- Summary verdict ---
  cat("\n  --- Verdict ---\n")

  if (caps$fp16 && caps$coopmat1_fa_support) {
    cat("  BEST: coopmat flash-attention path active (fastest)\n")
  } else if (caps$fp16 && caps$coopmat_support) {
    cat("  GOOD: coopmat GEMM path active, no flash-attention\n")
  } else if (caps$fp16) {
    cat("  OK:   FP16 active, no coopmat (scalar/subgroup shaders)\n")
  } else {
    cat("  WARN: FP32 only — slow, check driver/device support\n")
  }

  # --- Subgroup-shuffle mmq pipeline ---
  cat("\n  --- Subgroup-shuffle mmq (Q4_K/Q5_K/Q6_K) ---\n")
  shuffle_ok <- caps$integer_dot_product && caps$subgroup_size >= 64L
  if (shuffle_ok) {
    cat(sprintf("  subgroup_no_shmem mmq : ACTIVE (subgroup_size=%d, Q4_K/Q5_K/Q6_K)\n",
                caps$subgroup_size))
  } else if (caps$integer_dot_product) {
    cat(sprintf("  USE_SUBGROUP_NO_SHMEM: INACTIVE (subgroup_size=%d < 64)\n",
                caps$subgroup_size))
    cat("    Standard mmq path (shmem staging) used instead\n")
  } else {
    cat("  USE_SUBGROUP_NO_SHMEM: INACTIVE (integer_dot_product not supported)\n")
    cat("    mmq path unavailable, using dequant+F32 matmul\n")
  }

  cat("\n")
}
