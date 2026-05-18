# Cooperative Matrix Configuration Inspector
#
# vulkaninfo only reports the coopmat *feature* flag, never the per-config
# list. The full set of supported tile shapes / data types / scopes is
# returned exclusively by vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR.
#
# This script writes a tiny standalone C probe, compiles it against the
# system Vulkan loader (no Vulkan SDK required, no ggmlR rebuild), runs it
# and presents the result as a data.frame.
#
# Run: Rscript inst/examples/coopmat_configs.R
#
# Answers: does this driver expose coopmat tile shapes beyond 16x16x16,
# BF16/FP8/INT8 variants, or a WORKGROUP (multi-subgroup) scope?

# ---------------------------------------------------------------------------
# 1. Locate a C compiler
# ---------------------------------------------------------------------------
cc <- Sys.which(c("cc", "gcc", "clang"))
cc <- cc[nzchar(cc)][1]
if (is.na(cc) || !nzchar(cc)) {
  stop("No C compiler (cc/gcc/clang) found in PATH.")
}
cat(sprintf("Compiler : %s\n", cc))

# ---------------------------------------------------------------------------
# 2. The C probe
# ---------------------------------------------------------------------------
# Dynamically loads libvulkan, enumerates every physical device and prints
# one CSV line per supported cooperative-matrix configuration.
c_src <- '
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The system Vulkan header may predate BF16 / FP8 component-type enums.
 * Those enum values are stable in the spec, so map the known numeric codes
 * explicitly rather than relying on #ifdef of header symbols. Anything still
 * unknown is printed as OTHER(<raw enum value>) so it can be identified. */
static const char *fmt_name(VkComponentTypeKHR t) {
    static char buf[24];
    switch ((int)t) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return "FP16";
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return "FP32";
        case VK_COMPONENT_TYPE_FLOAT64_KHR: return "FP64";
        case VK_COMPONENT_TYPE_SINT8_KHR:   return "SINT8";
        case VK_COMPONENT_TYPE_SINT16_KHR:  return "SINT16";
        case VK_COMPONENT_TYPE_SINT32_KHR:  return "SINT32";
        case VK_COMPONENT_TYPE_SINT64_KHR:  return "SINT64";
        case VK_COMPONENT_TYPE_UINT8_KHR:   return "UINT8";
        case VK_COMPONENT_TYPE_UINT16_KHR:  return "UINT16";
        case VK_COMPONENT_TYPE_UINT32_KHR:  return "UINT32";
        case VK_COMPONENT_TYPE_UINT64_KHR:  return "UINT64";
        /* spec-stable values, may be absent from an old header */
        case 1000141000: return "BF16";      /* VK_COMPONENT_TYPE_BFLOAT16_KHR    */
        case 1000491000: return "FP8_E4M3";  /* VK_COMPONENT_TYPE_FLOAT_E4M3_KHR  */
        case 1000491001: return "FP8_E5M2";  /* VK_COMPONENT_TYPE_FLOAT_E5M2_KHR  */
        default:
            snprintf(buf, sizeof(buf), "OTHER(%d)", (int)t);
            return buf;
    }
}

static const char *scope_name(VkScopeKHR s) {
    switch (s) {
        case VK_SCOPE_DEVICE_KHR:       return "DEVICE";
        case VK_SCOPE_WORKGROUP_KHR:    return "WORKGROUP";
        case VK_SCOPE_SUBGROUP_KHR:     return "SUBGROUP";
        case VK_SCOPE_QUEUE_FAMILY_KHR: return "QUEUE_FAMILY";
        default: return "OTHER";
    }
}

int main(void) {
    VkApplicationInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici = {0};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &ai;

    VkInstance inst;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) {
        fprintf(stderr, "vkCreateInstance failed\\n");
        return 2;
    }

    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR pfn =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!pfn) {
        fprintf(stderr, "VK_KHR_cooperative_matrix entry point not available\\n");
        vkDestroyInstance(inst, NULL);
        return 3;
    }

    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(inst, &ndev, NULL);
    VkPhysicalDevice *devs = malloc(sizeof(VkPhysicalDevice) * ndev);
    vkEnumeratePhysicalDevices(inst, &ndev, devs);

    /* CSV header */
    printf("device,name,M,N,K,A,B,C,Result,scope,saturating\\n");

    for (uint32_t d = 0; d < ndev; d++) {
        VkPhysicalDeviceProperties dp;
        vkGetPhysicalDeviceProperties(devs[d], &dp);

        uint32_t nprop = 0;
        pfn(devs[d], &nprop, NULL);
        if (nprop == 0) continue;

        VkCooperativeMatrixPropertiesKHR *props =
            malloc(sizeof(VkCooperativeMatrixPropertiesKHR) * nprop);
        for (uint32_t i = 0; i < nprop; i++) {
            memset(&props[i], 0, sizeof(props[i]));
            props[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        }
        pfn(devs[d], &nprop, props);

        for (uint32_t i = 0; i < nprop; i++) {
            VkCooperativeMatrixPropertiesKHR *p = &props[i];
            printf("%u,\\"%s\\",%u,%u,%u,%s,%s,%s,%s,%s,%s\\n",
                   d, dp.deviceName,
                   p->MSize, p->NSize, p->KSize,
                   fmt_name(p->AType), fmt_name(p->BType),
                   fmt_name(p->CType), fmt_name(p->ResultType),
                   scope_name(p->scope),
                   p->saturatingAccumulation ? "true" : "false");
        }
        free(props);
    }

    free(devs);
    vkDestroyInstance(inst, NULL);
    return 0;
}
'

# ---------------------------------------------------------------------------
# 3. Compile & run
# ---------------------------------------------------------------------------
tmp_dir <- tempfile("coopmat_probe_")
dir.create(tmp_dir)
on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

c_file   <- file.path(tmp_dir, "probe.c")
bin_file <- file.path(tmp_dir, "probe")
writeLines(c_src, c_file)

cat("Compiling probe (system Vulkan loader, no SDK needed)...\n")
compile <- suppressWarnings(system2(
  cc,
  c(shQuote(c_file), "-o", shQuote(bin_file), "-lvulkan", "-O1"),
  stdout = TRUE, stderr = TRUE
))
if (!file.exists(bin_file)) {
  cat("Compilation failed:\n")
  cat(paste(compile, collapse = "\n"), "\n")
  cat("\nNeed: libvulkan-dev (headers + libvulkan.so).\n")
  quit(status = 1)
}

cat("Running probe...\n\n")
out <- suppressWarnings(system2(bin_file, stdout = TRUE, stderr = TRUE))
status <- attr(out, "status")

if (!is.null(status) && status != 0) {
  cat("Probe exited with status", status, ":\n")
  cat(paste(out, collapse = "\n"), "\n")
  if (any(grepl("cooperative_matrix entry point not available", out))) {
    cat("\n=> Driver does not expose VK_KHR_cooperative_matrix.\n")
  }
  quit(status = 1)
}

# ---------------------------------------------------------------------------
# 4. Parse & present
# ---------------------------------------------------------------------------
csv_lines <- out[grepl("^[0-9]", out) | grepl("^device,", out)]
if (length(csv_lines) <= 1) {
  cat("No cooperative-matrix configurations reported by any device.\n")
  cat("(Driver lists the feature flag but enumerates 0 configs.)\n")
  quit(status = 0)
}

df <- utils::read.csv(text = paste(csv_lines, collapse = "\n"),
                      stringsAsFactors = FALSE)

cat("=== Cooperative Matrix Configurations ===\n\n")
print(df, row.names = FALSE)

# ---------------------------------------------------------------------------
# 5. Analysis relevant to ggmlR shader selection
# ---------------------------------------------------------------------------
# Software rasterizers (llvmpipe / lavapipe / SwiftShader) expose their own
# coopmat configs that are irrelevant to GPU shader selection. Analyse the
# real GPU(s) only; fall back to all devices if nothing else is present.
is_sw <- grepl("llvmpipe|lavapipe|swiftshader", df$name, ignore.case = TRUE)
gpu_df <- if (any(!is_sw)) df[!is_sw, , drop = FALSE] else df

cat("\n--- Analysis (hardware GPU devices only) ---\n")
if (any(is_sw)) {
  sw_names <- paste(unique(df$name[is_sw]), collapse = ", ")
  cat(sprintf("Excluded software rasterizer(s): %s\n", sw_names))
}

tiles <- unique(gpu_df[, c("M", "N", "K")])
cat(sprintf("Distinct tile shapes : %d\n", nrow(tiles)))
for (i in seq_len(nrow(tiles))) {
  cat(sprintf("  %dx%dx%d\n", tiles$M[i], tiles$N[i], tiles$K[i]))
}

df <- gpu_df
in_types <- sort(unique(c(df$A, df$B)))
cat(sprintf("Input types          : %s\n", paste(in_types, collapse = ", ")))

acc_types <- sort(unique(c(df$C, df$Result)))
cat(sprintf("Accumulator types    : %s\n", paste(acc_types, collapse = ", ")))

scopes <- sort(unique(df$scope))
cat(sprintf("Scopes               : %s\n", paste(scopes, collapse = ", ")))

if ("WORKGROUP" %in% scopes) {
  cat("  -> WORKGROUP scope present: multi-subgroup tiles available\n")
  cat("     (path to larger effective tiles, e.g. 32x32 / 64x16)\n")
} else {
  cat("  -> SUBGROUP scope only: tile size bounded by one subgroup\n")
}

has_bf16 <- "BF16"   %in% in_types
has_fp8  <- any(grepl("FP8", in_types))
has_int8 <- any(c("SINT8", "UINT8") %in% in_types)
cat(sprintf("BF16 coopmat         : %s\n", if (has_bf16) "YES" else "NO"))
cat(sprintf("FP8  coopmat         : %s\n", if (has_fp8)  "YES" else "NO"))
cat(sprintf("INT8 coopmat         : %s\n", if (has_int8) "YES" else "NO"))

non_16 <- tiles[!(tiles$M == 16 & tiles$N == 16 & tiles$K == 16), , drop = FALSE]
if (nrow(non_16) > 0) {
  cat("\nNOTE: tile shapes beyond 16x16x16 exist - ggmlR currently only\n")
  cat("      reports the first config in vulkan_caps.R. Worth checking\n")
  cat("      whether the Vulkan backend picks an optimal shape.\n")
}
