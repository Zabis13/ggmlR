/* ggml-private.c — single compilation unit for ggml core.
 * Mirrors the pattern used by ggml-vulkan.cpp.
 * All sub-files share static symbols via #include. */

#include "ggml-core.c"
#include "ggml-context.c"
#include "ggml-ops-builders.c"
#include "ggml-graph.c"
#include "ggml-misc.c"
