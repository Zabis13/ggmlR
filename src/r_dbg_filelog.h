/*
 * Crash-survivable diagnostic file logger.
 *
 * Problem: under the R package build, fprintf/printf/fputs/fflush and even
 * stderr/stdout are macro-redirected (see r_ggml_compat.h) to R-safe wrappers
 * that buffer through REprintf and never flush (r_ggml_fflush is a no-op).
 * When the process dies via std::terminate()->abort() (an unhandled C++
 * vk::SystemError crossing the C .Call boundary on MinGW), every buffered
 * message is lost -> the crash looks completely silent.
 *
 * This logger opens a file, writes one line, and CLOSES it on every call.
 * fopen/fwrite/fclose are NOT redirected by r_ggml_compat.h, and fclose
 * forces the line to disk, so it survives an abort that happens immediately
 * after. Use only for crash localization; remove once the bug is fixed.
 *
 * Enable by setting the GGMLR_DBG_LOG env var to a writable path, e.g.
 *   Sys.setenv(GGMLR_DBG_LOG="C:/models/ggmlr_dbg.log")
 * If unset, all calls are cheap no-ops.
 */
#ifndef R_DBG_FILELOG_H
#define R_DBG_FILELOG_H

/* Pull in the real stdio symbols. r_ggml_compat.h only macro-redirects
 * fprintf/printf/fputs/fflush/stderr/stdout — fopen/fwrite/fclose are intact. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline void r_dbg_logf(const char * fmt, ...) {
    const char * path = getenv("GGMLR_DBG_LOG");
    if (path == NULL || path[0] == '\0') {
        return;
    }
    FILE * f = fopen(path, "a");
    if (f == NULL) {
        return;
    }
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fwrite(buf, 1, strlen(buf), f);
    fwrite("\n", 1, 1, f);
    fclose(f); /* close == flush to disk; survives a subsequent abort() */
}

#ifdef __cplusplus
}
#endif

#endif /* R_DBG_FILELOG_H */
