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
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Core: append one line to `path` (open+write+close each call so it survives a
 * subsequent abort/segfault). If path is "-" or "stderr", write to stderr via a
 * real (non-redirected) fd. Cheap no-op when path is NULL/empty. */
static inline void r_dbg_logf_to(const char * path, const char * fmt, va_list ap) {
    if (path == NULL || path[0] == '\0') {
        return;
    }
    char buf[512];
    vsnprintf(buf, sizeof(buf), fmt, ap);
    if (strcmp(path, "-") == 0 || strcmp(path, "stderr") == 0) {
        /* fileno(stderr)=2; write(2,...) is not macro-redirected and is unbuffered. */
        FILE * f = fopen("/dev/stderr", "a");
        if (f) { fwrite(buf, 1, strlen(buf), f); fwrite("\n", 1, 1, f); fclose(f); }
        return;
    }
    FILE * f = fopen(path, "a");
    if (f == NULL) {
        return;
    }
    fwrite(buf, 1, strlen(buf), f);
    fwrite("\n", 1, 1, f);
    fclose(f); /* close == flush to disk; survives a subsequent abort() */
}

static inline void r_dbg_logf(const char * fmt, ...) {
    const char * path = getenv("GGMLR_DBG_LOG");
    if (path == NULL || path[0] == '\0') {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    r_dbg_logf_to(path, fmt, ap);
    va_end(ap);
}

/* ggmlR TP: separate channel for tensor-parallel / P2P / teardown tracing.
 * Enable with GGMLR_TP_TRACE=<path> (or GGMLR_TP_TRACE=- for stderr). Silent
 * when unset. Kept in the tree at key P2P/destructor points for crash
 * localization; normal runs pay only one getenv per call. */
static inline void r_tp_tracef(const char * fmt, ...) {
    const char * path = getenv("GGMLR_TP_TRACE");
    if (path == NULL || path[0] == '\0') {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    r_dbg_logf_to(path, fmt, ap);
    va_end(ap);
}

#ifdef __cplusplus
}
#endif

#endif /* R_DBG_FILELOG_H */
