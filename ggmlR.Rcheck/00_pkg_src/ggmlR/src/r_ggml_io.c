/*
 * R-safe I/O implementation for ggml
 *
 * This file implements wrapper functions that redirect standard C I/O
 * to R's print functions. It includes R headers which is safe here
 * because this is a pure C file that doesn't use C++ standard library.
 */

/* We need to undef GGML_R_PACKAGE temporarily so we don't get the macro
 * redirections when including stdio.h for the FILE type definition */
#ifdef GGML_R_PACKAGE
#undef GGML_R_PACKAGE
#define WAS_GGML_R_PACKAGE
#endif

#include <stdio.h>
#include <stdarg.h>

#ifdef WAS_GGML_R_PACKAGE
#define GGML_R_PACKAGE
#undef WAS_GGML_R_PACKAGE
#endif

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Print.h>

/*
 * fprintf replacement
 * Redirects all output to R's error stream (REprintf) for safety
 */
int r_ggml_fprintf(FILE *stream, const char *format, ...) {
    (void)stream;
    va_list args;
    va_start(args, format);
    REvprintf(format, args);
    va_end(args);
    return 0;
}

int r_ggml_vfprintf(FILE *stream, const char *format, va_list args) {
    (void)stream;
    REvprintf(format, args);
    return 0;
}

/*
 * printf replacement
 * Uses R's standard print function
 */
int r_ggml_printf(const char *format, ...) {
    va_list args;
    va_start(args, format);
    Rvprintf(format, args);
    va_end(args);
    return 0;
}

int r_ggml_vprintf(const char *format, va_list args) {
    Rvprintf(format, args);
    return 0;
}

/*
 * puts replacement
 */
int r_ggml_puts(const char *s) {
    Rprintf("%s\n", s);
    return 0;
}

/*
 * putchar replacement
 */
int r_ggml_putchar(int c) {
    Rprintf("%c", c);
    return c;
}

/*
 * fflush replacement - R handles flushing internally
 */
int r_ggml_fflush(FILE *stream) {
    (void)stream;
    return 0;
}

/*
 * fputs replacement
 */
int r_ggml_fputs(const char *s, FILE *stream) {
    (void)stream;
    REprintf("%s", s);
    return 0;
}

/*
 * abort replacement
 * Uses R's error() which performs a longjmp back to R
 */
void r_ggml_abort(const char *file, int line, const char *msg) {
    Rf_error("ggml fatal error at %s:%d: %s", file, line, msg);
    /* Rf_error never returns, but compiler needs this for noreturn attribute */
    while(1) {}
}

/*
 * _Exit/exit replacement
 */
void r_ggml_exit(int status) {
    Rf_error("ggml: exit called with status %d", status);
    /* Rf_error never returns */
    while(1) {}
}
