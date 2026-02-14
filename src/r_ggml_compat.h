/*
 * R compatibility header for ggml
 *
 * This header redirects standard C I/O and error functions to R-safe alternatives.
 * It must be force-included via -include flag before any other headers.
 *
 * Addresses CRAN policy requirements:
 * - No direct writes to stdout/stderr
 * - No abort() or _Exit() calls that terminate R
 *
 * IMPORTANT: This header avoids including R.h/Rinternals.h directly because
 * R's macros (like 'length', 'error') conflict with C++ standard library.
 * Instead, we declare extern functions that are implemented in r_ggml_io.c
 */

// Windows MSYS2 GCC compatibility (CRAN requirement)
#ifndef PRIuSZ
#if defined(_WIN32) || defined(__MINGW32__) || defined(__MINGW64__)
#define PRIdSZ  "I64d"
#define PRIuSZ  "I64u"
#define PRIxSZ  "I64x"
#define PRIoSZ  "I64o"
#else
#define PRIdSZ  "zd"
#define PRIuSZ  "zu"
#define PRIxSZ  "zx"
#define PRIoSZ  "zo"
#endif
#endif



#ifndef R_GGML_COMPAT_H
#define R_GGML_COMPAT_H

/* Only apply redirects when building for R
 * Skip if R_GGML_IO_IMPL is defined (used when compiling r_ggml_io.c) */
#if defined(GGML_R_PACKAGE) && !defined(R_GGML_IO_IMPL)

/* Include standard headers FIRST to get their declarations
 * before we override with macros */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

/*
 * Pre-define GGML_ATTRIBUTE_FORMAT as empty to avoid format warnings.
 * The 'printf' token in __attribute__((format(printf, ...))) would get
 * macro-replaced to 'r_ggml_printf' (which GCC doesn't recognize as a
 * valid format function archetype). Disabling format checking is acceptable
 * since it's optional compiler verification, not runtime functionality.
 */
#define GGML_ATTRIBUTE_FORMAT(...)

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Format attribute macro for GCC/Clang
 * This tells the compiler our functions use printf-style format strings.
 * On MinGW, use gnu_printf to accept C99 format specifiers (%zu, %zd)
 * that MSVCRT-based printf checker does not recognize.
 */
#if defined(__GNUC__) || defined(__clang__)
#  if defined(__MINGW32__) || defined(__MINGW64__)
#    define R_GGML_FORMAT_PRINTF(fmt_idx, first_arg) __attribute__((format(gnu_printf, fmt_idx, first_arg)))
#  else
#    define R_GGML_FORMAT_PRINTF(fmt_idx, first_arg) __attribute__((format(printf, fmt_idx, first_arg)))
#  endif
#  define R_GGML_NORETURN __attribute__((noreturn))
#else
#  define R_GGML_FORMAT_PRINTF(fmt_idx, first_arg)
#  define R_GGML_NORETURN
#endif

/*
 * Wrapper functions implemented in r_ggml_io.c
 * These call R's Rprintf/REprintf/Rf_error internally
 * Using FILE* for stream parameters for compatibility with stdio.h declarations
 */
R_GGML_FORMAT_PRINTF(2, 3)
int r_ggml_fprintf(FILE *stream, const char *format, ...);
int r_ggml_vfprintf(FILE *stream, const char *format, va_list args);
R_GGML_FORMAT_PRINTF(1, 2)
int r_ggml_printf(const char *format, ...);
int r_ggml_vprintf(const char *format, va_list args);
int r_ggml_puts(const char *s);
int r_ggml_putchar(int c);
int r_ggml_fflush(FILE *stream);
int r_ggml_fputs(const char *s, FILE *stream);
R_GGML_NORETURN
void r_ggml_abort(const char *file, int line, const char *msg);
R_GGML_NORETURN
void r_ggml_exit(int status);

#ifdef __cplusplus
}
#endif

/*
 * Macro redirections
 * Note: These redefine standard library functions to use our R-safe wrappers
 */

/* I/O redirections */
#undef fprintf
#define fprintf r_ggml_fprintf

#undef vfprintf
#define vfprintf r_ggml_vfprintf

#undef printf
#define printf r_ggml_printf

#undef vprintf
#define vprintf r_ggml_vprintf

#undef puts
#define puts r_ggml_puts

#undef putchar
#define putchar r_ggml_putchar

#undef fflush
#define fflush r_ggml_fflush

#undef fputs
#define fputs r_ggml_fputs

/*
 * Redirect stdout/stderr to NULL
 * Our wrapper functions (r_ggml_fprintf, etc.) ignore the stream parameter,
 * so replacing these with NULL is safe and removes the global symbols
 * that trigger CRAN NOTE about stdout/stderr usage in compiled code.
 */
#undef stdout
#define stdout ((FILE *) 0)
#undef stderr
#define stderr ((FILE *) 0)

/*
 * Redirect abort() and _Exit() to R error
 * Note: abort() must be noreturn, so we use a wrapper that calls Rf_error
 * The do-while(0) wrapper + noreturn attribute ensures compiler knows this never returns
 */
#undef abort
#define abort() do { r_ggml_abort(__FILE__, __LINE__, "abort called"); __builtin_unreachable(); } while(0)

#undef exit
#define exit(status) r_ggml_exit(status)

#undef _Exit
#define _Exit(status) r_ggml_exit(status)

#endif /* GGML_R_PACKAGE && !R_GGML_IO_IMPL */

#endif /* R_GGML_COMPAT_H */
