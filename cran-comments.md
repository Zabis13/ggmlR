## Submission (0.8.10)

This is a resubmission addressing the NOTE on r-devel-windows-x86_64:

    Overall checktime 11 min > 10 min

Most of the check time on that platform is spent compiling the bundled
GGML C/C++ library, which we cannot reduce without dropping functionality.
We have reduced the remaining check time instead:

* **Documentation**: the manual was reduced from 866 to 544 Rd pages
  (Rd source from ~37,600 to ~22,800 lines). Closely related functions
  (in-place variants, 1D-5D views/reshapes, type/op information,
  backend and scheduler accessors, quantization routines, etc.) now share
  a single help page via `@rdname`; internal helpers are no longer
  rendered (`@noRd`); large `@family` cross-link lists, which grew
  quadratically with family size, were replaced by overview pages. Locally
  this cuts PDF manual build time by ~46% and HTML/Rd processing by ~50%.

* **Tests**: twelve more slow test files were moved to the extended suite
  that runs only when `NOT_CRAN=true`; the remaining tests run with the
  2-thread cap required by CRAN policy.

All examples together run in under 2 seconds locally.

## R CMD check results

0 errors | 0 warnings | N notes   <!-- TODO: fill in from new runs -->

* **installed package size**: The package includes a static library (lib/)
  for downstream packages (llamaR, sd2R) and a shared library (libs/) with
  the GGML tensor computation engine. The size is inherent to the compiled
  C/C++ codebase.

* **GNU make is a SystemRequirements**: GNU make is listed in
  SystemRequirements and is needed for the build.

* **valgrind "possibly lost: 352 bytes in 1 block"**: false positive. The
  block is thread-local storage allocated by glibc (`_dl_allocate_tls`)
  for an OpenMP worker thread that libgomp creates on the first parallel
  `ggml_graph_compute` and keeps alive in its persistent thread pool until
  process exit. `definitely lost` and `indirectly lost` are both 0 bytes.

## Test environments

<!-- TODO: fill in from new runs -->
* R-hub: clang23, gcc13, windows, macos-arm64, linux, valgrind, rchk
* Local: Linux Mint 22.3, R 4.3.3, GCC 13.3.0

## Downstream dependencies

<!-- TODO: fill in after checking -->
* llamaR
* sd2R
