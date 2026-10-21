## Submission (0.8.8.1)

This release fixes two test failures reported on
r-devel-linux-x86_64-fedora-gcc for 0.8.8 (test-ag-tape-memory.R).

`ag_tape_memory()` identified shared objects via `tracemem()`, which errors
on R builds configured without memory profiling. Deduplication was then
silently disabled and shared storage was counted several times. The object
address is now read through a small registered C routine, independent of
the R build configuration.

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
