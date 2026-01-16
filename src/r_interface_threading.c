
// ============================================================================
// Threading Control
// ============================================================================

SEXP R_ggml_set_n_threads(SEXP n_threads) {
    int threads = asInteger(n_threads);
    
    if (threads < 1) {
        error("Number of threads must be at least 1");
    }
    
    // GGML использует переменные окружения и внутренние настройки
    // Для CPU backend количество потоков устанавливается при создании backend
    // Но можем установить переменную окружения
    char buf[32];
    snprintf(buf, sizeof(buf), "%d", threads);
    setenv("OMP_NUM_THREADS", buf, 1);
    
    Rprintf("Set number of threads to %d\n", threads);
    
    return ScalarInteger(threads);
}

SEXP R_ggml_get_n_threads(void) {
    // Получить количество потоков из OpenMP
    #ifdef _OPENMP
    int threads = omp_get_max_threads();
    #else
    int threads = 1;
    #endif
    
    return ScalarInteger(threads);
}
