// Добавьте эти функции в существующий r_interface.c

// ============================================================================
// Memory Management Helpers
// ============================================================================

SEXP R_ggml_tensor_overhead(void) {
    // Размер метаданных GGML для одного тензора
    size_t overhead = ggml_tensor_overhead();
    return ScalarReal((double) overhead);
}

SEXP R_ggml_estimate_memory(SEXP type, SEXP ne0, SEXP ne1, SEXP ne2, SEXP ne3) {
    enum ggml_type dtype = (enum ggml_type) asInteger(type);
    int64_t n0 = length(ne0) > 0 ? (int64_t) asReal(ne0) : 1;
    int64_t n1 = length(ne1) > 0 ? (int64_t) asReal(ne1) : 1;
    int64_t n2 = length(ne2) > 0 ? (int64_t) asReal(ne2) : 1;
    int64_t n3 = length(ne3) > 0 ? (int64_t) asReal(ne3) : 1;
    
    // Количество элементов
    int64_t n_elements = n0 * n1 * n2 * n3;
    
    // Размер данных
    size_t type_size = ggml_type_size(dtype);
    size_t data_size = n_elements * type_size;
    
    // Общий размер с выравниванием и метаданными
    size_t total = data_size + ggml_tensor_overhead() + 256; // +256 для выравнивания
    
    return ScalarReal((double) total);
}

SEXP R_ggml_get_mem_size(SEXP ctx_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    if (ctx == NULL) {
        error("Invalid context pointer");
    }
    
    // Получить размер пула памяти контекста
    size_t mem_size = ggml_get_mem_size(ctx);
    return ScalarReal((double) mem_size);
}

SEXP R_ggml_used_mem(SEXP ctx_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    if (ctx == NULL) {
        error("Invalid context pointer");
    }
    
    // Получить использованную память
    size_t used = ggml_used_mem(ctx);
    return ScalarReal((double) used);
}
