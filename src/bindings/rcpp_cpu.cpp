#include <Rcpp.h>

// Импортируем C заголовки через extern "C"
extern "C" {
    #include "../ggml.h"
    #include "../ggml-common.h"
}

// Импортируем C++ заголовки
#include "../ggml-backend.h"

//' Test GGML CPU initialization
//' @export
// [[Rcpp::export]]
Rcpp::List test_ggml_cpu() {
    // Простой тест: создаём контекст
    struct ggml_init_params params = {
        /*.mem_size   =*/ 16*1024*1024,  // 16 MB
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    
    if (ctx == NULL) {
        Rcpp::stop("Failed to initialize GGML context");
    }
    
    // Получаем информацию о памяти
    size_t mem_used = ggml_used_mem(ctx);
    size_t mem_size = ggml_get_mem_size(ctx);
    
    // Освобождаем контекст
    ggml_free(ctx);
    
    return Rcpp::List::create(
        Rcpp::Named("success") = true,
        Rcpp::Named("mem_used") = (double)mem_used,
        Rcpp::Named("mem_size") = (double)mem_size,
        Rcpp::Named("backend") = "CPU"
    );
}
