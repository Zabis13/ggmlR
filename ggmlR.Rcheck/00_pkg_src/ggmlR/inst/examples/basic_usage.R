library(ggmlR)

# Проверка версии
cat("GGML version:", ggml_version(), "\n")

# Тест библиотеки
ggml_test()

# Создание контекста
ctx <- ggml_init(16 * 1024 * 1024)  # 16MB
cat("Context created successfully\n")

# Освобождение ресурсов
ggml_free(ctx)
cat("Context freed\n")
