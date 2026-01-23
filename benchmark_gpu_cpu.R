#!/usr/bin/env Rscript
# ============================================================================
# GGMLR GPU vs CPU Performance Benchmark
# ============================================================================

library(ggmlR)

cat("╔════════════════════════════════════════════════════════════════╗\n")
cat("║        GGMLR Performance: GPU (Vulkan) vs CPU Benchmark       ║\n")
cat("╚════════════════════════════════════════════════════════════════╝\n\n")

# Определяем количество ядер CPU
n_cores <- parallel::detectCores()
cat(sprintf("CPU: Обнаружено ядер: %d\n", n_cores))

# Проверка Vulkan
vulkan_available <- ggml_vulkan_available()
cat(sprintf("GPU: Vulkan %s\n", ifelse(vulkan_available, "ДОСТУПЕН", "НЕ ДОСТУПЕН")))

if (vulkan_available) {
  n_devices <- ggml_vulkan_device_count()
  cat(sprintf("GPU: Найдено устройств: %d\n", n_devices))

  if (n_devices > 0) {
    gpu_name <- ggml_vulkan_device_description(0)
    gpu_mem <- ggml_vulkan_device_memory(0)
    cat(sprintf("GPU: %s\n", gpu_name))
    cat(sprintf("GPU: Память %.2f GB / %.2f GB\n",
                gpu_mem$free / 1e9, gpu_mem$total / 1e9))
  }
}

cat("\n")

# Функция для бенчмарка на CPU
benchmark_cpu <- function(size, iterations = 10) {
  # Используем as.numeric для избежания integer overflow
  mem_size <- as.numeric(size) * 4 * 4
  ctx <- ggml_init(mem_size = mem_size)
  ggml_set_no_alloc(ctx, TRUE)

  # Создаём тензоры
  t1 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size)
  t2 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size)
  t3 <- ggml_add(ctx, t1, t2)

  # CPU backend
  backend <- ggml_backend_cpu_init()
  ggml_backend_cpu_set_n_threads(backend, n_cores)
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  # Подготовка данных
  data1 <- rnorm(size)
  data2 <- rnorm(size)
  ggml_backend_tensor_set_data(t1, data1)
  ggml_backend_tensor_set_data(t2, data2)

  # Прогрев
  graph <- ggml_build_forward_expand(ctx, t3)
  ggml_backend_graph_compute(backend, graph)

  # Benchmark
  times <- numeric(iterations)
  for (i in 1:iterations) {
    start <- Sys.time()
    ggml_backend_graph_compute(backend, graph)
    times[i] <- as.numeric(Sys.time() - start)
  }

  # Проверка результата
  result <- ggml_backend_tensor_get_data(t3)

  # Cleanup
  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)

  list(
    mean_time = mean(times),
    min_time = min(times),
    max_time = max(times),
    sd_time = sd(times),
    gflops = size / mean(times) / 1e9,
    result = result[1:5]  # Первые 5 элементов для проверки
  )
}

# Функция для бенчмарка на GPU
benchmark_gpu <- function(size, iterations = 10) {
  if (!vulkan_available || ggml_vulkan_device_count() == 0) {
    return(NULL)
  }

  # Используем as.numeric для избежания integer overflow
  mem_size <- as.numeric(size) * 4 * 4
  ctx <- ggml_init(mem_size = mem_size)
  ggml_set_no_alloc(ctx, TRUE)

  # Создаём тензоры
  t1 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size)
  t2 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size)
  t3 <- ggml_add(ctx, t1, t2)

  # Vulkan backend
  backend <- ggml_vulkan_init(0)
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  # Подготовка данных
  data1 <- rnorm(size)
  data2 <- rnorm(size)
  ggml_backend_tensor_set_data(t1, data1)
  ggml_backend_tensor_set_data(t2, data2)

  # Прогрев
  graph <- ggml_build_forward_expand(ctx, t3)
  ggml_backend_graph_compute(backend, graph)

  # Benchmark
  times <- numeric(iterations)
  for (i in 1:iterations) {
    start <- Sys.time()
    ggml_backend_graph_compute(backend, graph)
    times[i] <- as.numeric(Sys.time() - start)
  }

  # Проверка результата
  result <- ggml_backend_tensor_get_data(t3)

  # Cleanup
  ggml_backend_buffer_free(buffer)
  ggml_vulkan_free(backend)
  ggml_free(ctx)

  list(
    mean_time = mean(times),
    min_time = min(times),
    max_time = max(times),
    sd_time = sd(times),
    gflops = size / mean(times) / 1e9,
    result = result[1:5]
  )
}

# ============================================================================
# Тест 1: Различные размеры тензоров
# ============================================================================
cat("═══ Тест 1: Сравнение на разных размерах ═══\n\n")

sizes <- c(1e6, 5e6, 1e7, 5e7, 1e8, 2e8, 5e8)
iterations <- 50

results_table <- data.frame(
  Size = character(),
  CPU_Time = numeric(),
  GPU_Time = numeric(),
  CPU_GFLOPS = numeric(),
  GPU_GFLOPS = numeric(),
  Speedup = numeric(),
  stringsAsFactors = FALSE
)

for (size in sizes) {
  size_mb <- size * 4 / 1024 / 1024
  cat(sprintf("Размер: %.0e элементов (%.1f MB)\n", size, size_mb))

  # CPU benchmark
  cat("  CPU: ")
  cpu_result <- benchmark_cpu(size, iterations)
  cat(sprintf("%.4f сек (%.2f GFLOPS)\n", cpu_result$mean_time, cpu_result$gflops))

  # GPU benchmark
  if (vulkan_available) {
    cat("  GPU: ")
    gpu_result <- benchmark_gpu(size, iterations)
    if (!is.null(gpu_result)) {
      cat(sprintf("%.4f сек (%.2f GFLOPS)\n", gpu_result$mean_time, gpu_result$gflops))

      speedup <- cpu_result$mean_time / gpu_result$mean_time
      cat(sprintf("  Ускорение: %.2fx %s\n", speedup,
                  ifelse(speedup > 1, "🚀", "⚠️")))

      # Проверка корректности
      if (max(abs(cpu_result$result - gpu_result$result)) < 1e-4) {
        cat("  Результаты: ✓ идентичны\n")
      } else {
        cat("  Результаты: ⚠️ отличаются\n")
      }

      results_table <- rbind(results_table, data.frame(
        Size = sprintf("%.0e", size),
        CPU_Time = cpu_result$mean_time,
        GPU_Time = gpu_result$mean_time,
        CPU_GFLOPS = cpu_result$gflops,
        GPU_GFLOPS = gpu_result$gflops,
        Speedup = speedup
      ))
    }
  } else {
    cat("  GPU: недоступен\n")
  }

  cat("\n")
}

# ============================================================================
# Тест 2: Матричные операции
# ============================================================================
if (vulkan_available) {
  cat("═══ Тест 2: Матричное умножение ═══\n\n")

  mat_sizes <- c(512, 1024, 2048)

  for (mat_size in mat_sizes) {
    n_elem <- mat_size * mat_size
    size_mb <- n_elem * 4 / 1024 / 1024

    cat(sprintf("Матрица: %dx%d (%.1f MB)\n", mat_size, mat_size, size_mb))

    # CPU
    mem_size_cpu <- as.numeric(n_elem) * 4 * 4
    ctx_cpu <- ggml_init(mem_size = mem_size_cpu)
    ggml_set_no_alloc(ctx_cpu, TRUE)

    m1_cpu <- ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, mat_size, mat_size)
    m2_cpu <- ggml_new_tensor_2d(ctx_cpu, GGML_TYPE_F32, mat_size, mat_size)
    m3_cpu <- ggml_mul_mat(ctx_cpu, m1_cpu, m2_cpu)

    backend_cpu <- ggml_backend_cpu_init()
    ggml_backend_cpu_set_n_threads(backend_cpu, n_cores)
    buffer_cpu <- ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu)

    data_m1 <- rnorm(n_elem)
    data_m2 <- rnorm(n_elem)
    ggml_backend_tensor_set_data(m1_cpu, data_m1)
    ggml_backend_tensor_set_data(m2_cpu, data_m2)

    graph_cpu <- ggml_build_forward_expand(ctx_cpu, m3_cpu)

    # Прогрев и замер
    ggml_backend_graph_compute(backend_cpu, graph_cpu)
    mat_iters <- if (mat_size <= 2048) 10 else if (mat_size <= 4096) 5 else 3
    time_cpu <- system.time({
      for (i in 1:mat_iters) {
        ggml_backend_graph_compute(backend_cpu, graph_cpu)
      }
    })[3] / mat_iters

    cat(sprintf("  CPU: %.4f сек (%.2f GFLOPS)\n", time_cpu,
                2 * mat_size^3 / time_cpu / 1e9))

    # GPU
    mem_size_gpu <- as.numeric(n_elem) * 4 * 4
    ctx_gpu <- ggml_init(mem_size = mem_size_gpu)
    ggml_set_no_alloc(ctx_gpu, TRUE)

    m1_gpu <- ggml_new_tensor_2d(ctx_gpu, GGML_TYPE_F32, mat_size, mat_size)
    m2_gpu <- ggml_new_tensor_2d(ctx_gpu, GGML_TYPE_F32, mat_size, mat_size)
    m3_gpu <- ggml_mul_mat(ctx_gpu, m1_gpu, m2_gpu)

    backend_gpu <- ggml_vulkan_init(0)
    buffer_gpu <- ggml_backend_alloc_ctx_tensors(ctx_gpu, backend_gpu)

    ggml_backend_tensor_set_data(m1_gpu, data_m1)
    ggml_backend_tensor_set_data(m2_gpu, data_m2)

    graph_gpu <- ggml_build_forward_expand(ctx_gpu, m3_gpu)

    # Прогрев и замер
    ggml_backend_graph_compute(backend_gpu, graph_gpu)
    time_gpu <- system.time({
      for (i in 1:mat_iters) {
        ggml_backend_graph_compute(backend_gpu, graph_gpu)
      }
    })[3] / mat_iters

    cat(sprintf("  GPU: %.4f сек (%.2f GFLOPS)\n", time_gpu,
                2 * mat_size^3 / time_gpu / 1e9))
    cat(sprintf("  Ускорение: %.2fx %s\n\n", time_cpu / time_gpu,
                ifelse(time_cpu > time_gpu, "🚀", "⚠️")))

    # Cleanup
    ggml_backend_buffer_free(buffer_cpu)
    ggml_backend_free(backend_cpu)
    ggml_free(ctx_cpu)

    ggml_backend_buffer_free(buffer_gpu)
    ggml_vulkan_free(backend_gpu)
    ggml_free(ctx_gpu)
  }
}

# ============================================================================
# Итоговая таблица
# ============================================================================
if (nrow(results_table) > 0) {
  cat("\n═══ Итоговая таблица результатов ═══\n\n")
  print(results_table, row.names = FALSE)

  cat("\n═══ Статистика ═══\n")
  cat(sprintf("Средняя производительность CPU: %.2f GFLOPS\n",
              mean(results_table$CPU_GFLOPS)))
  if (vulkan_available) {
    cat(sprintf("Средняя производительность GPU: %.2f GFLOPS\n",
                mean(results_table$GPU_GFLOPS)))
    cat(sprintf("Среднее ускорение GPU vs CPU: %.2fx\n",
                mean(results_table$Speedup)))
    cat(sprintf("Максимальное ускорение: %.2fx\n",
                max(results_table$Speedup)))
    cat(sprintf("Минимальное ускорение: %.2fx\n",
                min(results_table$Speedup)))
  }
}

cat("\n╔════════════════════════════════════════════════════════════════╗\n")
cat("║                         ТЕСТЫ ЗАВЕРШЕНЫ                        ║\n")
cat("╚════════════════════════════════════════════════════════════════╝\n")
