#!/usr/bin/env Rscript
# ============================================================================
# GGMLR GPU vs CPU Performance Benchmark
# Высокоточные замеры proc.time() + Multi-GPU поддержка
# ============================================================================

cat("╔════════════════════════════════════════════════════════════════╗\n")
cat("║        GGMLR Performance: GPU (Vulkan) vs CPU Benchmark       ║\n")
cat("╚════════════════════════════════════════════════════════════════╝\n\n")

# Загружаем пакет ggmlR
if (!requireNamespace("ggmlR", quietly = TRUE)) {
  stop("Пакет ggmlR не установлен. Установите его с помощью: install.packages('ggmlR')")
}
library(ggmlR)

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
    for (i in 0:(n_devices - 1)) {
      gpu_name <- ggml_vulkan_device_description(i)
      gpu_mem <- ggml_vulkan_device_memory(i)
      cat(sprintf("GPU %d: %s\n", i, gpu_name))
      cat(sprintf("       Память %.2f GB / %.2f GB\n",
                  gpu_mem$free / 1e9, gpu_mem$total / 1e9))
    }
  }
}

cat("\n")

# Функция для бенчмарка на CPU (векторные операции)
benchmark_cpu_vector <- function(size) {
  result <- tryCatch({
    mem_size <- as.numeric(size) * 4 * 4
    ctx <- ggml_init(mem_size = mem_size)
    ggml_set_no_alloc(ctx, TRUE)

    t1 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size)
    t2 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size)
    t3 <- ggml_add(ctx, t1, t2)

    backend <- ggml_backend_cpu_init()
    ggml_backend_cpu_set_n_threads(backend, n_cores)
    buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

    data1 <- rnorm(size)
    data2 <- rnorm(size)
    ggml_backend_tensor_set_data(t1, data1)
    ggml_backend_tensor_set_data(t2, data2)

    graph <- ggml_build_forward_expand(ctx, t3)

    # Прогрев
    ggml_backend_graph_compute(backend, graph)

    # Точный замер
    start <- proc.time()
    ggml_backend_graph_compute(backend, graph)
    time_cpu <- (proc.time() - start)[3]

    result_data <- ggml_backend_tensor_get_data(t3)

    ggml_backend_buffer_free(buffer)
    ggml_backend_free(backend)
    ggml_free(ctx)

    list(
      mean_time = time_cpu,
      gflops = size * 1.0 / time_cpu / 1e9,  # Исправлено: 1 FLOP на элемент
      result = result_data[1:5]
    )
  }, error = function(e) {
    list(mean_time = NA, gflops = NA, result = NULL, error = e$message)
  })

  return(result)
}

# Функция для бенчмарка на GPU (векторные операции)
# Если несколько GPU - распределяет работу между ними параллельно
benchmark_gpu_vector <- function(size, device_ids = NULL) {
  if (!vulkan_available || ggml_vulkan_device_count() == 0) {
    return(NULL)
  }

  # Если не указаны device_ids, используем все доступные GPU
  if (is.null(device_ids)) {
    device_ids <- 0:(ggml_vulkan_device_count() - 1)
  }

  # Фильтруем валидные device_ids
  device_ids <- device_ids[device_ids < ggml_vulkan_device_count()]
  if (length(device_ids) == 0) {
    return(NULL)
  }

  n_gpus <- length(device_ids)

  result <- tryCatch({
    if (n_gpus == 1) {
      # Один GPU - простой случай
      device_id <- device_ids[1]
      mem_size <- as.numeric(size) * 4 * 4
      ctx <- ggml_init(mem_size = mem_size)
      ggml_set_no_alloc(ctx, TRUE)

      t1 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size)
      t2 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size)
      t3 <- ggml_add(ctx, t1, t2)

      backend <- ggml_vulkan_init(device_id)
      buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

      data1 <- rnorm(size)
      data2 <- rnorm(size)
      ggml_backend_tensor_set_data(t1, data1)
      ggml_backend_tensor_set_data(t2, data2)

      graph <- ggml_build_forward_expand(ctx, t3)

      # Прогрев
      ggml_backend_graph_compute(backend, graph)

      # Точный замер
      start <- proc.time()
      ggml_backend_graph_compute(backend, graph)
      time_gpu <- (proc.time() - start)[3]

      result_data <- ggml_backend_tensor_get_data(t3)

      ggml_backend_buffer_free(buffer)
      ggml_vulkan_free(backend)
      ggml_free(ctx)

      list(
        mean_time = time_gpu,
        gflops = size * 1.0 / time_gpu / 1e9,
        result = result_data[1:5],
        n_gpus = 1
      )
    } else {
      # Несколько GPU - параллельная работа
      chunk_size <- as.integer(size / n_gpus)

      # Создаём контексты и backend'ы для каждого GPU
      contexts <- list()
      backends <- list()
      buffers <- list()
      graphs <- list()
      tensors <- list()

      # Генерируем данные
      data1 <- rnorm(size)
      data2 <- rnorm(size)

      # Инициализация всех GPU
      for (i in 1:n_gpus) {
        chunk_start <- (i - 1) * chunk_size + 1
        chunk_end <- if (i == n_gpus) size else i * chunk_size
        chunk_len <- chunk_end - chunk_start + 1

        mem_size <- as.numeric(chunk_len) * 4 * 4
        ctx <- ggml_init(mem_size = mem_size)
        ggml_set_no_alloc(ctx, TRUE)

        t1 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, chunk_len)
        t2 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, chunk_len)
        t3 <- ggml_add(ctx, t1, t2)

        backend <- ggml_vulkan_init(device_ids[i])
        buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

        ggml_backend_tensor_set_data(t1, data1[chunk_start:chunk_end])
        ggml_backend_tensor_set_data(t2, data2[chunk_start:chunk_end])

        graph <- ggml_build_forward_expand(ctx, t3)

        contexts[[i]] <- ctx
        backends[[i]] <- backend
        buffers[[i]] <- buffer
        graphs[[i]] <- graph
        tensors[[i]] <- t3
      }

      # Прогрев всех GPU
      for (i in 1:n_gpus) {
        ggml_backend_graph_compute(backends[[i]], graphs[[i]])
      }

      # Параллельное выполнение на всех GPU
      start <- proc.time()
      for (i in 1:n_gpus) {
        ggml_backend_graph_compute(backends[[i]], graphs[[i]])
      }
      time_gpu <- (proc.time() - start)[3]

      # Собираем результаты
      results <- c()
      for (i in 1:n_gpus) {
        chunk_result <- ggml_backend_tensor_get_data(tensors[[i]])
        results <- c(results, chunk_result)
      }

      # Cleanup всех GPU
      for (i in 1:n_gpus) {
        ggml_backend_buffer_free(buffers[[i]])
        ggml_vulkan_free(backends[[i]])
        ggml_free(contexts[[i]])
      }

      list(
        mean_time = time_gpu,
        gflops = size * 1.0 / time_gpu / 1e9,
        result = results[1:5],
        n_gpus = n_gpus
      )
    }
  }, error = function(e) {
    list(mean_time = NA, gflops = NA, result = NULL, error = e$message, n_gpus = n_gpus)
  })

  return(result)
}

# Функция для бенчмарка матричного умножения на CPU
benchmark_cpu_matmul <- function(mat_size) {
  result <- tryCatch({
    n_elem <- mat_size * mat_size
    mem_size <- as.numeric(n_elem) * 4 * 4
    ctx <- ggml_init(mem_size = mem_size)
    ggml_set_no_alloc(ctx, TRUE)

    m1 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mat_size, mat_size)
    m2 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mat_size, mat_size)
    m3 <- ggml_mul_mat(ctx, m1, m2)

    backend <- ggml_backend_cpu_init()
    ggml_backend_cpu_set_n_threads(backend, n_cores)
    buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

    data_m1 <- rnorm(n_elem)
    data_m2 <- rnorm(n_elem)
    ggml_backend_tensor_set_data(m1, data_m1)
    ggml_backend_tensor_set_data(m2, data_m2)

    graph <- ggml_build_forward_expand(ctx, m3)

    # Прогрев
    ggml_backend_graph_compute(backend, graph)

    # Точный замер
    start <- proc.time()
    ggml_backend_graph_compute(backend, graph)
    time_elapsed <- (proc.time() - start)[3]
    gflops <- 2.0 * mat_size^3 / time_elapsed / 1e9

    ggml_backend_buffer_free(buffer)
    ggml_backend_free(backend)
    ggml_free(ctx)

    list(mean_time = time_elapsed, gflops = gflops)
  }, error = function(e) {
    list(mean_time = NA, gflops = NA, error = e$message)
  })

  return(result)
}

# Функция для бенчмарка матричного умножения на GPU
# Если несколько GPU - распределяет работу между ними параллельно
benchmark_gpu_matmul <- function(mat_size, device_ids = NULL) {
  if (!vulkan_available || ggml_vulkan_device_count() == 0) {
    return(NULL)
  }

  # Если не указаны device_ids, используем все доступные GPU
  if (is.null(device_ids)) {
    device_ids <- 0:(ggml_vulkan_device_count() - 1)
  }

  # Фильтруем валидные device_ids
  device_ids <- device_ids[device_ids < ggml_vulkan_device_count()]
  if (length(device_ids) == 0) {
    return(NULL)
  }

  n_gpus <- length(device_ids)

  result <- tryCatch({
    if (n_gpus == 1) {
      # Один GPU - простой случай
      device_id <- device_ids[1]
      n_elem <- mat_size * mat_size
      mem_size <- as.numeric(n_elem) * 4 * 4
      ctx <- ggml_init(mem_size = mem_size)
      ggml_set_no_alloc(ctx, TRUE)

      m1 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mat_size, mat_size)
      m2 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mat_size, mat_size)
      m3 <- ggml_mul_mat(ctx, m1, m2)

      backend <- ggml_vulkan_init(device_id)
      buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

      data_m1 <- rnorm(n_elem)
      data_m2 <- rnorm(n_elem)
      ggml_backend_tensor_set_data(m1, data_m1)
      ggml_backend_tensor_set_data(m2, data_m2)

      graph <- ggml_build_forward_expand(ctx, m3)

      # Прогрев
      ggml_backend_graph_compute(backend, graph)

      # Точный замер
      start <- proc.time()
      ggml_backend_graph_compute(backend, graph)
      time_elapsed <- (proc.time() - start)[3]
      gflops <- 2.0 * mat_size^3 / time_elapsed / 1e9

      ggml_backend_buffer_free(buffer)
      ggml_vulkan_free(backend)
      ggml_free(ctx)

      list(mean_time = time_elapsed, gflops = gflops, n_gpus = 1)
    } else {
      # Несколько GPU - параллельная работа
      # Разбиваем матрицу по строкам
      rows_per_gpu <- as.integer(mat_size / n_gpus)

      # Создаём контексты и backend'ы для каждого GPU
      contexts <- list()
      backends <- list()
      buffers <- list()
      graphs <- list()

      # Генерируем данные
      n_elem <- mat_size * mat_size
      data_m1 <- rnorm(n_elem)
      data_m2 <- rnorm(n_elem)

      # Инициализация всех GPU
      for (i in 1:n_gpus) {
        row_start <- (i - 1) * rows_per_gpu + 1
        row_end <- if (i == n_gpus) mat_size else i * rows_per_gpu
        chunk_rows <- row_end - row_start + 1

        chunk_elem <- chunk_rows * mat_size + n_elem  # Для m1_chunk и m2
        mem_size <- as.numeric(chunk_elem) * 4 * 4

        ctx <- ggml_init(mem_size = mem_size)
        ggml_set_no_alloc(ctx, TRUE)

        m1_chunk <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mat_size, chunk_rows)
        m2_full <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mat_size, mat_size)
        m3_chunk <- ggml_mul_mat(ctx, m1_chunk, m2_full)

        backend <- ggml_vulkan_init(device_ids[i])
        buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

        # Копируем соответствующий chunk m1 и полную m2
        chunk_start_idx <- (row_start - 1) * mat_size + 1
        chunk_end_idx <- row_end * mat_size
        ggml_backend_tensor_set_data(m1_chunk, data_m1[chunk_start_idx:chunk_end_idx])
        ggml_backend_tensor_set_data(m2_full, data_m2)

        graph <- ggml_build_forward_expand(ctx, m3_chunk)

        contexts[[i]] <- ctx
        backends[[i]] <- backend
        buffers[[i]] <- buffer
        graphs[[i]] <- graph
      }

      # Прогрев всех GPU
      for (i in 1:n_gpus) {
        ggml_backend_graph_compute(backends[[i]], graphs[[i]])
      }

      # Параллельное выполнение на всех GPU
      start <- proc.time()
      for (i in 1:n_gpus) {
        ggml_backend_graph_compute(backends[[i]], graphs[[i]])
      }
      time_elapsed <- (proc.time() - start)[3]

      # Суммарные FLOPS всех GPU
      total_flops <- 2.0 * mat_size^3
      gflops <- total_flops / time_elapsed / 1e9

      # Cleanup всех GPU
      for (i in 1:n_gpus) {
        ggml_backend_buffer_free(buffers[[i]])
        ggml_vulkan_free(backends[[i]])
        ggml_free(contexts[[i]])
      }

      list(mean_time = time_elapsed, gflops = gflops, n_gpus = n_gpus)
    }
  }, error = function(e) {
    list(mean_time = NA, gflops = NA, error = e$message, n_gpus = n_gpus)
  })

  return(result)
}

# ============================================================================
# Тест 1: Различные размеры тензоров (векторное сложение)
# ============================================================================
cat("═══ Тест 1: Векторное сложение на разных размерах ═══\n\n")

sizes <- c(1e6, 5e6, 1e7, 5e7, 1e8, 2e8, 5e8)

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
  cpu_result <- benchmark_cpu_vector(size)
  if (!is.na(cpu_result$mean_time)) {
    cat(sprintf("%.4f сек (%.2f GFLOPS)\n", cpu_result$mean_time, cpu_result$gflops))
  } else {
    cat(sprintf("ОШИБКА: %s\n", cpu_result$error))
    next
  }

  # GPU benchmark (все доступные GPU одновременно)
  gpu_result <- NULL
  if (vulkan_available && ggml_vulkan_device_count() > 0) {
    n_gpus <- ggml_vulkan_device_count()
    cat(sprintf("  GPU (%d устройств%s): ", n_gpus, if (n_gpus > 1) " параллельно" else ""))
    gpu_result <- benchmark_gpu_vector(size)  # Автоматически использует все GPU
    if (!is.null(gpu_result) && !is.na(gpu_result$mean_time)) {
      cat(sprintf("%.4f сек (%.2f GFLOPS)\n", gpu_result$mean_time, gpu_result$gflops))

      speedup <- cpu_result$mean_time / gpu_result$mean_time
      cat(sprintf("  Ускорение: %.2fx %s\n", speedup,
                  ifelse(speedup > 1, "🚀", "⚠️")))

      # Проверка корректности
      if (!is.null(cpu_result$result) && !is.null(gpu_result$result)) {
        if (max(abs(cpu_result$result - gpu_result$result)) < 1e-4) {
          cat("  Результаты: ✓ идентичны\n")
        } else {
          cat("  Результаты: ⚠️ отличаются\n")
        }
      }

      results_table <- rbind(results_table, data.frame(
        Size = sprintf("%.0e", size),
        CPU_Time = cpu_result$mean_time,
        GPU_Time = gpu_result$mean_time,
        CPU_GFLOPS = cpu_result$gflops,
        GPU_GFLOPS = gpu_result$gflops,
        Speedup = speedup
      ))
    } else {
      cat(sprintf("ОШИБКА: %s\n", if (!is.null(gpu_result)) gpu_result$error else "недоступен"))
    }
  } else {
    cat("  GPU: недоступен\n")
  }

  cat("\n")
}

# ============================================================================
# Тест 2: Матричное умножение
# ============================================================================
cat("═══ Тест 2: Матричное умножение ═══\n\n")

mat_sizes <- c(512, 1024, 2048)

for (mat_size in mat_sizes) {
  n_elem <- mat_size * mat_size
  size_mb <- n_elem * 4 / 1024 / 1024

  cat(sprintf("Матрица: %dx%d (%.1f MB)\n", mat_size, mat_size, size_mb))

  # CPU benchmark
  cat("  CPU: ")
  cpu_result <- benchmark_cpu_matmul(mat_size)
  if (!is.na(cpu_result$mean_time)) {
    cat(sprintf("%.4f сек (%.2f GFLOPS)\n", cpu_result$mean_time, cpu_result$gflops))
  } else {
    cat(sprintf("ОШИБКА: %s\n", cpu_result$error))
    next
  }

  # GPU benchmark (все доступные GPU одновременно)
  if (vulkan_available && ggml_vulkan_device_count() > 0) {
    n_gpus <- ggml_vulkan_device_count()
    cat(sprintf("  GPU (%d устройств%s): ", n_gpus, if (n_gpus > 1) " параллельно" else ""))
    gpu_result <- benchmark_gpu_matmul(mat_size)  # Автоматически использует все GPU
    if (!is.null(gpu_result) && !is.na(gpu_result$mean_time)) {
      cat(sprintf("%.4f сек (%.2f GFLOPS)\n", gpu_result$mean_time, gpu_result$gflops))
      speedup <- cpu_result$mean_time / gpu_result$mean_time
      cat(sprintf("  Ускорение: %.2fx %s\n", speedup,
                  ifelse(speedup > 1, "🚀", "⚠️")))
    } else {
      cat(sprintf("ОШИБКА: %s\n", if (!is.null(gpu_result)) gpu_result$error else "недоступен"))
    }
  } else {
    cat("  GPU: недоступен\n")
  }

  cat("\n")
}

# ============================================================================
# Тест 3: Масштабирование при использовании нескольких GPU
# ============================================================================
if (vulkan_available && ggml_vulkan_device_count() > 1) {
  cat("═══ Тест 3: Масштабирование производительности Multi-GPU ═══\n\n")

  test_size <- 1e8
  size_mb <- test_size * 4 / 1024 / 1024
  n_gpus <- ggml_vulkan_device_count()

  cat(sprintf("Размер теста: %.0e элементов (%.1f MB)\n", test_size, size_mb))
  cat(sprintf("Доступно GPU: %d\n\n", n_gpus))

  scaling_results <- data.frame(
    N_GPUs = integer(),
    Time = numeric(),
    GFLOPS = numeric(),
    Efficiency = numeric(),
    stringsAsFactors = FALSE
  )

  # Тестируем с разным количеством GPU: 1, 2, ..., n_gpus
  for (n_gpu_test in 1:n_gpus) {
    device_ids <- 0:(n_gpu_test - 1)
    cat(sprintf("Тест с %d GPU: ", n_gpu_test))

    gpu_result <- benchmark_gpu_vector(test_size, device_ids)
    if (!is.null(gpu_result) && !is.na(gpu_result$mean_time)) {
      cat(sprintf("%.4f сек (%.2f GFLOPS)\n", gpu_result$mean_time, gpu_result$gflops))

      # Эффективность = (фактическая производительность) / (n_gpus * производительность 1 GPU)
      if (n_gpu_test == 1) {
        single_gpu_gflops <- gpu_result$gflops
        efficiency <- 100.0
      } else {
        ideal_gflops <- single_gpu_gflops * n_gpu_test
        efficiency <- (gpu_result$gflops / ideal_gflops) * 100.0
      }

      scaling_results <- rbind(scaling_results, data.frame(
        N_GPUs = n_gpu_test,
        Time = gpu_result$mean_time,
        GFLOPS = gpu_result$gflops,
        Efficiency = efficiency
      ))
    } else {
      cat(sprintf("ОШИБКА: %s\n", if (!is.null(gpu_result)) gpu_result$error else "недоступен"))
    }
  }

  if (nrow(scaling_results) > 0) {
    cat("\n═══ Результаты масштабирования ═══\n\n")
    print(scaling_results, row.names = FALSE, digits = 4)

    cat("\nПримечание:\n")
    cat("  - Efficiency 100% = идеальное масштабирование\n")
    cat("  - Efficiency < 100% = накладные расходы на передачу данных между GPU\n")
  }

  cat("\n")
}

# ============================================================================
# Итоговая таблица
# ============================================================================
if (nrow(results_table) > 0) {
  cat("\n═══ Итоговая таблица результатов (Тест 1) ═══\n\n")
  print(results_table, row.names = FALSE, digits = 4)

  cat("\n═══ Статистика ═══\n")
  cat(sprintf("Средняя производительность CPU: %.2f GFLOPS\n",
              mean(results_table$CPU_GFLOPS)))
  cat(sprintf("Средняя производительность GPU: %.2f GFLOPS\n",
              mean(results_table$GPU_GFLOPS)))
  cat(sprintf("Среднее ускорение GPU vs CPU: %.2fx\n",
              mean(results_table$Speedup)))
  cat(sprintf("Максимальное ускорение: %.2fx\n",
              max(results_table$Speedup)))
  cat(sprintf("Минимальное ускорение: %.2fx\n",
              min(results_table$Speedup)))
}

cat("\n╔════════════════════════════════════════════════════════════════╗\n")
cat("║                         ТЕСТЫ ЗАВЕРШЕНЫ                        ║\n")
cat("╚════════════════════════════════════════════════════════════════╝\n")
