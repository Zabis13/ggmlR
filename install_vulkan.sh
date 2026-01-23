#!/bin/bash

# Скрипт для установки ggmlR с поддержкой Vulkan

echo "=== Удаление старой версии пакета ==="
R --vanilla -e 'if ("ggmlR" %in% rownames(installed.packages())) remove.packages("ggmlR")'

echo ""
echo "=== Установка с Vulkan поддержкой ==="
R CMD INSTALL . --configure-args="--with-vulkan"

echo ""
echo "=== Проверка установки ==="
R --vanilla -e 'library(ggmlR); ggml_vulkan_status()'
