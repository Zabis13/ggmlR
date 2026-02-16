FROM rocker/r-ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Базовые пакеты + рантайм Vulkan
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake pkg-config \
        wget curl ca-certificates \
        mesa-vulkan-drivers vulkan-tools libvulkan1 \
        && rm -rf /var/lib/apt/lists/*

# Vulkan SDK: libvulkan-dev + glslc и т.п.
RUN wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc \
        | tee /etc/apt/trusted.gpg.d/lunarg.asc && \
    wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list \
        http://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        vulkan-sdk \
        && rm -rf /var/lib/apt/lists/*

# Установка ggmlR из GitHub (с Vulkan backend, как у тебя в репо) [web:106][web:110]
RUN R -q -e "install.packages('remotes', repos = 'https://cloud.r-project.org')" && \
    R -q -e "remotes::install_github('Zabis13/ggmlR')"

WORKDIR /work

# Копируем пример из inst/examples в образ
# Важно: путь относительно корня репо, куда ты положишь Dockerfile
COPY inst/examples/mnist_cnn.R /work/mnist_cnn.R

CMD ["Rscript", "mnist_cnn.R"]

