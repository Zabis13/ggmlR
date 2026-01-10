
install.packages(c("devtools", "usethis", "Rcpp"))
library(usethis)

# Создание пакета
create_package("~/ggmlR")

# Настройка Rcpp
use_rcpp()

# Добавление лицензии MIT (как у ggml)
use_mit_license("Your Name")

# Создание README
use_readme_md()

# Git репозиторий
use_git()
