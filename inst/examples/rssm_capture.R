# Mini RSSM (DreamerV3-like world model) under ag_capture / ag_capture_step.
# Checks: (1) the loop records at all, (2) captured forward == eager forward,
# (3) captured train step == eager train step (losses), (4) timing, node count.
suppressMessages(library(ggmlR))
ag_device("gpu")
ag_local_mode(graph = TRUE, matmul_precision = "f32")

T  <- as.integer(Sys.getenv("RSSM_T", "16"))
B  <- 8L; OBS <- 16L; ACT <- 4L; DET <- 64L
NC <- 8L; NK <- 8L; Z <- NC * NK                 # 8 categoricals x 8 classes

make_model <- function() {
  set.seed(1)
  list(enc   = ag_linear(OBS, 32L, activation = "relu"),
       inp   = ag_linear(Z + ACT, DET, activation = "relu"),
       gru   = ag_gru_cell(DET, DET),
       post  = ag_linear(DET + 32L, Z),
       prior = ag_linear(DET, Z),
       dec   = ag_linear(DET + Z, OBS))
}

eye <- ag_buffer(NK, NK); ag_buffer_write(eye, diag(NK), 0L)

# obs [OBS*T, B], act [ACT*T, B], key [2, 1], z0 [Z, B] zeros.
# Constants come in as arguments: an ag_tensor() made inside fn would be an
# upload during the recording (refused). Plain numbers are fine (ag_add with a
# number is one scale_bias node), and the GRU makes its zero state on the device.
make_fn <- function(m) function(obs, act, key, z0) {
  s <- NULL                          # GRU zero state is made on the device
  z <- z0
  loss <- NULL
  for (t in seq_len(T)) {
    o <- ag_slice_rows(obs, (t - 1L) * OBS, OBS)
    a <- ag_slice_rows(act, (t - 1L) * ACT, ACT)
    s <- m$gru$forward(m$inp$forward(ag_concat_rows(z, a)), s); h <- s$h
    e <- m$enc$forward(o)
    lq <- ag_reshape(m$post$forward(ag_concat_rows(h, e)), NK, NC * B)
    lp <- ag_reshape(m$prior$forward(h), NK, NC * B)
    q  <- ag_add(ag_scale(ag_softmax(lq), 0.99), 0.01 / NK)      # unimix
    p  <- ag_add(ag_scale(ag_softmax(lp), 0.99), 0.01 / NK)
    idx <- ag_sample_categorical(ag_log(q), key = key, stream = t)
    oh  <- ag_get_rows(eye, idx)                                  # [NK, NC*B]
    zs  <- ag_add(oh, ag_sub(q, ag_detach(q)))                    # straight-through
    z   <- ag_reshape(zs, Z, B)
    kl  <- ag_sum(ag_mul(ag_detach(q), ag_sub(ag_log(ag_detach(q)), ag_log(p))))
    kl  <- ag_clamp(ag_scale(kl, 1 / (NC * B)), lo = 1)           # free bits
    rec <- ag_mse_loss(m$dec$forward(ag_concat_rows(h, z)), o)
    step <- ag_add(rec, kl)
    loss <- if (is.null(loss)) step else ag_add(loss, step)
  }
  ag_scale(loss, 1 / T)
}

set.seed(2)
obs <- matrix(rnorm(OBS * T * B), OBS * T, B)
act <- matrix(rnorm(ACT * T * B), ACT * T, B)
key <- matrix(c(123, 456), 2, 1)
z0  <- matrix(0, Z, B)
N <- 10L

cat(sprintf("T = %d, B = %d\n", T, B))

## 1-2. forward: eager vs captured
m <- make_model(); fn <- make_fn(m)
ps <- unlist(lapply(m, function(l) l$parameters()), use.names = FALSE)
t0 <- Sys.time()
eag <- as.numeric(as.matrix(fn(ag_tensor(obs), ag_tensor(act), ag_tensor(key), ag_tensor(z0))))
cat(sprintf("eager forward:      %.4f  (%.1f ms)\n", eag,
            1000 * as.numeric(Sys.time() - t0, units = "secs")))
cf <- tryCatch(ag_capture(fn, params = c(ps, list(eye))),
               error = function(e) { cat("ag_capture ERROR:", conditionMessage(e), "\n"); NULL })
if (!is.null(cf)) {
  r <- tryCatch(cf(obs = obs, act = act, key = key, z0 = z0),
                error = function(e) { cat("captured fwd ERROR:", conditionMessage(e), "\n"); NULL })
  if (!is.null(r)) {
    t0 <- Sys.time(); for (i in seq_len(N)) r <- cf(obs = obs, act = act, key = key, z0 = z0)
    cat(sprintf("captured forward:   %.4f  (%.2f ms/call), |diff| %.2e, nodes %s\n",
                as.numeric(r), 1000 * as.numeric(Sys.time() - t0, units = "secs") / N,
                abs(as.numeric(r) - eag), format(ggmlR:::.ag_launch$last_nodes)))
  }
  ag_capture_free(cf)
}

## 3-4. train step: eager vs captured, same init
eager_losses <- numeric(0)
m <- make_model(); fn <- make_fn(m)
ps <- unlist(lapply(m, function(l) l$parameters()), use.names = FALSE); opt <- optimizer_adam(ps, lr = 1e-3)
X <- ag_tensor(obs); A <- ag_tensor(act); K <- ag_tensor(key)
Z0 <- ag_tensor(z0)
t0 <- Sys.time()
for (i in seq_len(N)) {
  loss <- NULL
  with_grad_tape(loss <- fn(X, A, K, Z0))
  opt$step(backward(loss)); opt$zero_grad()
  eager_losses <- c(eager_losses, as.numeric(as.matrix(loss)))
}
te <- 1000 * as.numeric(Sys.time() - t0, units = "secs") / N
cat(sprintf("eager step:    %.1f ms/step\n", te))

m <- make_model(); fn <- make_fn(m)
ps <- unlist(lapply(m, function(l) l$parameters()), use.names = FALSE); opt <- optimizer_adam(ps, lr = 1e-3)
tr <- tryCatch(ag_capture_step(fn, params = list(eye), optimizer = opt),
               error = function(e) { cat("ag_capture_step ERROR:", conditionMessage(e), "\n"); NULL })
if (!is.null(tr)) {
  cap_losses <- numeric(0)
  t0 <- Sys.time(); t1 <- NULL
  for (i in seq_len(N)) {
    out <- tryCatch(tr(obs = obs, act = act, key = key, z0 = z0),
                    error = function(e) { cat("captured step ERROR:", conditionMessage(e), "\n"); NULL })
    if (is.null(out)) break
    if (i == 1L) t1 <- Sys.time()
    cap_losses <- c(cap_losses, as.numeric(if (is.list(out)) out$loss else out))
  }
  if (length(cap_losses) == N) {
    tc <- 1000 * as.numeric(Sys.time() - t1, units = "secs") / (N - 1L)
    cat(sprintf("captured step: %.1f ms/step (first call incl. recording: %.0f ms)\n",
                tc, 1000 * as.numeric(t1 - t0, units = "secs")))
    cat("eager losses:   ", sprintf("%.4f", eager_losses), "\n")
    cat("captured losses:", sprintf("%.4f", cap_losses), "\n")
    cat(sprintf("max |diff| losses: %.2e, speedup %.1fx\n",
                max(abs(eager_losses - cap_losses)), te / tc))
  }
  ag_capture_free(tr)
}
ag_buffer_free(eye)
cat("done\n")
