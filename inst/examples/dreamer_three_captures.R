# DreamerV3-style training as THREE ag_capture_step recordings, one optimizer
# each: world model, actor, critic. The other modules' weights are frozen
# inputs of a recording (passed in `params`, not in the optimizer).
# Checks: (1) every step records, (2) captured losses == uncaptured losses,
# (3) frozen weights are not changed by another module's step, (4) timing.
suppressMessages(library(ggmlR))
ag_device("gpu")
ag_local_mode(graph = TRUE, matmul_precision = "f32")

T  <- 16L; H <- 15L; B <- 8L
OBS <- 16L; NA_ <- 4L; DET <- 64L
NC <- 8L; NK <- 8L; Z <- NC * NK
GAMMA <- 0.997; LAMBDA <- 0.95; ETA <- 3e-4
N <- 5L

make_model <- function() {
  set.seed(1)
  list(enc   = ag_linear(OBS, 32L, activation = "relu"),
       inp   = ag_linear(Z + NA_, DET, activation = "relu"),
       gru   = ag_gru_cell(DET, DET),
       post  = ag_linear(DET + 32L, Z),
       prior = ag_linear(DET, Z),
       dec   = ag_linear(DET + Z, OBS),
       rew   = ag_linear(DET + Z, 1L),
       cont  = ag_linear(DET + Z, 1L),
       actor = ag_linear(DET + Z, NA_),
       critic = ag_linear(DET + Z, 1L))
}
WM_PARTS <- c("enc", "inp", "gru", "post", "prior", "dec", "rew", "cont")
params_of <- function(m, parts)
  unlist(lapply(m[parts], function(l) l$parameters()), use.names = FALSE)

eye_z <- ag_buffer(NK, NK);   ag_buffer_write(eye_z, diag(NK), 0L)
eye_a <- ag_buffer(NA_, NA_); ag_buffer_write(eye_a, diag(NA_), 0L)

unimix <- function(logits) ag_add(ag_scale(ag_softmax(logits), 0.99), 0.01 / NK)
sample_z <- function(logits_z, key, stream) {      # [Z, B] logits -> one-hot sample
  q   <- unimix(ag_reshape(logits_z, NK, NC * B))
  idx <- ag_sample_categorical(ag_log(q), key = key, stream = stream)
  oh  <- ag_get_rows(eye_z, idx)
  list(q = q, z = ag_reshape(ag_add(oh, ag_sub(q, ag_detach(q))), Z, B))
}

# ---- world model: obs [OBS*T, B], act [NA*T, B] one-hot, rew/cont [T, B] ----
# z0 [Z, B] zeros: the latent before the first step (an input, not an upload)
wm_fn <- function(m) function(obs, act, rew, cont, key, z0) {
  s <- NULL; z <- z0; loss <- NULL
  for (t in seq_len(T)) {
    o <- ag_slice_rows(obs, (t - 1L) * OBS, OBS)
    a <- ag_slice_rows(act, (t - 1L) * NA_, NA_)
    s <- m$gru$forward(m$inp$forward(ag_concat_rows(z, a)), s); h <- s$h
    zs <- sample_z(m$post$forward(ag_concat_rows(h, m$enc$forward(o))), key, t)
    z  <- zs$z
    lp <- ag_reshape(m$prior$forward(h), NK, NC * B)
    p  <- unimix(lp)
    kl <- ag_sum(ag_mul(ag_detach(zs$q), ag_sub(ag_log(ag_detach(zs$q)), ag_log(p))))
    kl <- ag_clamp(ag_scale(kl, 1 / (NC * B)), lo = 1)
    f  <- ag_concat_rows(h, z)
    l  <- ag_add(ag_add(ag_mse_loss(m$dec$forward(f), o), kl),
                 ag_add(ag_mse_loss(m$rew$forward(f), ag_slice_rows(rew, t - 1L, 1L)),
                        ag_mse_loss(ag_sigmoid(m$cont$forward(f)), ag_slice_rows(cont, t - 1L, 1L))))
    loss <- if (is.null(loss)) l else ag_add(loss, l)
  }
  list(loss = ag_scale(loss, 1 / T), h = h, z = z)
}

# ---- imagination from (h0, z0): H steps of the actor in the frozen prior -----
imagine <- function(m, h0, z0, key) {
  s <- list(h = h0); z <- z0
  feats <- list(); logp_a <- list(); ent <- list(); r <- list(); cn <- list()
  for (t in seq_len(H)) {
    f  <- ag_concat_rows(s$h, z); feats[[t]] <- f
    lg <- m$actor$forward(ag_detach(f))
    lp <- ag_log_softmax(lg)
    idx <- ag_sample_categorical(lg, key = key, stream = 100L + t)
    oh  <- ag_get_rows(eye_a, idx)
    logp_a[[t]] <- ag_sum(ag_mul(lp, oh), dim = 2L)                     # [1, B]
    ent[[t]]    <- ag_scale(ag_sum(ag_mul(ag_softmax(lg), lp), dim = 2L), -1)
    s <- m$gru$forward(m$inp$forward(ag_concat_rows(z, oh)), s)
    z <- sample_z(m$prior$forward(s$h), key, 200L + t)$z
    fn <- ag_detach(ag_concat_rows(s$h, z))
    r[[t]]  <- m$rew$forward(fn)
    cn[[t]] <- ag_sigmoid(m$cont$forward(fn))
  }
  feats[[H + 1L]] <- ag_concat_rows(s$h, z)
  list(feats = feats, logp_a = logp_a, ent = ent, r = r, cn = cn)
}

# lambda-returns R_1..R_H from detached values v_1..v_{H+1}
lambda_returns <- function(im, v) {
  R <- vector("list", H); nxt <- v[[H + 1L]]
  for (t in rev(seq_len(H))) {
    mix <- ag_add(ag_scale(v[[t + 1L]], 1 - LAMBDA), ag_scale(nxt, LAMBDA))
    R[[t]] <- ag_add(im$r[[t]], ag_scale(ag_mul(im$cn[[t]], mix), GAMMA))
    nxt <- R[[t]]
  }
  R
}

actor_fn <- function(m) function(h0, z0, key) {
  im <- imagine(m, h0, z0, key)
  v  <- lapply(im$feats, function(f) ag_detach(m$critic$forward(ag_detach(f))))
  R  <- lambda_returns(im, v)
  loss <- NULL
  for (t in seq_len(H)) {
    adv <- ag_detach(ag_sub(R[[t]], v[[t]]))
    l <- ag_sub(ag_scale(ag_mean(ag_mul(adv, im$logp_a[[t]])), -1),
                ag_scale(ag_mean(im$ent[[t]]), ETA))
    loss <- if (is.null(loss)) l else ag_add(loss, l)
  }
  ag_scale(loss, 1 / H)
}

critic_fn <- function(m) function(h0, z0, key) {
  im <- imagine(m, h0, z0, key)
  vd <- lapply(im$feats, function(f) ag_detach(m$critic$forward(ag_detach(f))))
  R  <- lambda_returns(im, vd)
  loss <- NULL
  for (t in seq_len(H)) {
    d <- ag_sub(m$critic$forward(ag_detach(im$feats[[t]])), ag_detach(R[[t]]))
    l <- ag_mean(ag_mul(d, d))
    loss <- if (is.null(loss)) l else ag_add(loss, l)
  }
  ag_scale(loss, 1 / H)
}

# ---- data --------------------------------------------------------------------
set.seed(2)
obs  <- matrix(rnorm(OBS * T * B), OBS * T, B)
act  <- matrix(0, NA_ * T, B)
for (t in seq_len(T)) for (b in seq_len(B)) act[(t - 1L) * NA_ + sample.int(NA_, 1L), b] <- 1
rew  <- matrix(rnorm(T * B), T, B)
cont <- matrix(as.numeric(runif(T * B) > 0.05), T, B)
z0   <- matrix(0, Z, B)
key  <- function(i) matrix(c(123, i), 2, 1)

snap <- function(ps) lapply(ps, as.matrix)
same <- function(a, b) all(mapply(identical, a, b))

run <- function(captured) {
  m <- make_model()
  p_wm <- params_of(m, WM_PARTS); p_a <- params_of(m, "actor"); p_c <- params_of(m, "critic")
  o_wm <- optimizer_adam(p_wm, lr = 1e-3)
  o_a  <- optimizer_adam(p_a, lr = 1e-3)
  o_c  <- optimizer_adam(p_c, lr = 1e-3)
  frozen_ok <- TRUE
  if (captured) {
    s_wm <- ag_capture_step(wm_fn(m), params = list(eye_z), optimizer = o_wm)
    s_a  <- ag_capture_step(actor_fn(m), params = c(p_wm, p_c, list(eye_z, eye_a)), optimizer = o_a)
    s_c  <- ag_capture_step(critic_fn(m), params = c(p_wm, p_a, list(eye_z, eye_a)), optimizer = o_c)
    on.exit({ ag_capture_free(s_wm); ag_capture_free(s_a); ag_capture_free(s_c) }, add = TRUE)
    step <- function(which, ...) switch(which, wm = s_wm(...), a = s_a(...), c = s_c(...))
  } else {
    one <- function(fn, opt, args) {
      out <- NULL
      with_grad_tape(out <- do.call(fn, lapply(args, ag_tensor)))
      loss <- if (is.list(out)) out$loss else out
      opt$step(backward(loss)); opt$zero_grad()
      res <- list(loss = as.matrix(loss))
      if (is.list(out)) { res$h <- as.matrix(out$h); res$z <- as.matrix(out$z) }
      res
    }
    step <- function(which, ...) switch(which,
      wm = one(wm_fn(m), o_wm, list(...)), a = one(actor_fn(m), o_a, list(...)),
      c = one(critic_fn(m), o_c, list(...)))
  }
  losses <- matrix(NA_real_, N, 3L, dimnames = list(NULL, c("wm", "actor", "critic")))
  ms <- matrix(NA_real_, N, 3L, dimnames = dimnames(losses))
  for (i in seq_len(N)) {
    t0 <- Sys.time()
    w <- step("wm", obs = obs, act = act, rew = rew, cont = cont, key = key(i), z0 = z0)
    t1 <- Sys.time()
    before <- snap(c(p_wm, p_c))
    a <- step("a", h0 = w$h, z0 = w$z, key = key(1000 + i))
    frozen_ok <- frozen_ok && same(before, snap(c(p_wm, p_c)))
    t2 <- Sys.time()
    before <- snap(c(p_wm, p_a))
    c_ <- step("c", h0 = w$h, z0 = w$z, key = key(2000 + i))
    frozen_ok <- frozen_ok && same(before, snap(c(p_wm, p_a)))
    t3 <- Sys.time()
    losses[i, ] <- c(as.numeric(w$loss), as.numeric(if (is.list(a)) a$loss else a),
                     as.numeric(if (is.list(c_)) c_$loss else c_))
    ms[i, ] <- 1000 * as.numeric(c(t1 - t0, t2 - t1, t3 - t2), units = "secs")
  }
  list(losses = losses, ms = ms, frozen_ok = frozen_ok)
}

ref <- tryCatch(run(FALSE), error = function(e) { cat("eager ERROR:", conditionMessage(e), "\n"); NULL })
got <- tryCatch(run(TRUE),  error = function(e) { cat("captured ERROR:", conditionMessage(e), "\n"); NULL })
if (!is.null(ref) && !is.null(got)) {
  cat(sprintf("T = %d, H = %d, B = %d\n", T, H, B))
  cat("eager losses:\n");    print(round(ref$losses, 5))
  cat("captured losses:\n"); print(round(got$losses, 5))
  cat(sprintf("max |diff| per module: %s\n",
              paste(sprintf("%s %.2e", colnames(ref$losses),
                            apply(abs(ref$losses - got$losses), 2, max)), collapse = ", ")))
  cat(sprintf("frozen weights unchanged: eager %s, captured %s\n", ref$frozen_ok, got$frozen_ok))
  cat("ms per step (mean of calls 2..N; call 1 records):\n")
  print(round(rbind(eager = colMeans(ref$ms[-1, , drop = FALSE]),
                    captured = colMeans(got$ms[-1, , drop = FALSE]),
                    first_call = got$ms[1, ]), 1))
}
ag_buffer_free(eye_z); ag_buffer_free(eye_a)
cat("done\n")
