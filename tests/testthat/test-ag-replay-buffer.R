# Replay buffer (R/ag_replay_buffer.R): packed layout, ring writes across the
# end, sampling from the filled part, fields of a batch, episode bookkeeping,
# prioritized sampling, save/load (own and rltoolsR files). CPU and GPU.

rb_devices <- function() c("cpu", if (ggml_vulkan_available() && ggml_vulkan_device_count() > 0L) "gpu")

# Step `t` of 3 environments with obs_dim 2, act_dim 1; values encode (t, env).
rb_step <- function(t, n = 3L) {
  e <- seq_len(n)
  list(obs = cbind(t * 10 + e, -(t * 10 + e)), act = matrix(t + e / 10, n),
       reward = t * 100 + e, terminated = e == 1L & t %% 2L == 0L,
       truncated = logical(n), next_obs = cbind(t * 10 + e + 0.5, 0))
}
rb_column <- function(s, j) c(s$obs[j, ], s$act[j, ], s$reward[j], s$terminated[j],
                              s$truncated[j], s$next_obs[j, ])
add_step <- function(rb, s) rb$add(s$obs, s$act, s$reward, s$terminated, s$truncated,
                                   s$next_obs)

test_that("the layout: obs and act adjacent, same as rltoolsR", {
  l <- ggmlR:::.ag_rb_layout(3L, 2L)
  expect_equal(l$rows, 3 + 2 + 3 + 3)
  expect_equal(unname(l$from), c(0, 3, 5, 6, 7, 8))
})

test_that("a ring with 3 environments and capacity 7 wraps across the end", {
  on.exit(ag_device("cpu"), add = TRUE)
  for (dev in rb_devices()) {
    ag_device(dev)
    rb <- ag_replay_buffer(2L, 1L, capacity = 7L, n_envs = 3L)
    steps <- lapply(1:3, rb_step)
    for (s in steps) add_step(rb, s)
    expect_equal(rb$size, 7L, info = dev)
    expect_equal(rb$pos, 2L, info = dev)
    want <- cbind(rb_column(steps[[3]], 2), rb_column(steps[[3]], 3),
                  rb_column(steps[[1]], 3), rb_column(steps[[2]], 1),
                  rb_column(steps[[2]], 2), rb_column(steps[[2]], 3),
                  rb_column(steps[[3]], 1))
    expect_equal(ag_buffer_read(rb$buf), want, tolerance = 1e-6, ignore_attr = TRUE,
                 info = dev)
    expect_true(rb$full())
    rb$free()
  }
})

test_that("uniform sampling: filled part only, weights 1, same draws as rltoolsR", {
  ag_device("cpu")
  rb <- ag_replay_buffer(2L, 1L, capacity = 100L, n_envs = 3L)
  expect_error(rb$sample(4L), "empty")
  add_step(rb, rb_step(1)); add_step(rb, rb_step(2))
  set.seed(1)
  b <- rb$sample(1000L)
  expect_true(all(b$idx >= 0L & b$idx < 6L))
  expect_setequal(unique(b$idx), 0:5)
  expect_equal(b$weights, rep(1, 1000L))
  set.seed(1)
  expect_identical(b$idx, sample.int(6L, 1000L, replace = TRUE) - 1L)
  expect_error(rb$update_priorities(0L, 1), "not prioritized")
  rb$free()
})

test_that("a gathered batch splits into its fields", {
  on.exit(ag_device("cpu"), add = TRUE)
  for (dev in rb_devices()) {
    ag_device(dev)
    rb <- ag_replay_buffer(2L, 1L, capacity = 10L, n_envs = 3L)
    s1 <- rb_step(1); s2 <- rb_step(2)
    add_step(rb, s1); add_step(rb, s2)
    x <- rb$gather(c(4L, 0L, 4L))
    f <- function(field) as.matrix(rb$field(x, field))
    expect_equal(f("obs"), cbind(s2$obs[2, ], s1$obs[1, ], s2$obs[2, ]), info = dev)
    expect_equal(f("reward"), matrix(c(s2$reward[2], s1$reward[1], s2$reward[2]), 1),
                 info = dev)
    expect_equal(f("next_obs"), cbind(s2$next_obs[2, ], s1$next_obs[1, ], s2$next_obs[2, ]),
                 info = dev)
    expect_equal(as.matrix(rb$slice(x, c("obs", "act"))), rbind(f("obs"), f("act")),
                 info = dev)
    expect_error(rb$slice(x, c("obs", "reward")), "not adjacent")
    rb$free()
  }
})

test_that("episode_start records where each copy's episode began", {
  ag_device("cpu")
  rb <- ag_replay_buffer(2L, 1L, capacity = 20L, n_envs = 3L)
  for (t in 1:3) add_step(rb, rb_step(t))
  expect_equal(rb$episode_start[1:9], c(0L, 1L, 2L, 0L, 1L, 2L, 6L, 1L, 2L))
  rb$free()
})

test_that("prioritized: added slots get max priority, updates steer the draws", {
  on.exit(ag_device("cpu"), add = TRUE)
  for (dev in rb_devices()) {
    ag_device(dev)
    rb <- ag_replay_buffer(2L, 1L, capacity = 7L, n_envs = 3L, prioritized = TRUE,
                           alpha = 1)
    add_step(rb, rb_step(1))
    expect_equal(rb$sampler$priorities()[1:3], c(1, 1, 1), info = dev)
    rb$update_priorities(0:2, c(0, 0, 100))
    add_step(rb, rb_step(2))                     # slots 3..5 get the max, 100 + eps
    expect_equal(rb$sampler$priorities()[4:6], rep(100 + 1e-6, 3), info = dev)
    rb$update_priorities(3:5, c(0, 0, 0))
    set.seed(8)
    b <- rb$sample(200L, beta = 1)
    expect_gt(mean(b$idx == 2L), 0.95)
    expect_true(all(b$weights <= 1 + 1e-12))
    # gather works on the drawn slots
    expect_equal(ncol(as.matrix(rb$gather(b$idx))), 200L, info = dev)
    rb$free()
  }
})

test_that("save and load restore the buffer and its priorities", {
  on.exit(ag_device("cpu"), add = TRUE)
  f <- tempfile(fileext = ".rb")
  on.exit(unlink(f), add = TRUE)
  for (dev in rb_devices()) {
    for (pri in c(FALSE, TRUE)) {
      ag_device(dev)
      rb <- ag_replay_buffer(2L, 1L, capacity = 7L, n_envs = 3L, prioritized = pri)
      for (t in 1:3) add_step(rb, rb_step(t))
      if (pri) rb$update_priorities(0:6, (1:7) / 10)
      rb$save(f, chunk = 3L)
      rb2 <- ag_replay_buffer_load(f, chunk = 2L)
      info <- paste(dev, pri)
      expect_equal(ag_buffer_read(rb2$buf), ag_buffer_read(rb$buf), tolerance = 1e-6,
                   info = info)
      for (nm in c("pos", "size", "episode_start", "cur_start", "capacity", "n_envs"))
        expect_identical(rb2[[nm]], rb[[nm]], info = paste(info, nm))
      expect_identical(is.null(rb2$sampler), !pri, info = info)
      if (pri) {
        expect_equal(rb2$sampler$priorities(), rb$sampler$priorities(), info = info)
        set.seed(9); a <- rb$sample(16L)
        set.seed(9); expect_identical(rb2$sample(16L), a, info = info)
      }
      rb$free(); rb2$free()
    }
  }
  expect_false(any(grepl("\\.tmp$", list.files(dirname(f)))))   # atomic write cleaned up
  writeLines("x", f)
  expect_error(ag_replay_buffer_load(f), "not a replay buffer")
})

test_that("files of rltoolsR's replay buffer load", {
  ag_device("cpu")
  f <- tempfile(fileext = ".rb")
  on.exit(unlink(f), add = TRUE)
  rb <- ag_replay_buffer(2L, 1L, capacity = 7L, n_envs = 3L)
  for (t in 1:2) add_step(rb, rb_step(t))
  # the rltoolsR v1 layout: its header, then the filled slots as f32
  h <- list(format = "rltoolsR.replay_buffer.v1", obs_dim = 2L, act_dim = 1L,
            capacity = 7L, n_envs = 3L, rows = rb$layout$rows, pos = rb$pos,
            size = rb$size, episode_start = rb$episode_start, cur_start = rb$cur_start)
  con <- file(f, "wb"); serialize(h, con)
  writeBin(as.numeric(ag_buffer_read(rb$buf, 0, rb$size)), con, size = 4L); close(con)
  rb2 <- ag_replay_buffer_load(f)
  expect_null(rb2$sampler)
  expect_equal(ag_buffer_read(rb2$buf, 0, 6), ag_buffer_read(rb$buf, 0, 6), tolerance = 1e-6)
  expect_identical(rb2$episode_start, rb$episode_start)
  rb$free(); rb2$free()
})

test_that("shapes and capacity are checked", {
  ag_device("cpu")
  expect_error(ag_replay_buffer(2L, 1L, capacity = 2L, n_envs = 3L), "capacity")
  rb <- ag_replay_buffer(2L, 1L, capacity = 10L, n_envs = 3L)
  s <- rb_step(1)
  expect_error(rb$add(s$obs, s$act, s$reward[1:2], s$terminated, s$truncated, s$next_obs),
               "shapes")
  rb$free()
})
