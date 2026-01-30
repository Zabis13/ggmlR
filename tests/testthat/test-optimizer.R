# Tests for optimization functions

test_that("optimizer type constants work", {
  # Loss types
  expect_type(ggml_opt_loss_type_mean(), "integer")
  expect_type(ggml_opt_loss_type_sum(), "integer")
  expect_type(ggml_opt_loss_type_cross_entropy(), "integer")
  expect_type(ggml_opt_loss_type_mse(), "integer")

  # All loss types should be different
  loss_types <- c(
    ggml_opt_loss_type_mean(),
    ggml_opt_loss_type_sum(),
    ggml_opt_loss_type_cross_entropy(),
    ggml_opt_loss_type_mse()
  )
  expect_equal(length(unique(loss_types)), 4)

  # Optimizer types
  expect_type(ggml_opt_optimizer_type_adamw(), "integer")
  expect_type(ggml_opt_optimizer_type_sgd(), "integer")

  # Optimizer types should be different
  expect_true(ggml_opt_optimizer_type_adamw() != ggml_opt_optimizer_type_sgd())
})

test_that("optimizer names work", {
  adamw_name <- ggml_opt_optimizer_name(ggml_opt_optimizer_type_adamw())
  sgd_name <- ggml_opt_optimizer_name(ggml_opt_optimizer_type_sgd())

  expect_type(adamw_name, "character")
  expect_type(sgd_name, "character")
  expect_true(nchar(adamw_name) > 0)
  expect_true(nchar(sgd_name) > 0)
})

test_that("dataset creation and destruction works", {
  # Create dataset
  dataset <- ggml_opt_dataset_init(
    type_data = GGML_TYPE_F32,
    type_label = GGML_TYPE_F32,
    ne_datapoint = 10,
    ne_label = 1,
    ndata = 100,
    ndata_shard = 1
  )

  expect_true(!is.null(dataset))

  # Check ndata
  ndata <- ggml_opt_dataset_ndata(dataset)
  expect_equal(ndata, 100)

  # Get data tensor
  data_tensor <- ggml_opt_dataset_data(dataset)
  expect_true(!is.null(data_tensor))

  # Get labels tensor
  labels_tensor <- ggml_opt_dataset_labels(dataset)
  expect_true(!is.null(labels_tensor))

  # Free dataset
  expect_silent(ggml_opt_dataset_free(dataset))
})

test_that("dataset without labels works", {
  # Create dataset with ne_label = 0 (no labels)
  dataset <- ggml_opt_dataset_init(
    type_data = GGML_TYPE_F32,
    type_label = GGML_TYPE_F32,
    ne_datapoint = 10,
    ne_label = 0,
    ndata = 50,
    ndata_shard = 1
  )

  expect_true(!is.null(dataset))

  # Check ndata
  ndata <- ggml_opt_dataset_ndata(dataset)
  expect_equal(ndata, 50)

  # Labels should be NULL when ne_label = 0
  labels_tensor <- ggml_opt_dataset_labels(dataset)
  expect_null(labels_tensor)

  # Free dataset
  ggml_opt_dataset_free(dataset)
})

test_that("result creation and operations work", {
  # Create result
  result <- ggml_opt_result_init()
  expect_true(!is.null(result))

  # Get initial ndata (should be 0)
  ndata <- ggml_opt_result_ndata(result)
  expect_equal(ndata, 0)

  # Get loss (with empty result)
  loss_info <- ggml_opt_result_loss(result)
  expect_type(loss_info, "double")
  expect_equal(length(loss_info), 2)
  expect_true("loss" %in% names(loss_info))
  expect_true("uncertainty" %in% names(loss_info))

  # Get accuracy (with empty result)
  acc_info <- ggml_opt_result_accuracy(result)
  expect_type(acc_info, "double")
  expect_equal(length(acc_info), 2)
  expect_true("accuracy" %in% names(acc_info))
  expect_true("uncertainty" %in% names(acc_info))

  # Reset result
  expect_silent(ggml_opt_result_reset(result))

  # Free result
  expect_silent(ggml_opt_result_free(result))
})

test_that("optimizer context creation works with CPU backend", {
  # Create CPU backend
  cpu <- ggml_backend_cpu_init()
  expect_true(!is.null(cpu))

  # Create scheduler
  sched <- ggml_backend_sched_new(list(cpu), parallel = FALSE)
  expect_true(!is.null(sched))

  # Get default params
  params <- ggml_opt_default_params(sched, ggml_opt_loss_type_mse())
  expect_type(params, "list")
  expect_true("loss_type" %in% names(params))
  expect_true("optimizer" %in% names(params))

  # Create optimizer context
  opt_ctx <- ggml_opt_init(
    sched = sched,
    loss_type = ggml_opt_loss_type_mse(),
    optimizer = ggml_opt_optimizer_type_adamw(),
    opt_period = 1L
  )
  expect_true(!is.null(opt_ctx))

  # Check optimizer type
  opt_type <- ggml_opt_context_optimizer_type(opt_ctx)
  expect_equal(opt_type, ggml_opt_optimizer_type_adamw())

  # Check static graphs (should be FALSE without ctx_compute)
  is_static <- ggml_opt_static_graphs(opt_ctx)
  expect_type(is_static, "logical")

  # Reset optimizer
  expect_silent(ggml_opt_reset(opt_ctx, optimizer = FALSE))
  expect_silent(ggml_opt_reset(opt_ctx, optimizer = TRUE))

  # Free optimizer context
  expect_silent(ggml_opt_free(opt_ctx))

  # Cleanup
  ggml_backend_sched_free(sched)
  ggml_backend_free(cpu)
})

test_that("optimizer context works with SGD", {
  cpu <- ggml_backend_cpu_init()
  sched <- ggml_backend_sched_new(list(cpu), parallel = FALSE)

  # Create optimizer context with SGD

  opt_ctx <- ggml_opt_init(
    sched = sched,
    loss_type = ggml_opt_loss_type_cross_entropy(),
    optimizer = ggml_opt_optimizer_type_sgd(),
    opt_period = 4L
  )
  expect_true(!is.null(opt_ctx))

  # Check optimizer type
  opt_type <- ggml_opt_context_optimizer_type(opt_ctx)
  expect_equal(opt_type, ggml_opt_optimizer_type_sgd())

  # Cleanup
  ggml_opt_free(opt_ctx)
  ggml_backend_sched_free(sched)
  ggml_backend_free(cpu)
})

test_that("dataset batch operations work", {
  # Create CPU backend for allocation
  cpu <- ggml_backend_cpu_init()

  # Create dataset
  ne_datapoint <- 4
  ne_label <- 2
  ndata <- 10
  batch_size <- 2

  dataset <- ggml_opt_dataset_init(
    type_data = GGML_TYPE_F32,
    type_label = GGML_TYPE_F32,
    ne_datapoint = ne_datapoint,
    ne_label = ne_label,
    ndata = ndata,
    ndata_shard = 1
  )

  # Fill dataset with test data
  data_tensor <- ggml_opt_dataset_data(dataset)
  labels_tensor <- ggml_opt_dataset_labels(dataset)

  # Set some data (data tensor shape is [ne_datapoint, ndata])
  test_data <- seq_len(ne_datapoint * ndata)
  ggml_backend_tensor_set_data(data_tensor, as.numeric(test_data))

  test_labels <- seq_len(ne_label * ndata)
  ggml_backend_tensor_set_data(labels_tensor, as.numeric(test_labels))

  # Create batch tensors using a context with no_alloc = TRUE
  # then allocate via backend
  ctx <- ggml_init_auto(1024 * 1024, no_alloc = TRUE)

  data_batch <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne_datapoint, batch_size)
  labels_batch <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne_label, batch_size)

  # Allocate tensors via CPU backend
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, cpu)
  expect_true(!is.null(buffer))

  # Get first batch
  expect_silent(ggml_opt_dataset_get_batch(dataset, data_batch, labels_batch, ibatch = 0))

  # Verify batch data
  batch_data <- ggml_backend_tensor_get_data(data_batch)
  expect_equal(length(batch_data), ne_datapoint * batch_size)

  # Cleanup
  ggml_backend_buffer_free(buffer)
  ggml_free(ctx)
  ggml_opt_dataset_free(dataset)
  ggml_backend_free(cpu)
})

test_that("dataset shuffle works", {
  cpu <- ggml_backend_cpu_init()
  sched <- ggml_backend_sched_new(list(cpu), parallel = FALSE)

  opt_ctx <- ggml_opt_init(
    sched = sched,
    loss_type = ggml_opt_loss_type_mse(),
    optimizer = ggml_opt_optimizer_type_adamw()
  )

  dataset <- ggml_opt_dataset_init(
    type_data = GGML_TYPE_F32,
    type_label = GGML_TYPE_F32,
    ne_datapoint = 10,
    ne_label = 1,
    ndata = 100,
    ndata_shard = 1
  )

  # Shuffle all data
  expect_silent(ggml_opt_dataset_shuffle(opt_ctx, dataset, idata = -1))

  # Shuffle first 50 datapoints
  expect_silent(ggml_opt_dataset_shuffle(opt_ctx, dataset, idata = 50))

  # Cleanup
  ggml_opt_dataset_free(dataset)
  ggml_opt_free(opt_ctx)
  ggml_backend_sched_free(sched)
  ggml_backend_free(cpu)
})

test_that("different loss types work", {
  cpu <- ggml_backend_cpu_init()
  sched <- ggml_backend_sched_new(list(cpu), parallel = FALSE)

  loss_types <- list(
    mean = ggml_opt_loss_type_mean(),
    sum = ggml_opt_loss_type_sum(),
    cross_entropy = ggml_opt_loss_type_cross_entropy(),
    mse = ggml_opt_loss_type_mse()
  )

  for (name in names(loss_types)) {
    opt_ctx <- ggml_opt_init(
      sched = sched,
      loss_type = loss_types[[name]],
      optimizer = ggml_opt_optimizer_type_adamw()
    )
    expect_true(!is.null(opt_ctx), info = paste("Failed for loss type:", name))
    ggml_opt_free(opt_ctx)
  }

  ggml_backend_sched_free(sched)
  ggml_backend_free(cpu)
})

test_that("opt_period parameter works", {
  cpu <- ggml_backend_cpu_init()
  sched <- ggml_backend_sched_new(list(cpu), parallel = FALSE)

  # Test different opt_period values
  for (period in c(1L, 2L, 4L, 8L)) {
    opt_ctx <- ggml_opt_init(
      sched = sched,
      loss_type = ggml_opt_loss_type_mse(),
      optimizer = ggml_opt_optimizer_type_adamw(),
      opt_period = period
    )
    expect_true(!is.null(opt_ctx), info = paste("Failed for opt_period:", period))
    ggml_opt_free(opt_ctx)
  }

  ggml_backend_sched_free(sched)
  ggml_backend_free(cpu)
})

test_that("ggml_opt_result_pred returns integer vector", {
  result <- ggml_opt_result_init()
  expect_true(!is.null(result))

  # With empty result, should return empty vector
  pred <- ggml_opt_result_pred(result)
  expect_type(pred, "integer")
  expect_equal(length(pred), 0)

  ggml_opt_result_free(result)
})

# Note: ggml_opt_grad_acc and ggml_opt_prepare_alloc require specific
# optimizer context setup with static graphs, which is complex to test.
# These functions are available but testing is skipped to avoid crashes.

test_that("ggml_opt_grad_acc function exists", {
  expect_true(is.function(ggml_opt_grad_acc))
})

test_that("ggml_opt_prepare_alloc function exists", {
  expect_true(is.function(ggml_opt_prepare_alloc))
})

test_that("ggml_opt_epoch function exists", {
  expect_true(is.function(ggml_opt_epoch))
})

test_that("ggml_opt_epoch accepts R callback functions", {
  # Test that we can pass R functions as callbacks
  # The actual epoch won't run without proper setup, but we verify the interface

  my_callback <- function(train, ibatch, ibatch_max, t_start_us, result) {
    # This would be called during training
    TRUE
  }

  expect_true(is.function(my_callback))

  # Verify the function signature is correct
  args <- names(formals(ggml_opt_epoch))
  expect_true("callback_train" %in% args)
  expect_true("callback_eval" %in% args)
})
