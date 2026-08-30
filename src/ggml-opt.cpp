#include "ggml-opt.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cinttypes>
#include <map>
#include <random>
#include <string>
#include <vector>

// Use R's REprintf for stderr output — fprintf(stderr,...) crashes in C++ within R packages
// because R may redirect stderr and the C++ runtime sees a different (invalid) symbol.
#include <R_ext/Print.h>
#define GGML_OPT_LOG(...) REprintf(__VA_ARGS__)
#define GGML_OPT_FFLUSH() do { } while(0)

// Get batch size from the last dimension of an N-D tensor.
// For 2D [features, batch] returns ne[1], for 4D [W, H, C, batch] returns ne[3].
static int64_t ggml_opt_batch_size(const struct ggml_tensor * t) {
    return t->ne[ggml_n_dims(t) - 1];
}

// Get the number of elements per sample (all dims except the batch dim).
static int64_t ggml_opt_ne_per_sample(const struct ggml_tensor * t) {
    int64_t n = 1;
    const int ndims = ggml_n_dims(t);
    for (int i = 0; i < ndims - 1; ++i) {
        n *= t->ne[i];
    }
    return n;
}

struct ggml_opt_dataset {
    struct ggml_context   * ctx    = nullptr;
    ggml_backend_buffer_t   buf    = nullptr;
    struct ggml_tensor    * data   = nullptr;
    struct ggml_tensor    * labels = nullptr;

    // Optional loss weights [ne0_weights, ndata], lazily allocated in a
    // separate ctx/buffer so ggml_opt_dataset_init() and its buffer stay
    // untouched. Used by GGML_OPT_LOSS_TYPE_WEIGHTED_MEAN_SQUARED_ERROR.
    // ne0_weights == 1 is the per-datapoint case (broadcast over the outputs);
    // ne0_weights == ne_label is a per-output-element mask.
    struct ggml_context   * ctx_weights = nullptr;
    ggml_backend_buffer_t   buf_weights = nullptr;
    struct ggml_tensor    * weights     = nullptr;
    size_t                  nbs_weights = 0;
    int64_t                 ne0_weights = 0;

    int64_t ndata       = -1;
    int64_t ndata_shard = -1;
    size_t  nbs_data    = -1;
    size_t  nbs_labels  = -1;

    std::vector<int64_t> permutation;
};

struct ggml_opt_context {
    ggml_backend_sched_t       backend_sched        = nullptr;
    ggml_cgraph              * allocated_graph      = nullptr;
    ggml_cgraph              * allocated_graph_copy = nullptr;
    struct ggml_context      * ctx_static           = nullptr;
    struct ggml_context      * ctx_momenta          = nullptr;
    struct ggml_context      * ctx_cpu              = nullptr;
    struct ggml_context      * ctx_compute          = nullptr;
    struct ggml_context      * ctx_copy             = nullptr;
    ggml_backend_buffer_t      buf_static           = nullptr;
    ggml_backend_buffer_t      buf_momenta          = nullptr;
    ggml_backend_buffer_t      buf_cpu              = nullptr;
    std::mt19937               rng;
    enum ggml_opt_build_type   build_type;
    enum ggml_opt_build_type   build_type_alloc;

    struct ggml_tensor * inputs = nullptr;

    // Multi-loss (ggmlR extension): the model may have several output heads,
    // each with its own labels, loss type and weight. Everything below is
    // indexed by head; n_loss == 1 reproduces the original single-loss layout
    // exactly (same graph, same node names — see ggml_opt_build).
    //
    // NOTE: loss_w[] (per-head weight in the total loss) and loss_weights[]
    // (per-datapoint weights used by WEIGHTED_MEAN_SQUARED_ERROR) are
    // different things and must not be conflated.
    std::vector<enum ggml_opt_loss_type> loss_type;
    std::vector<struct ggml_tensor *>    outputs;
    std::vector<struct ggml_tensor *>    labels;
    std::vector<struct ggml_tensor *>    loss_weights; // per-datapoint weights for weighted MSE
    std::vector<float>                   loss_w;       // per-head weight in the total loss
    int64_t                              loss_mask_ne0 = 0; // weighted-MSE weight width; 0/1 = per-datapoint
    std::vector<struct ggml_tensor *>    losses;       // per-head scalars, before reduction
    std::vector<bool>                    loss_per_datapoint_head;

    // Offset of each head's labels within a dataset label row, in elements.
    // The dataset stores all heads' labels concatenated along ne[0]; empty (or
    // all-zero for a single head) means "the whole row", i.e. the legacy layout.
    std::vector<int64_t>                 labels_offs;

    // Total loss: sum_i loss_w[i] * losses[i]. This is the only tensor that
    // carries GGML_TENSOR_FLAG_LOSS — the per-head scalars must NOT be flagged,
    // or the autodiff would add them a second time on top of the reduction
    // (the flag is documented as "multiple loss tensors add up").
    struct ggml_tensor * loss = nullptr;

    std::vector<struct ggml_tensor *> pred;
    std::vector<struct ggml_tensor *> ncorrect;

    struct ggml_cgraph * gf      = nullptr;
    struct ggml_cgraph * gb_grad = nullptr;
    struct ggml_cgraph * gb_opt  = nullptr;
    bool static_graphs           = false;
    bool eval_ready              = false;
    std::vector<struct ggml_tensor *> grad_accs;
    std::vector<struct ggml_tensor *> grad_m;
    std::vector<struct ggml_tensor *> grad_v;

    int64_t iter               = 1;
    int32_t opt_period         = 1;
    int32_t opt_i              = 0;
    bool    loss_per_datapoint = false;

    ggml_opt_get_optimizer_params get_opt_pars    = nullptr;
    void *                        get_opt_pars_ud = nullptr;
    struct ggml_tensor *          opt_step_params = nullptr; // Stores output of get_opt_pars.

    enum ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
};

struct ggml_opt_result {
    int64_t              ndata    = 0;
    std::vector<float>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;

    // Per-head results (ggmlR multi-loss extension), indexed by output head.
    // The scalar fields above stay head 0 so existing callers are unaffected.
    // ncorrect_head is -1 for heads that have no accuracy (non-CE losses),
    // matching the convention of the scalar ncorrect.
    std::vector<std::vector<float>> loss_head;
    std::vector<int64_t>            ncorrect_head;

    int64_t opt_period         = -1;
    bool    loss_per_datapoint = false;
};

// ====== Dataset ======

ggml_opt_dataset_t ggml_opt_dataset_init(
        enum ggml_type type_data,
        enum ggml_type type_label,
        int64_t        ne_datapoint,
        int64_t        ne_label,
        int64_t        ndata,
        int64_t        ndata_shard) {
    GGML_ASSERT(ne_datapoint >  0);
    GGML_ASSERT(ne_label     >= 0);
    GGML_ASSERT(ndata        >  0);
    GGML_ASSERT(ndata_shard  >  0);

    ggml_opt_dataset_t result = new ggml_opt_dataset;
    result->ndata       = ndata;
    result->ndata_shard = ndata_shard;

    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        result->ctx = ggml_init(params);
    }

    result->data = ggml_new_tensor_2d(result->ctx, type_data, ne_datapoint, ndata);
    result->nbs_data = ggml_nbytes(result->data) * ndata_shard/ndata;

    if (ne_label > 0) {
        result->labels = ggml_new_tensor_2d(result->ctx, type_label, ne_label, ndata);
        result->nbs_labels = ggml_nbytes(result->labels) * ndata_shard/ndata;
    } else {
        result->labels = nullptr;
        result->nbs_labels = 0;
    }

    result->buf = ggml_backend_alloc_ctx_tensors_from_buft(result->ctx, ggml_backend_cpu_buffer_type());

    const int64_t nshards = ndata/ndata_shard;
    result->permutation.resize(nshards);
    for (int64_t i = 0; i < nshards; ++i) {
        result->permutation[i] = i;
    }
    return result;
}

void ggml_opt_dataset_free(ggml_opt_dataset_t dataset) {
    ggml_backend_buffer_free(dataset->buf);
    ggml_free(dataset->ctx);
    if (dataset->buf_weights) {
        ggml_backend_buffer_free(dataset->buf_weights);
    }
    if (dataset->ctx_weights) {
        ggml_free(dataset->ctx_weights);
    }
    delete dataset;
}

struct ggml_tensor * ggml_opt_dataset_weights(ggml_opt_dataset_t dataset, int64_t ne0) {
    GGML_ASSERT(ne0 > 0 && "loss weight width must be positive");

    if (!dataset->weights) {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        dataset->ctx_weights = ggml_init(params);
        dataset->weights     = ggml_new_tensor_2d(dataset->ctx_weights, GGML_TYPE_F32, ne0, dataset->ndata);
        dataset->nbs_weights = ggml_nbytes(dataset->weights) * dataset->ndata_shard/dataset->ndata;
        dataset->ne0_weights = ne0;
        dataset->buf_weights = ggml_backend_alloc_ctx_tensors_from_buft(
            dataset->ctx_weights, ggml_backend_cpu_buffer_type());
    } else {
        // Returning the cached tensor under a different width would stay
        // broadcastable in ggml_mul() and silently weight the wrong thing, so
        // a mismatch has to be loud.
        GGML_ASSERT(dataset->ne0_weights == ne0 &&
            "ggml_opt_dataset_weights() called with a different width than the allocation");
    }
    return dataset->weights;
}

void ggml_opt_dataset_get_batch_weights(ggml_opt_dataset_t dataset, struct ggml_tensor * weights_batch, int64_t ibatch) {
    GGML_ASSERT(weights_batch && ggml_is_contiguous(weights_batch));
    GGML_ASSERT(dataset->weights && "ggml_opt_dataset_weights() must be called first");
    // The batch tensor must be as wide as the dataset's weights, otherwise the
    // shard arithmetic below would slice rectangular blocks of the wrong shape
    // while still copying a byte count that happens to divide evenly.
    GGML_ASSERT(weights_batch->ne[0] == dataset->weights->ne[0] &&
        "weights batch width does not match the dataset's loss weights");

    const size_t nb_weights_batch = ggml_nbytes(weights_batch);
    GGML_ASSERT(nb_weights_batch % dataset->nbs_weights == 0);
    const int64_t shards_per_batch = nb_weights_batch / dataset->nbs_weights;

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];
        const char * ptr_weights = (const char *) dataset->weights->data + ishard*dataset->nbs_weights;
        ggml_backend_tensor_set(weights_batch, ptr_weights, ishard_batch*dataset->nbs_weights, dataset->nbs_weights);
    }
}

int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset) {
    return dataset->ndata;
}

struct ggml_tensor * ggml_opt_dataset_data(ggml_opt_dataset_t dataset) {
    return dataset->data;
}

struct ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset) {
    return dataset->labels;
}

void ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata) {
    GGML_ASSERT(idata <= dataset->ndata);

    if (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        return;
    }

    GGML_ASSERT(idata % dataset->ndata_shard == 0);
    const int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

void ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, struct ggml_tensor * data_batch, struct ggml_tensor * labels_batch, int64_t ibatch) {
    GGML_ASSERT(   data_batch && ggml_is_contiguous(data_batch));
    GGML_ASSERT(!labels_batch || ggml_is_contiguous(labels_batch));
    GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    GGML_ASSERT(                   data_batch->type == dataset->data->type);
    GGML_ASSERT(!labels_batch || labels_batch->type == dataset->labels->type);

    const size_t nb_data_batch = ggml_nbytes(data_batch);
    GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);
    const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    if (labels_batch) {
        const size_t nb_labels_batch = ggml_nbytes(labels_batch);
        GGML_ASSERT(nb_labels_batch == shards_per_batch*dataset->nbs_labels);
    }

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        const char * ptr_data = (const char *) dataset->data->data + ishard*dataset->nbs_data;
        ggml_backend_tensor_set(data_batch, ptr_data, ishard_batch*dataset->nbs_data, dataset->nbs_data);

        if (!labels_batch) {
            continue;
        }

        const char * ptr_labels = (const char *) dataset->labels->data + ishard*dataset->nbs_labels;
        ggml_backend_tensor_set(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
    }
}

void ggml_opt_dataset_get_batch_head(
        ggml_opt_dataset_t   dataset,
        struct ggml_tensor * data_batch,
        struct ggml_tensor * labels_batch,
        int64_t              labels_off,
        int64_t              ibatch) {
    GGML_ASSERT(!data_batch   || ggml_is_contiguous(data_batch));
    GGML_ASSERT(labels_batch  && ggml_is_contiguous(labels_batch));
    GGML_ASSERT(dataset->labels && "dataset has no labels");
    GGML_ASSERT(labels_batch->type == dataset->labels->type);

    const int64_t ne_label_all  = dataset->labels->ne[0];
    const int64_t ne_label_head = labels_batch->ne[0];
    GGML_ASSERT(labels_off >= 0 && labels_off + ne_label_head <= ne_label_all &&
        "label head slice out of range");

    // Shard count is taken from the data batch when present, so that a head's
    // batch composition is identical to ggml_opt_dataset_get_batch's.
    int64_t shards_per_batch;
    if (data_batch) {
        GGML_ASSERT(data_batch->type == dataset->data->type);
        const size_t nb_data_batch = ggml_nbytes(data_batch);
        GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);
        shards_per_batch = nb_data_batch / dataset->nbs_data;
    } else {
        const int64_t ndata_batch = labels_batch->ne[1];
        GGML_ASSERT(ndata_batch % dataset->ndata_shard == 0);
        shards_per_batch = ndata_batch / dataset->ndata_shard;
    }

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    const size_t nb_label     = ggml_type_size(dataset->labels->type);
    const size_t nb_row_all   = ne_label_all *nb_label; // one full label row (all heads)
    const size_t nb_row_head  = ne_label_head*nb_label; // this head's slice of a row

    // A head's slice is strided inside the concatenated label rows, so it cannot
    // be handed to the backend as one range. Rather than issuing one
    // ggml_backend_tensor_set per datapoint -- which on a GPU backend is a
    // separate transfer each -- the shard is gathered into a contiguous host
    // buffer and uploaded in a single call. The whole-row case (one head owning
    // the entire label row) needs no gathering at all.
    const bool slice_is_whole_row = (labels_off == 0 && ne_label_head == ne_label_all);
    std::vector<char> staging;
    if (!slice_is_whole_row) {
        staging.resize(size_t(dataset->ndata_shard) * nb_row_head);
    }

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        if (data_batch) {
            const char * ptr_data = (const char *) dataset->data->data + ishard*dataset->nbs_data;
            ggml_backend_tensor_set(data_batch, ptr_data, ishard_batch*dataset->nbs_data, dataset->nbs_data);
        }

        const char * ptr_shard = (const char *) dataset->labels->data
            + ishard*dataset->ndata_shard*nb_row_all + labels_off*nb_label;
        const size_t off_dst = size_t(ishard_batch) * dataset->ndata_shard * nb_row_head;

        if (slice_is_whole_row) {
            ggml_backend_tensor_set(labels_batch, ptr_shard, off_dst,
                size_t(dataset->ndata_shard) * nb_row_head);
            continue;
        }

        for (int64_t idata_shard = 0; idata_shard < dataset->ndata_shard; ++idata_shard) {
            memcpy(staging.data() + size_t(idata_shard)*nb_row_head,
                   ptr_shard + size_t(idata_shard)*nb_row_all,
                   nb_row_head);
        }
        ggml_backend_tensor_set(labels_batch, staging.data(), off_dst, staging.size());
    }
}

void ggml_opt_dataset_get_batch_host(ggml_opt_dataset_t dataset, void * data_batch, size_t nb_data_batch, void * labels_batch, int64_t ibatch) {
    GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);

    const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        const char * ptr_data       = (const char *) dataset->data->data + ishard      *dataset->nbs_data;
        char       * ptr_data_batch = (char       *) data_batch          + ishard_batch*dataset->nbs_data;
        memcpy(ptr_data_batch, ptr_data, dataset->nbs_data);

        if (!labels_batch) {
            continue;
        }

        const char * ptr_labels       = (const char *) dataset->labels->data + ishard      *dataset->nbs_labels;
        char       * ptr_labels_batch = (char       *) labels_batch          + ishard_batch*dataset->nbs_labels;
        memcpy(ptr_labels_batch, ptr_labels, dataset->nbs_labels);
    }
}

// ====== Model / Context ======

struct ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(void * userdata) {
    GGML_UNUSED(userdata);

    ggml_opt_optimizer_params result;

    result.adamw.alpha = 0.001f;
    result.adamw.beta1 = 0.9f;
    result.adamw.beta2 = 0.999f;
    result.adamw.eps   = 1e-8f;
    result.adamw.wd    = 0.0f;

    result.sgd.alpha   = 1e-3f;
    result.sgd.wd      = 0.0f;

    return result;
}


struct ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(void * userdata) {
    return *((struct ggml_opt_optimizer_params *) userdata);
}

struct ggml_opt_params ggml_opt_default_params(
        ggml_backend_sched_t      backend_sched,
        enum ggml_opt_loss_type   loss_type) {
    return {
        /*backend_sched   =*/ backend_sched,
        /*ctx_compute     =*/ nullptr,
        /*inputs          =*/ nullptr,
        /*logits          =*/ nullptr,
        /*loss_type       =*/ loss_type,
        /*build_type      =*/ GGML_OPT_BUILD_TYPE_OPT,
        /*n_loss          =*/ 0,
        /*outputs_multi   =*/ nullptr,
        /*loss_type_multi =*/ nullptr,
        /*loss_w          =*/ nullptr,
        /*loss_mask_ne0   =*/ 0,
        /*opt_period      =*/ 1,
        /*get_opt_pars    =*/ ggml_opt_get_default_optimizer_params,
        /*get_opt_pars_ud =*/ nullptr,
        /*optimizer       =*/ GGML_OPT_OPTIMIZER_TYPE_ADAMW,
    };
}

struct ggml_opt_params ggml_opt_default_params_multi(
        ggml_backend_sched_t            backend_sched,
        int64_t                         n_loss,
        const enum ggml_opt_loss_type * loss_type,
        const float                   * loss_w) {
    GGML_ASSERT(n_loss >= 1);
    GGML_ASSERT(loss_type);
    // loss_type[0] also fills the legacy scalar field so that anything reading
    // params.loss_type without knowing about multi-loss sees a sane value.
    struct ggml_opt_params result = ggml_opt_default_params(backend_sched, loss_type[0]);
    result.n_loss          = n_loss;
    result.loss_type_multi = loss_type;
    result.loss_w          = loss_w;
    return result;
}

static ggml_tensor * map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }

    if (tensor_map.find(tensor) != tensor_map.end()) {
        return tensor_map[tensor];
    }

    ggml_tensor * new_tensor = ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    memcpy(new_tensor->op_params, tensor->op_params, sizeof(tensor->op_params));
    strcpy(new_tensor->name, tensor->name);
    new_tensor->data = tensor->data;
    new_tensor->buffer = tensor->buffer;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    return new_tensor;
}

static ggml_cgraph * dup_graph(ggml_context * ctx, ggml_cgraph * src) {
    std::map<ggml_tensor *, ggml_tensor *> tensor_map;

    ggml_cgraph * dst = ggml_new_graph_custom(ctx, src->size, /*grads =*/ true);

    for (int i = 0; i < src->n_leafs; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->leafs[i]));
    }
    GGML_ASSERT(dst->n_leafs == src->n_leafs);
    for (int i = 0; i < src->n_nodes; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->nodes[i]));
    }
    GGML_ASSERT(dst->n_nodes == src->n_nodes);
    for (int i = 0; i < src->n_nodes; ++i) {
        const size_t igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
        const size_t igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

        GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
        GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

        dst->grads[igrad_dst]     = src->grads[igrad_src];
        dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }

    return dst;
}

// Build the loss scalar for a single output head.
//
// Writes the head's labels / loss weights into *labels_out and
// *loss_weights_out (left as nullptr for loss types that need neither) and
// returns the resulting scalar. The caller owns the reduction over heads and
// is the only one allowed to call ggml_set_loss (see ggml_opt_context::loss).
//
// mask_ne0 is the width of the weighted-MSE weight tensor: 0 or 1 for the
// per-datapoint case, outputs->ne[0] for a per-output mask. Other loss types
// ignore it.
//
// `suffix` is appended to every node name so that heads stay distinguishable
// in graph dumps; it must be "" for the single-head case to keep the node
// names bit-identical to the pre-multi-loss graph.
static struct ggml_tensor * build_one_loss(
        struct ggml_context      * ctx_results,
        struct ggml_tensor       * outputs,
        enum ggml_opt_loss_type    loss_type,
        int32_t                    opt_period,
        int64_t                    mask_ne0,
        const std::string        & suffix,
        struct ggml_tensor      ** labels_out,
        struct ggml_tensor      ** loss_weights_out,
        bool                     * loss_per_datapoint_out) {
    struct ggml_tensor * labels       = nullptr;
    struct ggml_tensor * loss_weights = nullptr;
    struct ggml_tensor * loss         = nullptr;

    auto name = [&suffix](struct ggml_tensor * t, const char * base) {
        ggml_set_name(t, (std::string(base) + suffix).c_str());
    };

    switch (loss_type) {
        case GGML_OPT_LOSS_TYPE_MEAN: {
            loss = ggml_sum(ctx_results, outputs);
            name(loss, "loss_sum");
            const float scale = 1.0f / (opt_period * ggml_nelements(outputs));
            loss = ggml_scale(ctx_results, loss, scale);
            name(loss, "loss_mean");
            *loss_per_datapoint_out = true;
            break;
        }
        case GGML_OPT_LOSS_TYPE_SUM: {
            loss = ggml_sum(ctx_results, outputs);
            name(loss, "loss_sum");
            *loss_per_datapoint_out = false;
            break;
        }
        case GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
            labels = ggml_dup_tensor(ctx_results, outputs);
            ggml_set_input(labels);
            name(labels, "labels");
            loss = ggml_cross_entropy_loss(ctx_results, outputs, labels);
            name(loss, "loss_cross_entropy");
            if (opt_period > 1) {
                loss = ggml_scale(ctx_results, loss, 1.0f / opt_period);
                name(loss, "loss_cross_entropy_scaled");
            }
            *loss_per_datapoint_out = true;
            break;
        }
        case GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR: {
            labels = ggml_dup_tensor(ctx_results, outputs);
            ggml_set_input(labels);
            name(labels, "labels");
            loss = ggml_sub(ctx_results, outputs, labels);
            name(loss, "loss_error");
            loss = ggml_sqr(ctx_results, loss);
            name(loss, "loss_squared_error");
            loss = ggml_sum(ctx_results, loss);
            name(loss, "loss_sum_squared_error");
            const float scale = 1.0f / (opt_period * ggml_nelements(outputs));
            loss = ggml_scale(ctx_results, loss, scale);
            name(loss, "loss_mean_squared_error");
            *loss_per_datapoint_out = true;
            break;
        }
        case GGML_OPT_LOSS_TYPE_WEIGHTED_MEAN_SQUARED_ERROR: {
            // sum( w * (out - y)^2 ) / (opt_period * nelements).
            //
            // w is either one scalar per datapoint, broadcast over the output
            // dimension, or a full per-output-element mask. Note that the
            // denominator is the output's element count either way: a mask
            // zeroes terms out of the numerator but does NOT renormalise by
            // the number of active coordinates.
            labels = ggml_dup_tensor(ctx_results, outputs);
            ggml_set_input(labels);
            name(labels, "labels");

            // weights: [1, batch] (broadcast over ne[0]) or [ne0, batch] (mask).
            const int64_t nbatch = outputs->ne[ggml_n_dims(outputs) - 1];
            const int64_t ne0    = mask_ne0 <= 0 ? 1 : mask_ne0;
            GGML_ASSERT((ne0 == 1 || ne0 == outputs->ne[0]) &&
                "weighted MSE weights must be [1, nbatch] or [outputs->ne[0], nbatch]");
            loss_weights = ggml_new_tensor_2d(ctx_results, GGML_TYPE_F32, ne0, nbatch);
            ggml_set_input(loss_weights);
            name(loss_weights, "loss_weights");

            loss = ggml_sub(ctx_results, outputs, labels);
            name(loss, "loss_error");
            loss = ggml_sqr(ctx_results, loss);
            name(loss, "loss_squared_error");
            loss = ggml_mul(ctx_results, loss, loss_weights);
            name(loss, "loss_weighted_squared_error");
            loss = ggml_sum(ctx_results, loss);
            name(loss, "loss_sum_weighted_squared_error");
            const float scale = 1.0f / (opt_period * ggml_nelements(outputs));
            loss = ggml_scale(ctx_results, loss, scale);
            name(loss, "loss_weighted_mean_squared_error");
            *loss_per_datapoint_out = true;
            break;
        }
        case GGML_OPT_LOSS_TYPE_MEAN_ABSOLUTE_ERROR: {
            // mean(|out - y|). Less sensitive to outliers than MSE: the
            // gradient is sgn(out - y), so a far-off datapoint pulls no harder
            // than a near one.
            labels = ggml_dup_tensor(ctx_results, outputs);
            ggml_set_input(labels);
            name(labels, "labels");
            loss = ggml_sub(ctx_results, outputs, labels);
            name(loss, "loss_error");
            loss = ggml_abs(ctx_results, loss);
            name(loss, "loss_absolute_error");
            loss = ggml_sum(ctx_results, loss);
            name(loss, "loss_sum_absolute_error");
            const float scale = 1.0f / (opt_period * ggml_nelements(outputs));
            loss = ggml_scale(ctx_results, loss, scale);
            name(loss, "loss_mean_absolute_error");
            *loss_per_datapoint_out = true;
            break;
        }
        case GGML_OPT_LOSS_TYPE_HUBER: {
            // Huber with delta = 1: 0.5*e^2 where |e| <= 1, else |e| - 0.5.
            // Quadratic near zero (so the gradient vanishes at the optimum,
            // unlike MAE) and linear far out (so outliers do not dominate,
            // unlike MSE).
            //
            // Built without a branch: with a = min(|e|, 1) the two pieces are
            // one expression, 0.5*a^2 + (|e| - a). For |e| <= 1 the second term
            // is zero and it reduces to 0.5*e^2; beyond that a is pinned at 1
            // and it becomes 0.5 + |e| - 1 = |e| - 0.5.
            labels = ggml_dup_tensor(ctx_results, outputs);
            ggml_set_input(labels);
            name(labels, "labels");

            struct ggml_tensor * err = ggml_sub(ctx_results, outputs, labels);
            name(err, "loss_error");
            struct ggml_tensor * abs_err = ggml_abs(ctx_results, err);
            name(abs_err, "loss_absolute_error");

            // a = min(|e|, delta), written as |e| - relu(|e| - delta).
            //
            // NOT ggml_clamp(): that op returns a view of its input and has no
            // backward pass at all (its own source says "TODO: when implement
            // backward"), so a graph containing it fails the "inplace
            // operations are currently not supported" assert in
            // ggml_build_backward_expand. relu and abs both differentiate.
            struct ggml_tensor * over = ggml_relu(ctx_results,
                ggml_scale_bias(ctx_results, abs_err, 1.0f, -1.0f));
            name(over, "loss_huber_excess");
            struct ggml_tensor * a = ggml_sub(ctx_results, abs_err, over);
            name(a, "loss_huber_clamped");

            struct ggml_tensor * quad = ggml_scale(ctx_results,
                ggml_sqr(ctx_results, a), 0.5f);
            name(quad, "loss_huber_quadratic");
            struct ggml_tensor * lin = ggml_sub(ctx_results, abs_err, a);
            name(lin, "loss_huber_linear");

            loss = ggml_add(ctx_results, quad, lin);
            name(loss, "loss_huber_elementwise");
            loss = ggml_sum(ctx_results, loss);
            name(loss, "loss_sum_huber");
            const float scale = 1.0f / (opt_period * ggml_nelements(outputs));
            loss = ggml_scale(ctx_results, loss, scale);
            name(loss, "loss_huber");
            *loss_per_datapoint_out = true;
            break;
        }
        case GGML_OPT_LOSS_TYPE_BINARY_CROSS_ENTROPY: {
            // mean( -[ y*log(p) + (1-y)*log(1-p) ] ) over every element.
            //
            // Unlike GGML_OPT_LOSS_TYPE_CROSS_ENTROPY, which softmaxes its own
            // input over the class axis and so needs logits, this one expects
            // probabilities: each output is an independent Bernoulli, which is
            // what makes it the loss for multi-label targets and for a single
            // sigmoid output. The inputs are clamped away from 0 and 1 first,
            // since log(0) is -inf and would poison the whole batch.
            labels = ggml_dup_tensor(ctx_results, outputs);
            ggml_set_input(labels);
            name(labels, "labels");

            // Squeeze into [eps, 1-eps] with an affine map rather than a clamp:
            // ggml_clamp() returns a view and has no backward pass, which trips
            // the "inplace operations are currently not supported" assert when
            // the backward graph is built. p*(1-2*eps) + eps is exact at the
            // ends, differentiable everywhere, and moves the middle by less
            // than eps.
            const float eps = 1e-7f;
            struct ggml_tensor * p = ggml_scale_bias(ctx_results, outputs,
                1.0f - 2.0f*eps, eps);
            name(p, "loss_bce_squeezed");

            // log(p) and log(1 - p); 1 - p is built as -(p - 1) since ggml has
            // no scalar-minus-tensor op.
            struct ggml_tensor * log_p = ggml_log(ctx_results, p);
            name(log_p, "loss_bce_log_p");
            struct ggml_tensor * one_minus_p = ggml_scale_bias(ctx_results, p, -1.0f, 1.0f);
            name(one_minus_p, "loss_bce_one_minus_p");
            struct ggml_tensor * log_1mp = ggml_log(ctx_results, one_minus_p);
            name(log_1mp, "loss_bce_log_1mp");

            struct ggml_tensor * one_minus_y = ggml_scale_bias(ctx_results, labels, -1.0f, 1.0f);
            name(one_minus_y, "loss_bce_one_minus_y");

            loss = ggml_add(ctx_results,
                ggml_mul(ctx_results, labels,      log_p),
                ggml_mul(ctx_results, one_minus_y, log_1mp));
            name(loss, "loss_bce_elementwise");
            loss = ggml_sum(ctx_results, loss);
            name(loss, "loss_sum_bce");
            // Negated by the scale, so no separate ggml_neg node.
            const float scale = -1.0f / (opt_period * ggml_nelements(outputs));
            loss = ggml_scale(ctx_results, loss, scale);
            name(loss, "loss_binary_cross_entropy");
            *loss_per_datapoint_out = true;
            break;
        }
    }

    *labels_out       = labels;
    *loss_weights_out = loss_weights;
    return loss;
}

static void ggml_opt_build(ggml_opt_context_t opt_ctx) {
    GGML_ASSERT(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc");
    GGML_ASSERT((!opt_ctx->static_graphs || opt_ctx->inputs->data) && "when using static graphs the inputs must be allocated statically");

    const enum ggml_opt_optimizer_type optimizer = opt_ctx->optimizer;

    const bool accumulate = opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD &&
        !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period == 1);

    const bool need_momenta = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT &&
        opt_ctx->optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    ggml_set_input(opt_ctx->inputs);
    for (struct ggml_tensor * out : opt_ctx->outputs) {
        ggml_set_output(out);
    }

    int n_param = 0;
    for (int i = 0; i < opt_ctx->gf->n_nodes; ++i) {
        const struct ggml_tensor * node = opt_ctx->gf->nodes[i];
        if (node->flags & GGML_TENSOR_FLAG_PARAM) {
            n_param++;
        }
        GGML_ASSERT(!(node->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
    }

    if (!opt_ctx->ctx_static) {
        // The static context is used for:
        //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        // NOTE: optimizer momenta live in ctx_momenta instead, see below.
        //
        // Every one of these is per output head, so the per-head costs scale
        // with n_loss; undersizing here does not degrade gracefully, it aborts
        // the process from ggml_new_object's GGML_ASSERT(obj_new).
        const size_t n_loss = opt_ctx->loss_type.size();
        const size_t tensors_per_param = accumulate ? 1 : 0;
        // 16 per head (labels, loss chain, pred, ncorrect, ...) plus a small
        // allowance for the cross-head reduction (scale + add per head).
        //
        // The budget is set by the LONGEST loss chain, not the typical one:
        // MSE needs 5 nodes, but Huber needs 12 and binary cross-entropy 11. A
        // head using one of those with the old allowance of 9 aborted the
        // process instead of failing gracefully, so this must be raised
        // whenever a longer loss chain is added below.
        const size_t tensors_const = opt_ctx->static_graphs ? 16*n_loss + 2*n_loss : 0;
        const size_t size_meta = (n_loss + tensors_per_param*n_param + tensors_const) * ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        opt_ctx->ctx_static = ggml_init(params);
    }

    if (need_momenta && !opt_ctx->ctx_momenta) {
        // The momenta context holds the AdamW m/v tensors (2 per param).
        //
        // These are kept separate from ctx_static because GGML_OP_OPT_STEP_ADAMW *writes* to
        // them even though they are passed as sources. ggml_backend_sched only copies split
        // inputs one way (into the executing backend, see ggml_backend_sched_compute_splits);
        // there is no copy back. So if m/v live on a different backend than the one running
        // the optimizer step, every update is written into a scratch copy and discarded,
        // leaving m/v pinned at their initial values. ctx_static must stay on CPU because it
        // also holds loss/labels tensors whose ops are not supported on GPU backends, hence
        // the split: momenta are allocated on the backend that actually owns the params.
        const size_t size_meta = 2 * n_param * ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        opt_ctx->ctx_momenta = ggml_init(params);
    }
    GGML_ASSERT(opt_ctx->build_type <= opt_ctx->build_type_alloc);

    {
        // The cpu context is allocated statically if using static graphs, dynamically otherwise.
        // It is used for:
        //   - optimizer parameters (1 shared for all optimizer invocations)
        const size_t size_meta = 1 * ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_free(opt_ctx->ctx_cpu);
        opt_ctx->ctx_cpu = ggml_init(params);

        ggml_backend_buffer_free(opt_ctx->buf_cpu);
        opt_ctx->buf_cpu = nullptr;
    }

    struct ggml_context * ctx_results = opt_ctx->static_graphs ? opt_ctx->ctx_static : opt_ctx->ctx_compute;

    const size_t n_loss = opt_ctx->loss_type.size();
    GGML_ASSERT(n_loss >= 1);
    GGML_ASSERT(opt_ctx->outputs.size() == n_loss);
    GGML_ASSERT(opt_ctx->loss_w.size()  == n_loss);

    opt_ctx->labels.assign(n_loss, nullptr);
    opt_ctx->loss_weights.assign(n_loss, nullptr);
    opt_ctx->losses.assign(n_loss, nullptr);
    opt_ctx->loss_per_datapoint_head.assign(n_loss, false);
    opt_ctx->pred.assign(n_loss, nullptr);
    opt_ctx->ncorrect.assign(n_loss, nullptr);

    for (size_t i = 0; i < n_loss; ++i) {
        // Single-head models keep the original node names ("labels", "loss_mean", ...)
        // so their graph stays bit-identical to the pre-multi-loss version.
        const std::string suffix = n_loss == 1 ? "" : "_" + std::to_string(i);

        bool per_datapoint = false;
        opt_ctx->losses[i] = build_one_loss(
            ctx_results, opt_ctx->outputs[i], opt_ctx->loss_type[i], opt_ctx->opt_period,
            opt_ctx->loss_mask_ne0, suffix,
            &opt_ctx->labels[i], &opt_ctx->loss_weights[i], &per_datapoint);
        opt_ctx->loss_per_datapoint_head[i] = per_datapoint;
    }

    // Reduction over heads: loss = sum_i loss_w[i] * losses[i].
    // With one head of weight 1 no extra nodes are emitted at all, so the
    // graph matches the original single-loss build node for node.
    if (n_loss == 1 && opt_ctx->loss_w[0] == 1.0f) {
        opt_ctx->loss = opt_ctx->losses[0];
    } else {
        struct ggml_tensor * total = nullptr;
        for (size_t i = 0; i < n_loss; ++i) {
            struct ggml_tensor * term = opt_ctx->losses[i];
            if (opt_ctx->loss_w[i] != 1.0f) {
                term = ggml_scale(ctx_results, term, opt_ctx->loss_w[i]);
                ggml_set_name(term, ("loss_scaled_" + std::to_string(i)).c_str());
            }
            total = total ? ggml_add(ctx_results, total, term) : term;
        }
        opt_ctx->loss = total;
        ggml_set_name(opt_ctx->loss, "loss_total");
    }

    // loss_per_datapoint is a single flag on ggml_opt_result, but heads may mix
    // MEAN/SUM semantics. Only a consistent set has a well-defined meaning for
    // the aggregate; reject mixtures instead of silently reporting a wrong loss.
    for (size_t i = 1; i < n_loss; ++i) {
        GGML_ASSERT(opt_ctx->loss_per_datapoint_head[i] == opt_ctx->loss_per_datapoint_head[0] &&
            "all loss heads must agree on per-datapoint vs total loss semantics (do not mix SUM with MEAN/CE/MSE)");
    }
    opt_ctx->loss_per_datapoint = opt_ctx->loss_per_datapoint_head[0];

    ggml_set_output(opt_ctx->loss);
    // Only the total carries the LOSS flag: the flag makes autodiff seed a
    // gradient at every tensor that has it ("multiple loss tensors add up"),
    // so flagging the per-head scalars too would count them twice.
    ggml_set_loss(opt_ctx->loss);
    ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);

    for (size_t i = 0; i < n_loss; ++i) {
        if (opt_ctx->loss_type[i] != GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
            continue;
        }
        const std::string suffix = n_loss == 1 ? "" : "_" + std::to_string(i);

        opt_ctx->pred[i] = ggml_argmax(ctx_results, opt_ctx->outputs[i]);
        ggml_set_name(opt_ctx->pred[i], ("pred" + suffix).c_str());
        ggml_set_output(opt_ctx->pred[i]);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred[i]);

        opt_ctx->ncorrect[i] = ggml_count_equal(ctx_results, opt_ctx->pred[i],
            ggml_argmax(ctx_results, opt_ctx->labels[i]));
        ggml_set_name(opt_ctx->ncorrect[i], ("ncorrect" + suffix).c_str());
        ggml_set_output(opt_ctx->ncorrect[i]);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->ncorrect[i]);
    }

    if (opt_ctx->buf_static) {
        if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_FORWARD) {
            return;
        }
    } else if (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_FORWARD) {
        // Allocate ctx_static on CPU (last backend) — contains loss/labels/results
        // which may use ops not supported on GPU backends (e.g. CROSS_ENTROPY_LOSS on Vulkan)
        const int n_backends = ggml_backend_sched_get_n_backends(opt_ctx->backend_sched);
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, n_backends - 1));
        return;
    }

    if (opt_ctx->grad_accs.empty()) {
        GGML_ASSERT(opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD);

        const int n_nodes = opt_ctx->gf->n_nodes;
        opt_ctx->grad_accs.resize(n_nodes);
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor * node = opt_ctx->gf->nodes[i];
            if ((accumulate && (node->flags & GGML_TENSOR_FLAG_PARAM)) || (node->flags & GGML_TENSOR_FLAG_LOSS)) {
                opt_ctx->grad_accs[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            } else {
                opt_ctx->grad_accs[i] = nullptr;
            }
        }

        if (need_momenta && opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_OPT) {
            opt_ctx->grad_m.resize(n_nodes);
            opt_ctx->grad_v.resize(n_nodes);
            for (int i = 0; i < n_nodes; ++i) {
                ggml_tensor * node = opt_ctx->gf->nodes[i];
                if (node->flags & GGML_TENSOR_FLAG_PARAM) {
                    opt_ctx->grad_m[i] = ggml_new_tensor(opt_ctx->ctx_momenta, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
                    opt_ctx->grad_v[i] = ggml_new_tensor(opt_ctx->ctx_momenta, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
                } else {
                    opt_ctx->grad_m[i] = nullptr;
                    opt_ctx->grad_v[i] = nullptr;
                }
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    opt_ctx->gb_grad = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads =*/ true);
    ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.data());

    if (opt_ctx->buf_static) {
        if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_GRAD) {
            return;
        }
    } else if (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_GRAD) {
        // Allocate ctx_static on CPU (last backend) — contains grad accumulators + loss tensors
        const int n_backends = ggml_backend_sched_get_n_backends(opt_ctx->backend_sched);
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, n_backends - 1));
        ggml_graph_reset(opt_ctx->gb_grad);
    }

    GGML_ASSERT(opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    opt_ctx->gb_opt = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads =*/ true);

    opt_ctx->opt_step_params = ggml_new_tensor_1d(opt_ctx->ctx_cpu, GGML_TYPE_F32, need_momenta ? 7 : 2);
    ggml_tensor * adamw_params = opt_ctx->opt_step_params;
    ggml_set_input(adamw_params);
    const char * optimizer_name = ggml_opt_optimizer_name(opt_ctx->optimizer);
    ggml_format_name(adamw_params, "%s_params", optimizer_name);
    for (int i = opt_ctx->gf->n_nodes-1; i >= 0; --i) {
        struct ggml_tensor * node = opt_ctx->gb_opt->nodes[i];
        struct ggml_tensor * grad = ggml_graph_get_grad(opt_ctx->gb_opt, node);

        if (grad && (node->flags & GGML_TENSOR_FLAG_PARAM)) {
            struct ggml_tensor * m = nullptr;
            struct ggml_tensor * v = nullptr;
            if (need_momenta) {
                m = opt_ctx->grad_m[i];
                v = opt_ctx->grad_v[i];
                ggml_format_name(m, "AdamW m for %s", node->name);
                ggml_format_name(v, "AdamW v for %s", node->name);
            }
            struct ggml_tensor * opt_step;
            switch (optimizer) {
                case GGML_OPT_OPTIMIZER_TYPE_ADAMW:
                    opt_step = ggml_opt_step_adamw(opt_ctx->ctx_compute, node, grad, m, v, adamw_params);
                    break;
                case GGML_OPT_OPTIMIZER_TYPE_SGD:
                    opt_step = ggml_opt_step_sgd(opt_ctx->ctx_compute, node, grad, adamw_params);
                    break;
                default:
                    GGML_ABORT("fatal error");
            }
            ggml_format_name(opt_step, "%s step for %s", optimizer_name, node->name);
            ggml_build_forward_expand(opt_ctx->gb_opt, opt_step);
        }
    }

    // Allocate the momenta on the backend that owns the params, so that the in-place m/v
    // updates done by GGML_OP_OPT_STEP_ADAMW land in the real tensors instead of a scratch
    // split-input copy that is thrown away after each step (see ctx_momenta above).
    if (need_momenta && !opt_ctx->buf_momenta) {
        ggml_backend_buffer_type_t buft = nullptr;
        for (int i = 0; i < opt_ctx->gf->n_nodes; ++i) {
            ggml_tensor * node = opt_ctx->gf->nodes[i];
            if ((node->flags & GGML_TENSOR_FLAG_PARAM) && node->buffer) {
                buft = ggml_backend_buffer_get_type(node->buffer);
                break;
            }
        }
        if (buft) {
            opt_ctx->buf_momenta = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_momenta, buft);
        }
        if (!opt_ctx->buf_momenta) {
            // Fall back to the last backend (CPU), matching the pre-split behavior.
            const int n_backends = ggml_backend_sched_get_n_backends(opt_ctx->backend_sched);
            opt_ctx->buf_momenta = ggml_backend_alloc_ctx_tensors(
                opt_ctx->ctx_momenta, ggml_backend_sched_get_backend(opt_ctx->backend_sched, n_backends - 1));
        }
        GGML_ASSERT(opt_ctx->buf_momenta);

        // Zero the momenta explicitly: ggml_graph_reset() below only runs when buf_static is
        // allocated in this call, which is not the case when the graph was already built for
        // GGML_OPT_BUILD_TYPE_GRAD.
        for (size_t i = 0; i < opt_ctx->grad_m.size(); ++i) {
            if (opt_ctx->grad_m[i]) {
                ggml_set_zero(opt_ctx->grad_m[i]);
            }
            if (opt_ctx->grad_v[i]) {
                ggml_set_zero(opt_ctx->grad_v[i]);
            }
        }
    }

    if (!opt_ctx->buf_static) {
        // Allocate ctx_static on CPU (last backend) — contains grad accumulators, loss tensors
        const int n_backends = ggml_backend_sched_get_n_backends(opt_ctx->backend_sched);
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, n_backends - 1));
        ggml_graph_reset(opt_ctx->gb_opt);
    }

    opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, ggml_backend_cpu_buffer_type());
}

ggml_opt_context_t ggml_opt_init(struct ggml_opt_params params) {
    ggml_opt_context_t result = new struct ggml_opt_context;
    result->backend_sched    = params.backend_sched;
    result->ctx_compute      = params.ctx_compute;
    result->build_type       = params.build_type;
    result->build_type_alloc = params.build_type;
    result->inputs           = params.inputs;

    // Multi-loss arrays take precedence; otherwise fall back to the single-head
    // scalar fields. The arrays are copied here, so the caller's may be temporary.
    if (params.n_loss > 0) {
        GGML_ASSERT(params.loss_type_multi && "n_loss > 0 requires loss_type_multi");
        const size_t n = size_t(params.n_loss);
        result->loss_type.assign(params.loss_type_multi, params.loss_type_multi + n);
        result->loss_w = params.loss_w ? std::vector<float>(params.loss_w, params.loss_w + n)
                                       : std::vector<float>(n, 1.0f);
        if (params.outputs_multi) {
            result->outputs.assign(params.outputs_multi, params.outputs_multi + n);
            for (struct ggml_tensor * out : result->outputs) {
                GGML_ASSERT(out && "outputs_multi must not contain null entries");
            }
        }
    } else {
        result->loss_type = { params.loss_type };
        result->loss_w    = { 1.0f };
        result->outputs   = params.outputs ? std::vector<struct ggml_tensor *>{ params.outputs }
                                           : std::vector<struct ggml_tensor *>{};
    }
    result->opt_period       = params.opt_period;
    result->loss_mask_ne0    = params.loss_mask_ne0;
    result->get_opt_pars     = params.get_opt_pars;
    result->get_opt_pars_ud  = params.get_opt_pars_ud;
    result->optimizer        = params.optimizer;

    GGML_ASSERT(result->opt_period >= 1);

    result->static_graphs = result->ctx_compute;

    if (!result->static_graphs) {
        GGML_ASSERT(!result->inputs);
        GGML_ASSERT(result->outputs.empty());
        return result;
    }

    GGML_ASSERT(result->inputs);
    GGML_ASSERT(result->outputs.size() == result->loss_type.size());

    result->gf = ggml_new_graph_custom(result->ctx_compute, GGML_DEFAULT_GRAPH_SIZE, /*grads =*/ true); // Forward pass.
    for (struct ggml_tensor * out : result->outputs) {
        ggml_build_forward_expand(result->gf, out);
    }

    ggml_opt_build(result);

    return result;
}

void ggml_opt_free(ggml_opt_context_t opt_ctx) {
    if (opt_ctx == nullptr) {
        return;
    }
    ggml_backend_buffer_free(opt_ctx->buf_static);
    ggml_backend_buffer_free(opt_ctx->buf_momenta);
    ggml_backend_buffer_free(opt_ctx->buf_cpu);
    ggml_free(opt_ctx->ctx_static);
    ggml_free(opt_ctx->ctx_momenta);
    ggml_free(opt_ctx->ctx_cpu);
    ggml_free(opt_ctx->ctx_copy);
    delete opt_ctx;
}

void ggml_opt_reset(ggml_opt_context_t opt_ctx, bool optimizer) {
    if (optimizer) {
        ggml_graph_reset(opt_ctx->gb_opt);
        opt_ctx->iter = 1;
    } else {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
}

bool ggml_opt_static_graphs(ggml_opt_context_t opt_ctx) {
    return opt_ctx->static_graphs;
}

struct ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx) {
    return opt_ctx->inputs;
}

// Head-0 accessors. The vectors are empty until ggml_opt_build() has run (and
// labels/pred/ncorrect stay null for loss types that do not use them), so these
// return nullptr rather than indexing an empty vector. Indexed per-head
// variants are added with the multi-loss public API.
static struct ggml_tensor * head0(const std::vector<struct ggml_tensor *> & v) {
    return v.empty() ? nullptr : v[0];
}

struct ggml_tensor * ggml_opt_outputs(ggml_opt_context_t opt_ctx) {
    return head0(opt_ctx->outputs);
}

struct ggml_tensor * ggml_opt_labels(ggml_opt_context_t opt_ctx) {
    return head0(opt_ctx->labels);
}

struct ggml_tensor * ggml_opt_loss_weights(ggml_opt_context_t opt_ctx) {
    return head0(opt_ctx->loss_weights);
}

struct ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx) {
    return opt_ctx->loss;
}

struct ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx) {
    return head0(opt_ctx->pred);
}

struct ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx) {
    return head0(opt_ctx->ncorrect);
}

// ---- per-head accessors (ggmlR multi-loss extension) ----

int64_t ggml_opt_n_loss(ggml_opt_context_t opt_ctx) {
    return int64_t(opt_ctx->loss_type.size());
}

// Bounds-checked head lookup: an out-of-range index is a caller bug, and
// silently returning head 0 would train the wrong head without any symptom.
static struct ggml_tensor * head_i(const std::vector<struct ggml_tensor *> & v, int64_t i, size_t n_loss) {
    GGML_ASSERT(i >= 0 && size_t(i) < n_loss && "loss head index out of range");
    return size_t(i) < v.size() ? v[size_t(i)] : nullptr;
}

struct ggml_tensor * ggml_opt_outputs_i(ggml_opt_context_t opt_ctx, int64_t i) {
    return head_i(opt_ctx->outputs, i, opt_ctx->loss_type.size());
}

struct ggml_tensor * ggml_opt_labels_i(ggml_opt_context_t opt_ctx, int64_t i) {
    return head_i(opt_ctx->labels, i, opt_ctx->loss_type.size());
}

struct ggml_tensor * ggml_opt_loss_weights_i(ggml_opt_context_t opt_ctx, int64_t i) {
    return head_i(opt_ctx->loss_weights, i, opt_ctx->loss_type.size());
}

struct ggml_tensor * ggml_opt_pred_i(ggml_opt_context_t opt_ctx, int64_t i) {
    return head_i(opt_ctx->pred, i, opt_ctx->loss_type.size());
}

struct ggml_tensor * ggml_opt_ncorrect_i(ggml_opt_context_t opt_ctx, int64_t i) {
    return head_i(opt_ctx->ncorrect, i, opt_ctx->loss_type.size());
}

struct ggml_tensor * ggml_opt_loss_i(ggml_opt_context_t opt_ctx, int64_t i) {
    return head_i(opt_ctx->losses, i, opt_ctx->loss_type.size());
}

void ggml_opt_set_labels_offs(ggml_opt_context_t opt_ctx, int64_t n_loss, const int64_t * labels_offs) {
    GGML_ASSERT(size_t(n_loss) == opt_ctx->loss_type.size() &&
        "n_loss must match the head count the context was initialized with");
    if (labels_offs) {
        opt_ctx->labels_offs.assign(labels_offs, labels_offs + size_t(n_loss));
    } else {
        opt_ctx->labels_offs.clear();
    }
}

struct ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, struct ggml_tensor * node) {
    return ggml_graph_get_grad_acc(opt_ctx->gb_opt, node);
}

// ====== Optimization Result ======

ggml_opt_result_t ggml_opt_result_init() {
    return new ggml_opt_result;
}

void ggml_opt_result_free(ggml_opt_result_t result) {
    delete result;
}

void ggml_opt_result_reset(ggml_opt_result_t result) {
    result->ndata = 0;
    result->loss.clear();
    result->pred.clear();
    result->ncorrect = 0;
    // Cleared, not resized: the head count is re-established on the next eval,
    // so a result object can be reused across epochs (and models).
    result->loss_head.clear();
    result->ncorrect_head.clear();
}

void ggml_opt_result_ndata(ggml_opt_result_t result, int64_t * ndata) {
    *ndata = result->ndata;
}

// Shared by ggml_opt_result_loss and its per-head variant: the scaling and
// uncertainty rules must stay identical for the total and for each head.
static void ggml_opt_result_loss_vec(
        const std::vector<float> & losses, int64_t opt_period, bool per_datapoint,
        double * loss, double * unc) {
    const int64_t nbatches = losses.size(); // Number of physical batches.

    if (nbatches == 0) {
        *loss = 0.0;
        if (unc) {
            *unc = NAN;
        }
        return;
    }

    double sum         = 0.0;
    double sum_squared = 0.0;

    for (const float & l : losses) {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        const float loss_scaled = per_datapoint ? l*opt_period : l;
        sum         += loss_scaled;
        sum_squared += loss_scaled*loss_scaled;
    }

    const double mean = sum/nbatches;
    *loss = per_datapoint ? mean : sum;

    if (!unc) {
        return;
    }

    if (nbatches < 2) {
        *unc = NAN;
        return;
    }

    const double var_sum = sum_squared/nbatches - mean*mean; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    *unc = per_datapoint ? sqrt(var_sum / (nbatches - 1)) : sqrt(var_sum * nbatches/(nbatches - 1));
}

void ggml_opt_result_loss(ggml_opt_result_t result, double * loss, double * unc) {
    ggml_opt_result_loss_vec(result->loss, result->opt_period,
                             result->loss_per_datapoint, loss, unc);
}

int64_t ggml_opt_result_n_loss(ggml_opt_result_t result) {
    return int64_t(result->loss_head.size());
}

void ggml_opt_result_loss_i(ggml_opt_result_t result, int64_t i, double * loss, double * unc) {
    GGML_ASSERT(i >= 0 && size_t(i) < result->loss_head.size() && "loss head index out of range");
    ggml_opt_result_loss_vec(result->loss_head[size_t(i)], result->opt_period,
                             result->loss_per_datapoint, loss, unc);
}

void ggml_opt_result_accuracy_i(ggml_opt_result_t result, int64_t i, double * accuracy, double * unc) {
    GGML_ASSERT(i >= 0 && size_t(i) < result->ncorrect_head.size() && "loss head index out of range");
    const int64_t ncorrect = result->ncorrect_head[size_t(i)];
    if (ncorrect < 0 || result->ndata == 0) {
        // This head has no accuracy (its loss is not cross-entropy).
        *accuracy = NAN;
        if (unc) {
            *unc = NAN;
        }
        return;
    }
    *accuracy = double(ncorrect) / double(result->ndata);
    if (unc) {
        // Same rule as ggml_opt_result_accuracy: undefined for a single datapoint.
        *unc = result->ndata >= 2 ?
            sqrt((*accuracy) * (1.0 - (*accuracy)) / double(result->ndata - 1)) : NAN;
    }
}

void ggml_opt_result_pred(ggml_opt_result_t result, int32_t * pred) {
    for (size_t i = 0; i < result->pred.size(); ++i) {
        pred[i] = result->pred[i];
    }
}

void ggml_opt_result_accuracy(ggml_opt_result_t result, double * accuracy, double * unc) {
    *accuracy = result->ncorrect >= 0 ? double(result->ncorrect) / double(result->ndata) : NAN;

    if (!unc) {
        return;
    }

    *unc = result->ncorrect >= 0 && result->ndata >= 2 ?
        sqrt((*accuracy) * (1.0 - (*accuracy)) / double(result->ndata - 1)) : NAN;
}

// ====== Computation ======

void ggml_opt_prepare_alloc(
        ggml_opt_context_t    opt_ctx,
        struct ggml_context * ctx_compute,
        struct ggml_cgraph  * gf,
        struct ggml_tensor  * inputs,
        struct ggml_tensor  * outputs) {
    GGML_ASSERT(!opt_ctx->static_graphs);
    opt_ctx->ctx_compute = ctx_compute;
    opt_ctx->gf          = gf;
    opt_ctx->inputs      = inputs;
    opt_ctx->outputs     = { outputs };
}

void ggml_opt_prepare_alloc_multi(
        ggml_opt_context_t             opt_ctx,
        struct ggml_context          * ctx_compute,
        struct ggml_cgraph           * gf,
        struct ggml_tensor           * inputs,
        int64_t                        n_loss,
        struct ggml_tensor   * const * outputs) {
    GGML_ASSERT(!opt_ctx->static_graphs);
    GGML_ASSERT(n_loss >= 1 && outputs);
    GGML_ASSERT(size_t(n_loss) == opt_ctx->loss_type.size() &&
        "n_loss must match the head count the context was initialized with");
    opt_ctx->ctx_compute = ctx_compute;
    opt_ctx->gf          = gf;
    opt_ctx->inputs      = inputs;
    opt_ctx->outputs.assign(outputs, outputs + size_t(n_loss));
    for (struct ggml_tensor * out : opt_ctx->outputs) {
        GGML_ASSERT(out && "outputs must not contain null entries");
    }
}

void ggml_opt_alloc(ggml_opt_context_t opt_ctx, bool backward) {
    GGML_ASSERT(!opt_ctx->eval_ready);
    if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period > 1 && opt_ctx->opt_i == 0) {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
    if (backward) {
        const int32_t opt_i_next = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
        opt_ctx->build_type = opt_i_next == 0 ? GGML_OPT_BUILD_TYPE_OPT : GGML_OPT_BUILD_TYPE_GRAD;
    } else {
        opt_ctx->build_type = GGML_OPT_BUILD_TYPE_FORWARD;
    }

    if (!opt_ctx->static_graphs) {
        ggml_opt_build(opt_ctx);
    }

    struct ggml_cgraph * graph = nullptr;
    switch (opt_ctx->build_type) {
        case GGML_OPT_BUILD_TYPE_FORWARD: {
            graph = opt_ctx->gf;
        } break;
        case GGML_OPT_BUILD_TYPE_GRAD: {
            graph = opt_ctx->gb_grad;
        } break;
        case GGML_OPT_BUILD_TYPE_OPT: {
            graph = opt_ctx->gb_opt;
        } break;
    }
    GGML_ASSERT(graph);

    if (opt_ctx->allocated_graph == graph) {
        opt_ctx->eval_ready = true;
        return;
    }

    ggml_backend_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph

    if (opt_ctx->static_graphs) {
        ggml_init_params params = {
            /*.mem_size   =*/ graph->size*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph->size, graph->grads),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_free(opt_ctx->ctx_copy);
        opt_ctx->ctx_copy = ggml_init(params);

        opt_ctx->allocated_graph_copy = dup_graph(opt_ctx->ctx_copy, graph);
    } else {
        opt_ctx->allocated_graph_copy = graph;
    }

    ggml_backend_sched_alloc_graph(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->allocated_graph = graph;

    opt_ctx->eval_ready = true;
}

void ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_result_t result) {
    GGML_ASSERT(opt_ctx->eval_ready);
    if (opt_ctx->allocated_graph == opt_ctx->gb_opt) {
        const ggml_opt_optimizer_params & opt_pars = opt_ctx->get_opt_pars(opt_ctx->get_opt_pars_ud);

        switch (opt_ctx->optimizer) {
            case GGML_OPT_OPTIMIZER_TYPE_ADAMW: {
                GGML_ASSERT(opt_pars.adamw.alpha > 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.eps >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd <= 1.0f);

                // beta1, beta2 after applying warmup
                const float beta1h = 1.0f / (1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
                const float beta2h = 1.0f / (1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));

                float * adamw_par_data = ggml_get_data_f32(opt_ctx->opt_step_params);
                adamw_par_data[0] = opt_pars.adamw.alpha;
                adamw_par_data[1] = opt_pars.adamw.beta1;
                adamw_par_data[2] = opt_pars.adamw.beta2;
                adamw_par_data[3] = opt_pars.adamw.eps;
                adamw_par_data[4] = opt_pars.adamw.wd;
                adamw_par_data[5] = beta1h;
                adamw_par_data[6] = beta2h;
            } break;
            case GGML_OPT_OPTIMIZER_TYPE_SGD: {
                GGML_ASSERT(opt_pars.sgd.alpha > 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd >= 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd <= 1.0f);
                float * sgd = ggml_get_data_f32(opt_ctx->opt_step_params);
                sgd[0] = opt_pars.sgd.alpha;
                sgd[1] = opt_pars.sgd.wd;
            } break;
            default:
                GGML_ABORT("fatal error");
        }
    }

    ggml_backend_sched_graph_compute(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->iter += opt_ctx->allocated_graph == opt_ctx->gb_opt;
    opt_ctx->opt_i = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;

    if (!opt_ctx->static_graphs) {
        opt_ctx->gf                   = nullptr;
        opt_ctx->gb_grad              = nullptr;
        opt_ctx->gb_opt               = nullptr;
        opt_ctx->allocated_graph      = nullptr;
        opt_ctx->allocated_graph_copy = nullptr;
    }

    opt_ctx->eval_ready = false;

    if (!result) {
        return;
    }

    if (result->ndata == 0) {
        result->loss_per_datapoint = opt_ctx->loss_per_datapoint;
        result->opt_period         = opt_ctx->opt_period;
    } else {
        GGML_ASSERT(result->loss_per_datapoint == opt_ctx->loss_per_datapoint);
        GGML_ASSERT(result->opt_period         == opt_ctx->opt_period);
    }

    // Batch size is shared by all heads (they see the same datapoints).
    const int64_t ndata = ggml_opt_batch_size(opt_ctx->outputs[0]);
    GGML_ASSERT(result->ndata == ndata*int64_t(result->loss.size()) && "varying batch size not supported");
    result->ndata += ndata;

    GGML_ASSERT(ggml_is_scalar(opt_ctx->loss));
    GGML_ASSERT(opt_ctx->loss->type == GGML_TYPE_F32);
    float loss;
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
    result->loss.push_back(loss);

    // Per-head loss and accuracy. The scalar fields below stay head 0, so
    // single-head callers see exactly what they saw before.
    const size_t n_loss = opt_ctx->loss_type.size();
    if (result->loss_head.empty()) {
        result->loss_head.resize(n_loss);
        result->ncorrect_head.assign(n_loss, 0);
    }
    GGML_ASSERT(result->loss_head.size() == n_loss && "result reused across models with different head counts");

    for (size_t i = 0; i < n_loss; ++i) {
        struct ggml_tensor * loss_i = opt_ctx->losses[i];
        if (loss_i) {
            GGML_ASSERT(ggml_is_scalar(loss_i));
            GGML_ASSERT(loss_i->type == GGML_TYPE_F32);
            float lv;
            ggml_backend_tensor_get(loss_i, &lv, 0, ggml_nbytes(loss_i));
            result->loss_head[i].push_back(lv);
        }

        struct ggml_tensor * nc_i = i < opt_ctx->ncorrect.size() ? opt_ctx->ncorrect[i] : nullptr;
        if (!nc_i || result->ncorrect_head[i] < 0) {
            result->ncorrect_head[i] = -1; // no accuracy for this head (non-CE loss)
            continue;
        }
        GGML_ASSERT(ggml_is_scalar(nc_i));
        GGML_ASSERT(nc_i->type == GGML_TYPE_I64);
        int64_t nc;
        ggml_backend_tensor_get(nc_i, &nc, 0, ggml_nbytes(nc_i));
        result->ncorrect_head[i] += nc;
    }

    // ggml_opt_result carries one pred/ncorrect; report head 0 (per-head
    // values are in loss_head/ncorrect_head above).
    struct ggml_tensor * pred_t     = head0(opt_ctx->pred);
    struct ggml_tensor * ncorrect_t = head0(opt_ctx->ncorrect);

    if (pred_t) {
        GGML_ASSERT(pred_t->type == GGML_TYPE_I32);
        std::vector<int32_t> pred(ndata);
        ggml_backend_tensor_get(pred_t, pred.data(), 0, ggml_nbytes(pred_t));
        result->pred.insert(result->pred.end(), pred.begin(), pred.end());
    }

    if (!ncorrect_t || result->ncorrect < 0) {
        result->ncorrect = -1;
        return;
    }

    GGML_ASSERT(ggml_is_scalar(ncorrect_t));
    GGML_ASSERT(ncorrect_t->type == GGML_TYPE_I64);
    int64_t ncorrect;
    ggml_backend_tensor_get(ncorrect_t, &ncorrect, 0, ggml_nbytes(ncorrect_t));
    result->ncorrect += ncorrect;
}

// ====== High-Level Functions ======

// Copy one batch into the context's input and per-head label tensors.
// Single-head contexts take the original ggml_opt_dataset_get_batch path so
// their behaviour (and the dataset asserts they rely on) stays unchanged.
static void ggml_opt_feed_batch(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t ibatch) {
    struct ggml_tensor * inputs = ggml_opt_inputs(opt_ctx);
    const size_t n_loss = opt_ctx->loss_type.size();

    if (n_loss == 1) {
        ggml_opt_dataset_get_batch(dataset, inputs, opt_ctx->labels[0], ibatch);
    } else {
        // Data is copied once, with the first head; the remaining heads only
        // pull their own slice of the concatenated labels.
        for (size_t i = 0; i < n_loss; ++i) {
            struct ggml_tensor * labels_i = opt_ctx->labels[i];
            const int64_t off = i < opt_ctx->labels_offs.size() ? opt_ctx->labels_offs[i] : 0;
            if (!labels_i) {
                // Loss types without labels (MEAN/SUM) still need the inputs copied.
                if (i == 0) {
                    ggml_opt_dataset_get_batch(dataset, inputs, nullptr, ibatch);
                }
                continue;
            }
            ggml_opt_dataset_get_batch_head(dataset, i == 0 ? inputs : nullptr, labels_i, off, ibatch);
        }
    }

    for (size_t i = 0; i < n_loss; ++i) {
        struct ggml_tensor * weights_i = i < opt_ctx->loss_weights.size() ? opt_ctx->loss_weights[i] : nullptr;
        if (weights_i) { // non-null only for weighted MSE
            ggml_opt_dataset_get_batch_weights(dataset, weights_i, ibatch);
        }
    }
}

void ggml_opt_epoch(
        ggml_opt_context_t      opt_ctx,
        ggml_opt_dataset_t      dataset,
        ggml_opt_result_t       result_train,
        ggml_opt_result_t       result_eval,
        int64_t                 idata_split,
        ggml_opt_epoch_callback callback_train,
        ggml_opt_epoch_callback callback_eval) {
    GGML_ASSERT(ggml_opt_static_graphs(opt_ctx) && "ggml_opt_epoch requires static graphs");
    struct ggml_tensor * inputs  = ggml_opt_inputs(opt_ctx);
    struct ggml_tensor * data    = ggml_opt_dataset_data(dataset);
    // Dataset is always 2D [ne_datapoint, ndata], inputs may be N-D with batch in last dim.
    // Verify that per-sample element counts match.
    GGML_ASSERT(data->ne[0] == ggml_opt_ne_per_sample(inputs));

    const int64_t ndata       = data->ne[1];
    const int64_t ndata_batch = ggml_opt_batch_size(inputs);

    GGML_ASSERT(ndata % ndata_batch == 0);
    const int64_t nbatches = ndata/ndata_batch;

    idata_split = idata_split < 0 ? ndata : idata_split;
    GGML_ASSERT(idata_split % ndata_batch == 0);
    const int64_t ibatch_split = idata_split / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_start = ggml_time_us();
    for (; ibatch < ibatch_split; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ true);
        ggml_opt_feed_batch(opt_ctx, dataset, ibatch);
        ggml_opt_eval(opt_ctx, result_train);
        if (callback_train) {
            callback_train(true, opt_ctx, dataset, result_train, ibatch+1, ibatch_split, t_loop_start);
        }
    }
    t_loop_start = ggml_time_us();
    for (; ibatch < nbatches; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ false);
        ggml_opt_feed_batch(opt_ctx, dataset, ibatch);
        ggml_opt_eval(opt_ctx, result_eval);
        if (callback_eval) {
            callback_eval(false, opt_ctx, dataset, result_eval, ibatch+1-ibatch_split, nbatches-ibatch_split, t_loop_start);
        }
    }
}

void ggml_opt_epoch_callback_progress_bar(
        bool               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    GGML_OPT_LOG("%s[", train ? "train: " : "val:   ");

    // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
    constexpr int64_t bar_length = 8;
    const int64_t ibatch8 = 8 * ibatch;
    for (int64_t j = 0; j < bar_length; ++j) {
        if        (ibatch_max * (8*j + 8) / bar_length < ibatch8) {
            GGML_OPT_LOG("\u2588"); // full block
        } else if (ibatch_max * (8*j + 7) / bar_length < ibatch8) {
            GGML_OPT_LOG("\u2589"); // 7/8 filled
        } else if (ibatch_max * (8*j + 6) / bar_length < ibatch8) {
            GGML_OPT_LOG("\u258A"); // 6/8 filled
        } else if (ibatch_max * (8*j + 5) / bar_length < ibatch8) {
            GGML_OPT_LOG("\u258B"); // 5/8 filled
        } else if (ibatch_max * (8*j + 4) / bar_length < ibatch8) {
            GGML_OPT_LOG("\u258C"); // 4/8 filled
        } else if (ibatch_max * (8*j + 3) / bar_length < ibatch8) {
            GGML_OPT_LOG("\u258D"); // 3/8 filled
        } else if (ibatch_max * (8*j + 2) / bar_length < ibatch8) {
            GGML_OPT_LOG("\u258E"); // 2/8 filled
        } else if (ibatch_max * (8*j + 1) / bar_length < ibatch8) {
            GGML_OPT_LOG("\u258F"); // 1/8 filled
        } else {
            GGML_OPT_LOG(" ");
        }
    }

    const int64_t batch_size = ggml_opt_batch_size(ggml_opt_inputs(opt_ctx));
    const int64_t idata      = ibatch*batch_size;
    const int64_t idata_max  = ibatch_max*batch_size;

    double loss;
    double loss_unc;
    ggml_opt_result_loss(result, &loss, &loss_unc);

    double accuracy;
    double accuracy_unc;
    ggml_opt_result_accuracy(result, &accuracy, &accuracy_unc);

    const int64_t t_ibatch_us = ggml_time_us() - t_start_us;
    int64_t t_ibatch_s = t_ibatch_us / 1000000;
    const int64_t t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    const int64_t t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    const int64_t t_eta_us = t_ibatch_us * (ibatch_max - ibatch)/ibatch;
    int64_t t_eta_s = t_eta_us / 1000000;
    const int64_t t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    const int64_t t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    GGML_OPT_LOG("] data=%07" PRId64 "/%07" PRId64 " loss=%.5lf±%.5lf acc=%.2lf±%.2lf%% "
            "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
            idata, idata_max, loss, loss_unc, 100.0*accuracy, 100.0*accuracy_unc,
            t_ibatch_h, t_ibatch_m, t_ibatch_s, t_eta_h, t_eta_m, t_eta_s);
    if (ibatch == ibatch_max) {
        GGML_OPT_LOG("\n");
    }
    GGML_OPT_FFLUSH();

    GGML_UNUSED(dataset);
}

void ggml_opt_fit(
        ggml_backend_sched_t            backend_sched,
        ggml_context                  * ctx_compute,
        ggml_tensor                   * inputs,
        ggml_tensor                   * outputs,
        ggml_opt_dataset_t              dataset,
        enum ggml_opt_loss_type         loss_type,
        enum ggml_opt_optimizer_type    optimizer,
        ggml_opt_get_optimizer_params   get_opt_pars,
        int64_t                         nepoch,
        int64_t                         nbatch_logical,
        float                           val_split,
        bool                            silent) {
    ggml_time_init();
    const int64_t t_start_us = ggml_time_us();

    const int64_t ndata           = ggml_opt_dataset_data(dataset)->ne[1];
    const int64_t nbatch_physical = ggml_opt_batch_size(inputs);
    GGML_ASSERT(ndata          % nbatch_logical  == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    const int64_t opt_period       = nbatch_logical / nbatch_physical;
    const int64_t nbatches_logical = ndata / nbatch_logical;

    GGML_ASSERT(val_split >= 0.0f);
    GGML_ASSERT(val_split <  1.0f);
    const int64_t ibatch_split = int64_t(((1.0f - val_split) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    const int64_t idata_split  = ibatch_split * nbatch_physical;

    int64_t epoch = 1;

    ggml_opt_params params = ggml_opt_default_params(backend_sched, loss_type);
    params.ctx_compute     = ctx_compute;
    params.inputs          = inputs;
    params.outputs         = outputs;
    params.opt_period      = opt_period;
    params.get_opt_pars    = get_opt_pars;
    params.get_opt_pars_ud = &epoch;
    params.optimizer       = optimizer;
    ggml_opt_context_t opt_ctx = ggml_opt_init(params);

    // Shuffling the data is generally useful but there is only a point if not all data is used in a single batch.
    if (nbatch_logical < ndata) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_val   = ggml_opt_result_init();

    ggml_opt_epoch_callback epoch_callback = silent ? nullptr : ggml_opt_epoch_callback_progress_bar;

    for (; epoch <= nepoch; ++epoch) {
        if (nbatch_logical < idata_split) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split);
        }

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_val);

        if (!silent) {
            GGML_OPT_LOG("%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        ggml_opt_epoch(opt_ctx, dataset, result_train, result_val, idata_split, epoch_callback, epoch_callback);
        if (!silent) {
            GGML_OPT_LOG("\n");
        }
    }

    if (!silent) {
        int64_t t_total_s = (ggml_time_us() - t_start_us) / 1000000;
        const int64_t t_total_h = t_total_s / 3600;
        t_total_s -= t_total_h * 3600;
        const int64_t t_total_m = t_total_s / 60;
        t_total_s -= t_total_m * 60;
        GGML_OPT_LOG("%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, t_total_h, t_total_m, t_total_s);
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_val);
}

void ggml_opt_fit_multi(
        ggml_backend_sched_t            backend_sched,
        ggml_context                  * ctx_compute,
        ggml_tensor                   * inputs,
        int64_t                         n_loss,
        ggml_tensor           * const * outputs,
        const enum ggml_opt_loss_type * loss_type,
        const float                   * loss_w,
        const int64_t                 * labels_offs,
        ggml_opt_dataset_t              dataset,
        enum ggml_opt_optimizer_type    optimizer,
        ggml_opt_get_optimizer_params   get_opt_pars,
        int64_t                         nepoch,
        int64_t                         nbatch_logical,
        float                           val_split,
        bool                            silent) {
    ggml_time_init();
    const int64_t t_start_us = ggml_time_us();

    GGML_ASSERT(n_loss >= 1 && outputs && loss_type);

    const int64_t ndata           = ggml_opt_dataset_data(dataset)->ne[1];
    const int64_t nbatch_physical = ggml_opt_batch_size(inputs);
    GGML_ASSERT(ndata          % nbatch_logical  == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    // All heads share the datapoints, so they must agree on the batch size.
    for (int64_t i = 0; i < n_loss; ++i) {
        GGML_ASSERT(outputs[i] && "outputs must not contain null entries");
        GGML_ASSERT(ggml_opt_batch_size(outputs[i]) == nbatch_physical &&
            "all loss heads must share the same batch size");
    }

    const int64_t opt_period       = nbatch_logical / nbatch_physical;
    const int64_t nbatches_logical = ndata / nbatch_logical;

    GGML_ASSERT(val_split >= 0.0f);
    GGML_ASSERT(val_split <  1.0f);
    const int64_t ibatch_split = int64_t(((1.0f - val_split) * nbatches_logical)) * opt_period;
    const int64_t idata_split  = ibatch_split * nbatch_physical;

    int64_t epoch = 1;

    ggml_opt_params params = ggml_opt_default_params_multi(backend_sched, n_loss, loss_type, loss_w);
    params.ctx_compute     = ctx_compute;
    params.inputs          = inputs;
    params.outputs_multi   = outputs;
    params.opt_period      = opt_period;
    params.get_opt_pars    = get_opt_pars;
    params.get_opt_pars_ud = &epoch;
    params.optimizer       = optimizer;
    ggml_opt_context_t opt_ctx = ggml_opt_init(params);

    ggml_opt_set_labels_offs(opt_ctx, n_loss, labels_offs);

    // Shuffling the data is generally useful but there is only a point if not all data is used in a single batch.
    if (nbatch_logical < ndata) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_val   = ggml_opt_result_init();

    ggml_opt_epoch_callback epoch_callback = silent ? nullptr : ggml_opt_epoch_callback_progress_bar;

    for (; epoch <= nepoch; ++epoch) {
        if (nbatch_logical < idata_split) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split);
        }

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_val);

        if (!silent) {
            GGML_OPT_LOG("%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        ggml_opt_epoch(opt_ctx, dataset, result_train, result_val, idata_split, epoch_callback, epoch_callback);
        if (!silent) {
            GGML_OPT_LOG("\n");
        }
    }

    if (!silent) {
        int64_t t_total_s = (ggml_time_us() - t_start_us) / 1000000;
        const int64_t t_total_h = t_total_s / 3600;
        t_total_s -= t_total_h * 3600;
        const int64_t t_total_m = t_total_s / 60;
        t_total_s -= t_total_m * 60;
        GGML_OPT_LOG("%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, t_total_h, t_total_m, t_total_s);
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_val);
}

enum ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t c) {
    return c->optimizer;
}

GGML_API const char * ggml_opt_optimizer_name(enum ggml_opt_optimizer_type o) {
    switch (o) {
        case GGML_OPT_OPTIMIZER_TYPE_ADAMW:
            return "adamw";
        case GGML_OPT_OPTIMIZER_TYPE_SGD:
            return "sgd";
        default:
            return "undefined";
    };
}
