// Sum / min segment tree for prioritized experience replay (Schaul et al. 2016),
// proportional variant. Backs ag_per_sampler() in R/ag_per.R.
//
// Leaves hold p_i^alpha, where p_i (the "raw" priority, |td| + eps) is kept
// separately so that a change of alpha rebuilds the tree in one O(capacity)
// pass. The sum tree gives proportional sampling in O(log N); the min tree
// gives the smallest leaf, which normalises importance weights over the whole
// buffer: max_i w_i = (N * p_min / total)^-beta.
//
// A slot that was never set has leaf 0 in the sum tree and +Inf in the min
// tree, so it is never drawn and never the minimum. Random numbers come from R
// (runif in the caller), so set.seed() reproduces a draw.

#include <R.h>
#include <Rinternals.h>
#include <math.h>

typedef struct {
    int     cap;       // slots
    int     P;         // leaves: smallest power of two >= cap
    int     n_used;    // slots with a priority
    double  alpha;
    double  max_raw;   // largest raw priority seen (new slots get it)
    double *sum;       // [2P], node 1 is the root, leaves at P..P+cap-1
    double *mn;        // [2P]
    double *raw;       // [cap]
    char   *used;      // [cap]
} per_tree;

static void per_finalize(SEXP ptr) {
    per_tree *t = (per_tree *) R_ExternalPtrAddr(ptr);
    if (!t) return;
    R_Free(t->sum); R_Free(t->mn); R_Free(t->raw); R_Free(t->used);
    R_Free(t);
    R_ClearExternalPtr(ptr);
}

static per_tree *per_get(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP) Rf_error("per tree: not an external pointer");
    per_tree *t = (per_tree *) R_ExternalPtrAddr(ptr);
    if (!t) Rf_error("per tree: freed or not restored after a session restart");
    return t;
}

static inline double per_leaf(const per_tree *t, int i) {
    return t->used[i] ? pow(t->raw[i], t->alpha) : 0.0;
}

static inline void per_pull(per_tree *t, int n) {
    t->sum[n] = t->sum[2 * n] + t->sum[2 * n + 1];
    t->mn[n]  = fmin(t->mn[2 * n], t->mn[2 * n + 1]);
}

static void per_set_leaf(per_tree *t, int i) {
    int n = t->P + i;
    double v = per_leaf(t, i);
    t->sum[n] = v;
    t->mn[n]  = t->used[i] ? v : R_PosInf;
    for (n >>= 1; n >= 1; n >>= 1) per_pull(t, n);
}

static void per_rebuild(per_tree *t) {
    for (int i = 0; i < t->P; i++) {
        int n = t->P + i;
        if (i < t->cap) {
            double v = per_leaf(t, i);
            t->sum[n] = v;
            t->mn[n]  = t->used[i] ? v : R_PosInf;
        } else {
            t->sum[n] = 0.0;
            t->mn[n]  = R_PosInf;
        }
    }
    for (int n = t->P - 1; n >= 1; n--) per_pull(t, n);
}

SEXP R_per_new(SEXP capacity_, SEXP alpha_) {
    int cap = Rf_asInteger(capacity_);
    double alpha = Rf_asReal(alpha_);
    if (cap == NA_INTEGER || cap < 1) Rf_error("per tree: capacity must be a positive integer");
    if (cap > (1 << 29)) Rf_error("per tree: capacity too large");
    if (!R_FINITE(alpha) || alpha < 0) Rf_error("per tree: alpha must be >= 0");
    int P = 1;
    while (P < cap) P <<= 1;
    per_tree *t = R_Calloc(1, per_tree);
    t->cap = cap; t->P = P; t->n_used = 0; t->alpha = alpha; t->max_raw = 1.0;
    t->sum  = R_Calloc(2 * (size_t) P, double);
    t->mn   = R_Calloc(2 * (size_t) P, double);
    t->raw  = R_Calloc((size_t) cap, double);
    t->used = R_Calloc((size_t) cap, char);
    per_rebuild(t);
    SEXP ptr = PROTECT(R_MakeExternalPtr(t, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, per_finalize, TRUE);
    UNPROTECT(1);
    return ptr;
}

// Set raw priorities of slots idx (0-based); raw recycled when length 1.
// A raw value of NA clears the slot (it is no longer sampled).
SEXP R_per_set(SEXP ptr, SEXP idx_, SEXP raw_) {
    per_tree *t = per_get(ptr);
    R_xlen_t n = XLENGTH(idx_), nr = XLENGTH(raw_);
    if (TYPEOF(idx_) != INTSXP || TYPEOF(raw_) != REALSXP)
        Rf_error("per tree: idx must be integer, raw double");
    if (nr != n && nr != 1) Rf_error("per tree: raw must have length 1 or length(idx)");
    const int *idx = INTEGER(idx_);
    const double *raw = REAL(raw_);
    for (R_xlen_t k = 0; k < n; k++) {
        int i = idx[k];
        if (i == NA_INTEGER || i < 0 || i >= t->cap) Rf_error("per tree: slot out of range");
        double r = raw[nr == 1 ? 0 : k];
        if (ISNAN(r)) {
            if (t->used[i]) t->n_used--;
            t->used[i] = 0; t->raw[i] = 0.0;
        } else {
            if (!R_FINITE(r) || r <= 0) Rf_error("per tree: priorities must be positive and finite");
            if (!t->used[i]) t->n_used++;
            t->used[i] = 1; t->raw[i] = r;
            if (r > t->max_raw) t->max_raw = r;
        }
        per_set_leaf(t, i);
    }
    return R_NilValue;
}

// Stratified proportional draw: sample k falls in segment k of the total mass,
// at offset u[k] (in [0, 1)) inside it. Returns 0-based slots.
SEXP R_per_sample(SEXP ptr, SEXP u_) {
    per_tree *t = per_get(ptr);
    if (TYPEOF(u_) != REALSXP) Rf_error("per tree: u must be double");
    R_xlen_t B = XLENGTH(u_);
    const double *u = REAL(u_);
    double total = t->sum[1];
    if (!(total > 0)) Rf_error("per tree: sampling from an empty sampler");
    SEXP out = PROTECT(Rf_allocVector(INTSXP, B));
    int *o = INTEGER(out);
    double seg = total / (double) B;
    for (R_xlen_t k = 0; k < B; k++) {
        double m = ((double) k + u[k]) * seg;
        if (m >= total) m = nextafter(total, 0.0);
        if (m < 0) m = 0;
        int n = 1;
        while (n < t->P) {
            int l = 2 * n;
            if (m < t->sum[l]) n = l;
            else { m -= t->sum[l]; n = l + 1; }
        }
        int i = n - t->P;
        // rounding can land on an empty leaf: step to the nearest used one
        if (i >= t->cap || !t->used[i]) {
            int j = i < t->cap ? i : t->cap - 1;
            while (j >= 0 && !t->used[j]) j--;
            if (j < 0) { j = i; while (j < t->cap && !t->used[j]) j++; }
            i = j;
        }
        o[k] = i;
    }
    UNPROTECT(1);
    return out;
}

// Leaf values p^alpha of slots idx.
SEXP R_per_leaf(SEXP ptr, SEXP idx_) {
    per_tree *t = per_get(ptr);
    if (TYPEOF(idx_) != INTSXP) Rf_error("per tree: idx must be integer");
    R_xlen_t n = XLENGTH(idx_);
    SEXP out = PROTECT(Rf_allocVector(REALSXP, n));
    for (R_xlen_t k = 0; k < n; k++) {
        int i = INTEGER(idx_)[k];
        if (i == NA_INTEGER || i < 0 || i >= t->cap) Rf_error("per tree: slot out of range");
        REAL(out)[k] = t->sum[t->P + i];
    }
    UNPROTECT(1);
    return out;
}

// c(total, min leaf, max raw, n_used, alpha, capacity)
SEXP R_per_stats(SEXP ptr) {
    per_tree *t = per_get(ptr);
    SEXP out = PROTECT(Rf_allocVector(REALSXP, 6));
    REAL(out)[0] = t->sum[1];
    REAL(out)[1] = t->mn[1];
    REAL(out)[2] = t->max_raw;
    REAL(out)[3] = (double) t->n_used;
    REAL(out)[4] = t->alpha;
    REAL(out)[5] = (double) t->cap;
    UNPROTECT(1);
    return out;
}

// Raw priorities of all slots, NA where unset (for state / save).
SEXP R_per_raw(SEXP ptr) {
    per_tree *t = per_get(ptr);
    SEXP out = PROTECT(Rf_allocVector(REALSXP, t->cap));
    for (int i = 0; i < t->cap; i++) REAL(out)[i] = t->used[i] ? t->raw[i] : NA_REAL;
    UNPROTECT(1);
    return out;
}

SEXP R_per_set_alpha(SEXP ptr, SEXP alpha_) {
    per_tree *t = per_get(ptr);
    double alpha = Rf_asReal(alpha_);
    if (!R_FINITE(alpha) || alpha < 0) Rf_error("per tree: alpha must be >= 0");
    t->alpha = alpha;
    per_rebuild(t);
    return R_NilValue;
}

SEXP R_per_set_max(SEXP ptr, SEXP max_) {
    per_tree *t = per_get(ptr);
    double m = Rf_asReal(max_);
    if (!R_FINITE(m) || m <= 0) Rf_error("per tree: max priority must be positive");
    t->max_raw = m;
    return R_NilValue;
}
