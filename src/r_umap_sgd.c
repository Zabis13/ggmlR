// CPU UMAP-SGD layout optimisation — the single-threaded reference, in C.
//
// This is a straight, sequential transcription of .ggmlr_umap_sgd() in
// R/sc_umap.R: one epoch = one pass over the positive edges, each edge attracts
// its endpoints and repels n_neg random negatives, updates applied in place. It
// exists purely for speed: the R version is an interpreted double loop over
// ~n_epochs * ne * n_neg iterations (tens of millions), which is why RunUMAP's
// CPU-SGD phase took ~700 s where uwot's compiled C++ takes seconds. This file
// closes that gap while keeping the exact numerics and RNG, so the embedding is
// identical to the R reference (best single-threaded quality) — unlike the
// Hogwild GPU shader, which trades quality for parallelism.
//
// Determinism: same schedule and same PCG-hash negative sampling as both the R
// reference and vulkan-shaders/umap_sgd.comp, so all three agree. Arithmetic is
// done in double (matching R's reference), which the C uint32 RNG reproduces
// natively (no 16-bit-half juggling the R code needs to dodge NA).

#include <R.h>
#include <Rinternals.h>
#include <math.h>
#include <stdint.h>

// PCG hash -> uint32, identical to pcg_hash() in umap_sgd.comp and
// .ggmlr_pcg_hash() in R. Native uint32 wraparound gives the mod-2^32 arithmetic
// for free.
static inline uint32_t pcg_hash(uint32_t x) {
    uint32_t state = x * 747796405u + 2891336453u;
    uint32_t word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

static inline double clampg(double g) {
    return g < -4.0 ? -4.0 : (g > 4.0 ? 4.0 : g);
}

// Y:      n*2 doubles, [x0,y0,x1,y1,...], updated IN PLACE (caller copies first).
// from/to: ne edge endpoints, 0-based (as stored in the graph).
// w:      ne fuzzy weights, already normalised so max(w) = 1.
// Mirrors the R reference line-for-line; see R/sc_umap.R:.ggmlr_umap_sgd.
static void umap_sgd_run(double *Y, int n, const int *from, const int *to,
                         const double *w, int ne, int n_epochs, int n_neg,
                         double a, double b, double alpha0, double gamma,
                         uint32_t base_seed) {
    const double eps = 1e-3;
    const uint32_t golden = 2654435769u;   // 0x9e3779b9

    for (int epoch = 1; epoch <= n_epochs; ++epoch) {
        const double alpha = alpha0 * (1.0 - (double)(epoch - 1) / (double)n_epochs);
        const uint32_t epoch_seed = base_seed + (uint32_t)(epoch - 1);

        for (int e = 0; e < ne; ++e) {
            // weighted edge sampling: fire only when the reference cursor comes
            // due. Form (epoch-1)*w as its own product (not epoch*w - w) to match
            // the R reference / shader exactly once rounding kicks in.
            const double we = w[e];
            if (floor((double)epoch * we) <= floor((double)(epoch - 1) * we)) {
                continue;
            }

            const int i = from[e];
            const int j = to[e];

            // ---- attraction ----
            double yix = Y[2*i], yiy = Y[2*i + 1];
            double yjx = Y[2*j], yjy = Y[2*j + 1];
            double dx = yix - yjx, dy = yiy - yjy;
            double d2 = dx*dx + dy*dy;

            if (d2 > 0.0) {
                // coef = -2ab d2^(b-1) / (1 + a d2^b), as the R reference writes it
                double coef = (-2.0 * a * b * pow(d2, b - 1.0)) / (1.0 + a * pow(d2, b));
                double gx = clampg(coef * dx) * alpha;
                double gy = clampg(coef * dy) * alpha;
                Y[2*i]     = yix + gx;  Y[2*i + 1] = yiy + gy;
                Y[2*j]     = yjx - gx;  Y[2*j + 1] = yjy - gy;
            }

            // ---- repulsion against n_neg random negatives ----
            uint32_t rng = epoch_seed ^ (uint32_t)((uint32_t)e * golden);
            for (int k = 0; k < n_neg; ++k) {
                rng = pcg_hash(rng);
                int c = (int)(rng % (uint32_t)n);
                if (c == i) continue;

                // re-read i and c (freshest coords), mirroring Y[i,] - Y[c,] in R
                double yixr = Y[2*i], yiyr = Y[2*i + 1];
                double ycx  = Y[2*c], ycy  = Y[2*c + 1];
                double rdx = yixr - ycx, rdy = yiyr - ycy;
                double rd2 = rdx*rdx + rdy*rdy;

                double gx, gy;
                if (rd2 > 0.0) {
                    double coef = (2.0 * gamma * b) / ((eps + rd2) * (1.0 + a * pow(rd2, b)));
                    gx = clampg(coef * rdx) * alpha;
                    gy = clampg(coef * rdy) * alpha;
                } else {
                    gx = 4.0 * alpha;
                    gy = 4.0 * alpha;
                }
                Y[2*i] = yixr + gx;  Y[2*i + 1] = yiyr + gy;
            }
        }
    }
}

// R entry point. embedding: n*2 doubles [x0,y0,...]; from/to: ne 0-based ints;
// weights: ne doubles (normalised max=1). Returns the optimised n*2 vector.
SEXP R_umap_sgd_cpu(SEXP embedding, SEXP from_, SEXP to_, SEXP weights_,
                    SEXP n_, SEXP ne_, SEXP n_epochs_, SEXP n_neg_,
                    SEXP a_, SEXP b_, SEXP alpha0_, SEXP gamma_, SEXP seed_) {
    const int n        = asInteger(n_);
    const int ne       = asInteger(ne_);
    const int n_epochs = asInteger(n_epochs_);
    const int n_neg    = asInteger(n_neg_);
    const double a      = asReal(a_);
    const double b      = asReal(b_);
    const double alpha0 = asReal(alpha0_);
    const double gamma  = asReal(gamma_);
    const uint32_t seed = (uint32_t)asInteger(seed_);

    if ((R_xlen_t)XLENGTH(embedding) != (R_xlen_t)n * 2)
        error("R_umap_sgd_cpu: embedding length != n*2");
    if ((R_xlen_t)XLENGTH(from_) != (R_xlen_t)ne ||
        (R_xlen_t)XLENGTH(to_)   != (R_xlen_t)ne)
        error("R_umap_sgd_cpu: from/to length != ne");
    if ((R_xlen_t)XLENGTH(weights_) != (R_xlen_t)ne)
        error("R_umap_sgd_cpu: weights length != ne");

    // copy the embedding so the input SEXP is not mutated in place
    SEXP out = PROTECT(allocVector(REALSXP, (R_xlen_t)n * 2));
    double *Y = REAL(out);
    memcpy(Y, REAL(embedding), (size_t)n * 2 * sizeof(double));

    umap_sgd_run(Y, n, INTEGER(from_), INTEGER(to_), REAL(weights_),
                 ne, n_epochs, n_neg, a, b, alpha0, gamma, seed);

    UNPROTECT(1);
    return out;
}
