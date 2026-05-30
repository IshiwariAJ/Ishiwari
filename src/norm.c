#include "model.h"
#include <math.h>
#include <string.h>

/*
 * Layer Norm: y[i][j] = gamma[j] * xhat[i][j] + beta[j]
 * where xhat[i][j] = (x[i][j] - mean[i]) / sqrt(var[i] + eps)
 * mean[i] and var[i] are computed over the D features.
 */
void ln_fwd(const Mat *x, const LN *w, Mat *y,
            float *mean, float *var, float eps) {
    int S = x->r, D = x->c;
    const float *gm = w->scale.w;
    const float *bt = w->shift.w;
    for (int i = 0; i < S; i++) {
        const float *xi = x->d + i*D;
        float *yi = y->d + i*D;
        /* mean */
        float mu = 0.f;
        for (int j = 0; j < D; j++) mu += xi[j];
        mu /= D;
        mean[i] = mu;
        /* variance */
        float v = 0.f;
        for (int j = 0; j < D; j++) { float d = xi[j]-mu; v += d*d; }
        v /= D;
        var[i] = v;
        float inv = 1.f / sqrtf(v + eps);
        for (int j = 0; j < D; j++)
            yi[j] = gm[j] * (xi[j] - mu) * inv + bt[j];
    }
}

/*
 * Layer Norm backward.
 * Let g[j] = gamma[j] * dy[j]  (gradient through gamma).
 * dx[i][j] += inv * (g[j] - (1/n)*sum_k(g[k]) - xhat[j]*(1/n)*sum_k(g[k]*xhat[k]))
 * dgamma[j] += sum_i(dy[i][j] * xhat[i][j])
 * dbeta[j]  += sum_i(dy[i][j])
 *
 * Note: sum terms must be computed over g = gamma*dy, not dy alone,
 * because gamma is not uniform across dimensions.
 */
void ln_bwd(const Mat *x, const LN *w,
            const float *mean, const float *var,
            const Mat *dy, Mat *dx, LN *dw, float eps) {
    int S = x->r, D = x->c;
    const float *gm  = w->scale.w;
    float *dgm = dw->scale.g;
    float *dbt = dw->shift.g;

    for (int i = 0; i < S; i++) {
        const float *xi  = x->d  + i*D;
        const float *dyi = dy->d + i*D;
        float *dxi = dx->d + i*D;
        float inv  = 1.f / sqrtf(var[i] + eps);
        float mu   = mean[i];

        /* sum_g = sum(gamma*dy),  sum_gx = sum(gamma*dy*xhat) */
        float sum_g = 0.f, sum_gx = 0.f;
        for (int j = 0; j < D; j++) {
            float xh = (xi[j] - mu) * inv;
            float g  = gm[j] * dyi[j];
            sum_g  += g;
            sum_gx += g * xh;
            dgm[j] += dyi[j] * xh;
            dbt[j] += dyi[j];
        }
        float rD = 1.f / D;
        for (int j = 0; j < D; j++) {
            float xh = (xi[j] - mu) * inv;
            float g  = gm[j] * dyi[j];
            dxi[j] += inv * (g - rD*sum_g - xh*rD*sum_gx);
        }
    }
}
