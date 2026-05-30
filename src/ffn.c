#include "model.h"
#include <string.h>

/*
 * Feed-Forward Network:
 *   h   = GELU(x @ W1 + b1)   (S × F)
 *   out = h @ W2 + b2          (S × D)
 */
void ffn_fwd(const Mat *x, const FW *w, Mat *h, Mat *out) {
    int D = x->c, F = h->c;

    /* pre-activation: x @ W1 + b1 */
    Mat W1 = {w->W1.w, D, F};
    mm(x, &W1, h);
    add_bias(h, w->b1.w);

    /* GELU in-place (reuse h for pre-act, write to h) */
    /* we need pre-activation for backward, so save to h directly
     * and keep it: gelu_fwd reads h and writes to h (same buffer OK
     * because gelu is element-wise) */
    gelu_fwd(h, h);  /* h = GELU(h) */

    /* out = h @ W2 + b2 */
    Mat W2 = {w->W2.w, F, D};
    mm(h, &W2, out);
    add_bias(out, w->b2.w);
}

/*
 * FFN backward.
 * NOTE: h here already contains GELU(pre-act). We need the pre-activation
 * to compute the GELU gradient, but we've overwritten it.
 * SOLUTION: caller must save pre-activation separately OR we recompute.
 *
 * To keep it simple and avoid a separate pre-act buffer, we pass the
 * original x and recompute pre-activation inside backward.
 *
 * dx accumulated into dx.
 */
void ffn_bwd(const Mat *x, const FW *w, const Mat *h,
             const Mat *dout, Mat *dx, FW *dw) {
    int S = x->r, D = x->c, F = h->c;

    /* dW2 += h^T @ dout  (F×D) */
    Mat dW2 = {dw->W2.g, F, D};
    mm_at_add(h, dout, &dW2);
    /* db2 += sum_rows(dout) */
    bias_bwd(dout, dw->b2.g);

    /* dh = dout @ W2^T  (S×F) */
    Mat dh = mat_new(S, F);
    Mat W2 = {w->W2.w, F, D};
    mm_bt_add(dout, &W2, &dh);   /* dh += dout @ W2^T */

    /* recompute pre-activation: pre = x @ W1 + b1  (S×F) */
    Mat pre = mat_new(S, F);
    Mat W1  = {w->W1.w, D, F};
    mm(x, &W1, &pre);
    add_bias(&pre, w->b1.w);

    /* d_pre = GELU'(pre) * dh  (S×F) */
    Mat d_pre = mat_new(S, F);
    gelu_bwd(&pre, &dh, &d_pre);  /* d_pre += GELU'(pre)*dh */

    /* dW1 += x^T @ d_pre  (D×F) */
    Mat dW1 = {dw->W1.g, D, F};
    mm_at_add(x, &d_pre, &dW1);
    /* db1 += sum_rows(d_pre) */
    bias_bwd(&d_pre, dw->b1.g);

    /* dx += d_pre @ W1^T  (S×D) */
    mm_bt_add(&d_pre, &W1, dx);

    mat_del(&dh); mat_del(&pre); mat_del(&d_pre);
}
