#include "matrix.h"

Mat mat_new(int r, int c) {
    Mat m; m.r = r; m.c = c;
    m.d = (float*)calloc(r * c, sizeof(float));
    if (!m.d) { fprintf(stderr, "OOM mat_new(%d,%d)\n", r, c); exit(1); }
    return m;
}
void mat_del(Mat *m)  { free(m->d); m->d = NULL; m->r = m->c = 0; }
void mat_zero(Mat *m) { memset(m->d, 0, (size_t)m->r * m->c * sizeof(float)); }

static float randn1(void) {
    static int has = 0; static float nxt;
    if (has) { has = 0; return nxt; }
    float u = ((float)rand() + 1) / ((float)RAND_MAX + 2);
    float v = ((float)rand() + 1) / ((float)RAND_MAX + 2);
    float r = sqrtf(-2.f * logf(u));
    float t = 6.2831853f * v;
    nxt = r * sinf(t); has = 1;
    return r * cosf(t);
}
void mat_randn(Mat *m, float s) {
    for (int i = 0; i < m->r * m->c; i++) m->d[i] = randn1() * s;
}

/* C += A @ B  (naive, correct) */
static void sgemm_add(int M, int N, int K,
                      const float *A, int ldA,
                      const float *B, int ldB,
                      float *C, int ldC) {
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            float a = A[i*ldA+k];
            if (a == 0.f) continue;
            for (int j = 0; j < N; j++)
                C[i*ldC+j] += a * B[k*ldB+j];
        }
}

void mm(const Mat *A, const Mat *B, Mat *C) {
    mat_zero(C);
    sgemm_add(A->r, B->c, A->c, A->d, A->c, B->d, B->c, C->d, C->c);
}
void mm_add(const Mat *A, const Mat *B, Mat *C) {
    sgemm_add(A->r, B->c, A->c, A->d, A->c, B->d, B->c, C->d, C->c);
}
/* C += A^T @ B */
void mm_at_add(const Mat *A, const Mat *B, Mat *C) {
    int M = A->c, K = A->r, N = B->c;
    for (int k = 0; k < K; k++)
        for (int i = 0; i < M; i++) {
            float a = A->d[k*A->c+i];
            if (a == 0.f) continue;
            for (int j = 0; j < N; j++)
                C->d[i*C->c+j] += a * B->d[k*B->c+j];
        }
}
/* C += A @ B^T */
void mm_bt_add(const Mat *A, const Mat *B, Mat *C) {
    int M = A->r, N = B->r, K = A->c;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.f;
            for (int k = 0; k < K; k++)
                s += A->d[i*A->c+k] * B->d[j*B->c+k];
            C->d[i*C->c+j] += s;
        }
}

void mat_iadd(Mat *A, const Mat *B) {
    int n = A->r * A->c;
    for (int i = 0; i < n; i++) A->d[i] += B->d[i];
}

void add_bias(Mat *x, const float *b) {
    for (int i = 0; i < x->r; i++)
        for (int j = 0; j < x->c; j++)
            x->d[i*x->c+j] += b[j];
}
void bias_bwd(const Mat *dy, float *db) {
    for (int i = 0; i < dy->r; i++)
        for (int j = 0; j < dy->c; j++)
            db[j] += dy->d[i*dy->c+j];
}

/* GELU: 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3))) */
static inline float gelu1(float x) {
    float k = 0.7978845608f;
    float t = tanhf(k*(x + 0.044715f*x*x*x));
    return 0.5f*x*(1.f+t);
}
static inline float gelu_d(float x) {
    float k = 0.7978845608f;
    float u = k*(x + 0.044715f*x*x*x);
    float t = tanhf(u);
    float dt = 1.f - t*t;
    float du = k*(1.f + 3.f*0.044715f*x*x);
    return 0.5f*(1.f+t) + 0.5f*x*dt*du;
}
void gelu_fwd(const Mat *x, Mat *y) {
    int n = x->r*x->c;
    for (int i = 0; i < n; i++) y->d[i] = gelu1(x->d[i]);
}
void gelu_bwd(const Mat *x, const Mat *dy, Mat *dx) {
    int n = x->r*x->c;
    for (int i = 0; i < n; i++) dx->d[i] += gelu_d(x->d[i]) * dy->d[i];
}

void softmax_fwd(const Mat *x, Mat *y) {
    for (int i = 0; i < x->r; i++) {
        const float *xi = x->d + i*x->c;
        float *yi = y->d + i*y->c;
        float mx = xi[0];
        for (int j = 1; j < x->c; j++) if (xi[j] > mx) mx = xi[j];
        float s = 0.f;
        for (int j = 0; j < x->c; j++) { yi[j] = expf(xi[j]-mx); s += yi[j]; }
        for (int j = 0; j < x->c; j++) yi[j] /= s;
    }
}
/* dx += y*(dy - dot(y,dy))  per row */
void softmax_bwd(const Mat *y, const Mat *dy, Mat *dx) {
    for (int i = 0; i < y->r; i++) {
        const float *yi  = y->d  + i*y->c;
        const float *dyi = dy->d + i*dy->c;
        float *dxi = dx->d + i*dx->c;
        float dot = 0.f;
        for (int j = 0; j < y->c; j++) dot += yi[j]*dyi[j];
        for (int j = 0; j < y->c; j++) dxi[j] += yi[j]*(dyi[j]-dot);
    }
}
