#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

typedef struct { float *d; int r, c; } Mat;

#define M(m,i,j)  ((m).d[(i)*(m).c+(j)])

Mat  mat_new(int r, int c);
void mat_del(Mat *m);
void mat_zero(Mat *m);
void mat_randn(Mat *m, float s);

/* C = A @ B */
void mm(const Mat *A, const Mat *B, Mat *C);
/* C += A @ B */
void mm_add(const Mat *A, const Mat *B, Mat *C);
/* C += A^T @ B  (A:K×M, B:K×N → C:M×N) */
void mm_at_add(const Mat *A, const Mat *B, Mat *C);
/* C += A @ B^T  (A:M×K, B:N×K → C:M×N) */
void mm_bt_add(const Mat *A, const Mat *B, Mat *C);

void mat_iadd(Mat *A, const Mat *B);  /* A += B */

/* broadcast bias b[cols] to every row of x */
void add_bias(Mat *x, const float *b);
/* db += sum_rows(dy) */
void bias_bwd(const Mat *dy, float *db);

/* GELU: dy += ... convention for bwd */
void gelu_fwd(const Mat *x, Mat *y);
void gelu_bwd(const Mat *x, const Mat *dy, Mat *dx);

/* row-wise softmax */
void softmax_fwd(const Mat *x, Mat *y);
void softmax_bwd(const Mat *y, const Mat *dy, Mat *dx);

#endif
