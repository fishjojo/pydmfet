#if defined __cplusplus
extern "C" {
#endif

void dcopy_(const int *n, const double *dx, const int *incx,
            double *dy, const int *incy);
void daxpy_(const int *n, const double *da, const double *dx,
            const int *incx, double *dy, const int *incy);
void dgemm_(const char*, const char*,
            const int*, const int*, const int*,
            const double*, const double*, const int*,
            const double*, const int*,
            const double*, double*, const int*);
void dger_(const int *m, const int *n,
           const double *alpha, const double *x,
           const int *incx, const double *y, const int *incy,
           double *a, const int *lda);
void dgesvd_(const char*, const char*,
             const int*, const int*,
             const double*, const int*,
             double*, double*, int*, double*, int*,
             double*, int*, int*);
void dgesdd_(const char*, const int*, const int*,
             const double*,  const int*,
             double*, double*, int*, double*, int*,
             double*, int*, int*, int*);
#if defined __cplusplus
} // end extern "C"
#endif
