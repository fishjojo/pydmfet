/* C source code is found in dgemm_threading_effect_example.c */
#include <mkl.h>
#include <stdio.h>
int main()
{
    printf (" Finding max number of threads Intel(R) MKL can use for parallel runs \n\n");
    int max_threads = mkl_get_max_threads();
    //int max_threads = 4;
    int LOOP_COUNT =10;

    printf (" Running Intel(R) MKL from 1 to %i threads \n\n", max_threads);
    int i, j, r;
    int m = 1000;
    int n = 1000;
    int k = 1000;
    double *A = new double [m*k];
    double *B = new double [k*n];
    double *C = new double [m*n];
    double beta = 0.0;
    double alpha = 1.0;
    for (i = 1; i <= max_threads; i++) {
        for (j = 0; j < (m*n); j++)
            C[j] = 0.0;
        
        printf (" Requesting Intel(R) MKL to use %i thread(s) \n\n", i);
        mkl_set_num_threads(i);

        printf (" Making the first run of matrix product using Intel(R) MKL dgemm function \n"
                " via CBLAS interface to get stable run time measurements \n\n");
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    m, n, k, alpha, A, k, B, n, beta, C, n);
        
        printf (" Measuring performance of matrix product using Intel(R) MKL dgemm function \n"
                " via CBLAS interface on %i thread(s) \n\n", i);
        double s_initial = dsecnd();
        for (r = 0; r < LOOP_COUNT; r++) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                        m, n, k, alpha, A, k, B, n, beta, C, n);
        }
        double s_elapsed = (dsecnd() - s_initial) / LOOP_COUNT;

        printf (" == Matrix multiplication using Intel(R) MKL dgemm completed ==\n"
                " == at %.5f milliseconds using %d thread(s) ==\n\n", (s_elapsed * 1000), i);
    }

    return 0;
}
 
