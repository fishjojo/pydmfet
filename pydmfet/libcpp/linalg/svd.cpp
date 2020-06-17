#include "config.h"
#include "fblas.h"

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#if defined __cplusplus
extern "C" {
#endif

// svd of a matrix
void mkl_svd(double* A, double* sigma, double* U, double* VT, int m, int n, int method, int* info)
{
    if(method == 1){
        const char JOBZ = 'A';
        int mn = MIN(m,n);
        int mx = MAX(m,n);
        int lwork = (4*mn*mn + 6*mn + mx)*2;
        double *tmpwork = new double[lwork];
        int *iwork = new int[8*MIN(m,n)*2];
	    dgesdd_(&JOBZ, &m, &n, A, &m, sigma,
                U, &m, VT, &n, tmpwork, &lwork, iwork, info);
        delete [] tmpwork;
        delete [] iwork;
    }
    else{
        const char JOBU = 'A';
        const char JOBVT = 'A';
        int lwork = MAX(3*MIN(m,n) + MAX(m,n), 5*MIN(m,n))*2;
        double *tmpwork = new double[lwork];
	    dgesvd_(&JOBU,&JOBVT, &m, &n, A, &m, sigma,
                U , &m, VT , &n, tmpwork, &lwork, info);
	    delete[] tmpwork;
    }

    return;
}
#if defined __cplusplus
} // end extern "C"
#endif
