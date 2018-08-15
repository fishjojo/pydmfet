#include <mkl.h>

extern "C" {

// svd of a matrix
void mkl_svd(double* A, double* sigma, double* U, double* VT, int m, int n, int method, int* info)
{

      if(method == 1){
	*info = LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'A', m, n, A, m, sigma,
             U , m, VT , n);
      }
      else{
	double *tmpwork = new double[m+n];
	*info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A','A', m, n, A, m, sigma,
             U , m, VT , n, tmpwork);
	delete[] tmpwork;
      }

      return;
}

}
