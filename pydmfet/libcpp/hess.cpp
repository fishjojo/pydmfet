#include <omp.h>
#include <mkl.h>
#include <iostream>
#include <ctime>
#include <chrono>
#include <math.h>
//#include <rank_revealing_algorithms_intel_mkl.h>

using namespace std;

extern "C" {

void VRadd(double* C, double* A, double* B, int N)
{
   for (int i=0; i<N; i++)
	C[i] = A[i] + B[i];

   return;
}

void VRaxpy(double* C, double a, double* x, double* y, int N)
{
   for (int i=0; i<N; i++)
	C[i] = a*x[i] + y[i];

   return;
}


void VRcopy(double* y, double* x, int N)
{
   for (int i=0; i<N; i++)
	y[i] = x[i];

   return;
}

void calc_hess_dm_fast(double* hess, double* jCa, double* orb_Ea, int dim, int NBas, int NAlpha, int nthread)
{
   //clock_t startcputime = clock();
   //auto wcts = chrono::system_clock::now();

   int N2 = NBas*NBas;
   int NOrb = NBas;  //assume no linear dependence

   int imax = NAlpha;
   int amax = NOrb;
   int NOa = imax;
   int NVa = amax-imax;
   int NOVa = NOa*NVa;

   double *jt = new double [N2*NOVa];
   for(int i=0; i<N2*NOVa; i++)
      jt[i] = 0.0;


   //mkl_set_dynamic( 0 );
   omp_set_num_threads(nthread);
   //mkl_set_num_threads(1);

   #pragma omp parallel for schedule(static)
   for(int i=0; i<imax; i++){
      int index_munu = i*N2*NVa;
      for(int a=imax; a<amax; a++){
         for(int mu=0; mu<NBas; mu++){
	    double Cmui=jCa[mu+i*NBas];
	    //VRaxpy(jt+index_munu,Cmui,jCa+a*NBas,jt+index_munu,NBas);
	    cblas_daxpy(NBas, Cmui, jCa+a*NBas, 1, jt+index_munu, 1);
	    index_munu += NBas;
	 }
      }
   } 

   double *jt_dia = new double [N2*NOVa];
   for(int i=0; i<N2*NOVa; i++)
      jt_dia[i] = 0.0;


   #pragma omp parallel for schedule(static)
   for(int i=0; i<imax; i++){
      int index_ia = i*NVa;
      for(int a=imax; a<amax; a++){
	 double eps_ia = orb_Ea[i] - orb_Ea[a];
	 double dia = 1.0/eps_ia;
	 int ioff = N2*index_ia;
	 //VRaxpy(jt_dia+ioff, dia, jt+ioff, jt_dia+ioff, N2);
	 cblas_daxpy(N2, dia, jt+ioff, 1, jt_dia+ioff, 1);
	 index_ia++;
      }
   }

   double *jHfull = new double [N2*N2];
   //mkl_set_num_threads(nthread);
   //cout<<"mkl_max_threads = "<<mkl_get_max_threads()<<endl;
   //AtimsB(jHfull,jt,jt_dia,N2,N2,NOVa,N2,N2,N2,3);
   cblas_dgemm (CblasColMajor, CblasNoTrans, CblasTrans, N2, N2, NOVa, 1.0, jt, N2, jt_dia, N2, 0.0, jHfull, N2);

   delete [] jt;
   delete [] jt_dia;

  
   //mkl_set_num_threads(1);
 #pragma omp parallel
 {
   //int nthread = omp_get_num_threads();
   //int tid = omp_get_thread_num();
   //if(tid == 0){
   //  cout<<" nthread = " << nthread << endl;
   //}

   //are we private
   double *jTemp = new double[N2];
   double *jTemp1 = new double[N2];

   #pragma omp for schedule(dynamic)
   for(int mu=0; mu<NBas; mu++){
      int index = (2*NBas-mu+1)*mu/2*dim;
      for(int nu=mu; nu<NBas; nu++){
	 //VRcopy(jTemp, jHfull + (mu+nu*NBas)*N2, N2);
	 cblas_dcopy (N2, jHfull + (mu+nu*NBas)*N2, 1, jTemp, 1);
	 mkl_dimatcopy ('C', 'T', NBas, NBas, 1.0, jTemp, NBas, NBas);
	 VRadd(jTemp, jTemp, jHfull + (mu+nu*NBas)*N2, N2);

	 //VRcopy(jTemp1, jHfull + (nu+mu*NBas)*N2, N2);
	 cblas_dcopy (N2, jHfull + (nu+mu*NBas)*N2, 1, jTemp1, 1);
	 mkl_dimatcopy ('C', 'T', NBas, NBas, 1.0, jTemp1, NBas, NBas);
	 VRadd(jTemp1, jTemp1, jHfull + (nu+mu*NBas)*N2, N2);

	 VRadd(jTemp,jTemp, jTemp1, N2);

	 for(int i=0; i<NBas; i++)
	    jTemp[i*NBas+i] *= 0.5;

         for(int lam=0; lam<NBas; lam++){
	    //VRcopy(hess+index, jTemp+lam*NBas+lam, NBas-lam);
	    cblas_dcopy (NBas-lam, jTemp+lam*NBas+lam, 1, hess+index, 1);
	    index += NBas-lam;
	 }
      }
   }

   delete[] jTemp; 
   delete[] jTemp1;
 }

   delete [] jHfull; 

   //mkl_set_num_threads(nthread);
   mkl_dimatcopy ('C', 'T', dim, dim, -2.0, hess, dim, dim);

/*
   double cpu_duration = (clock() - startcputime) / (double)CLOCKS_PER_SEC;
   cout << "Finished in " << cpu_duration << " seconds [CPU Clock] " << endl;
   chrono::duration<double> wctduration = (chrono::system_clock::now() - wcts);
   cout << "Finished in " << wctduration.count() << " seconds [Wall Clock]" << endl;
*/


   return;
}




void calc_hess_dm_fast_frac(double* hess, double* jCa, double* orb_Ea, double* mo_occ, 
			    int dim, int NBas, int NAlpha, int nthread, double smear=0.0, double tol=1e-8)
{
   //clock_t startcputime = clock();
   //auto wcts = chrono::system_clock::now();

   int N2 = NBas*NBas;
   int NOrb = NBas;  //assume no linear dependence

   int amax = NOrb;
   int imax = 0;
   for(int i=0; i<NOrb; i++){
     if(mo_occ[i] > tol) { imax++; }
   }

   int NOa = imax;
   int NVa = NOrb-1;
   int NOVa = NOa*NVa;

   double *jt = new double [N2*NOVa];
   for(int i=0; i<N2*NOVa; i++)
      jt[i] = 0.0;


   //mkl_set_dynamic( 0 );
   omp_set_num_threads(nthread);
   //mkl_set_num_threads(1);

   #pragma omp parallel for schedule(static)
   for(int i=0; i<imax; i++){
      int index_munu = i*N2*NVa;
      for(int a=0; a<amax; a++){
	if(a != i){
         for(int mu=0; mu<NBas; mu++){
	    double Cmui=jCa[mu+i*NBas];
	    //VRaxpy(jt+index_munu,Cmui,jCa+a*NBas,jt+index_munu,NBas);
	    cblas_daxpy(NBas, Cmui, jCa+a*NBas, 1, jt+index_munu, 1);
	    index_munu += NBas;
	 }
        }
      }
   } 

   double *jt_dia = new double [N2*NOVa];
   for(int i=0; i<N2*NOVa; i++)
      jt_dia[i] = 0.0;


   #pragma omp parallel for schedule(static)
   for(int i=0; i<imax; i++){
      int index_ia = i*NVa;
      double occ_i = mo_occ[i];
      for(int a=0; a<amax; a++){
	double occ_a = mo_occ[a];
	if(a != i){
	 double eps_ia = orb_Ea[i] - orb_Ea[a];
	 double dia;
	 if(fabs(eps_ia) < 1e-3){
	    cout << "degenerate orbitals detected!!!!!"<<endl;
	    dia = 0.0;
	 }
	 else{
	    dia = occ_i/eps_ia;
	 }
	 int ioff = N2*index_ia;
	 //VRaxpy(jt_dia+ioff, dia, jt+ioff, jt_dia+ioff, N2);
	 cblas_daxpy(N2, dia, jt+ioff, 1, jt_dia+ioff, 1);
	 index_ia++;
	}
      }
   }

   double *jHfull = new double [N2*N2];
   //mkl_set_num_threads(nthread);
   //cout<<"mkl_max_threads = "<<mkl_get_max_threads()<<endl;
   //AtimsB(jHfull,jt,jt_dia,N2,N2,NOVa,N2,N2,N2,3);
   cblas_dgemm (CblasColMajor, CblasNoTrans, CblasTrans, N2, N2, NOVa, 1.0, jt, N2, jt_dia, N2, 0.0, jHfull, N2);

   delete [] jt;
   delete [] jt_dia;


   //occupation number contribution
   //
   if(smear > 1e-8){
     cout<<" Adding contributions from fluctuation of occupation numbers."<<endl;
     double *jt_homo = new double [N2];
     double *jt_lumo = new double [N2];
     for(int i=0; i<N2; i++){
	jt_homo[i] = 0.0;
	jt_lumo[i] = 0.0;
     } 

     int i_homo = NAlpha - 1;
     int i_lumo = i_homo + 1;
     cblas_dger(CblasColMajor, NBas, NBas, 1.0, jCa+i_homo*NBas, 1, jCa+i_homo*NBas, 1, jt_homo, NBas);
     cblas_dger(CblasColMajor, NBas, NBas, 1.0, jCa+i_lumo*NBas, 1, jCa+i_lumo*NBas, 1, jt_lumo, NBas);

     for(int i=0; i<NOrb; i++){
        double occ = mo_occ[i]/2.0;
	if(occ<tol || 1.0-occ<tol) continue;

	double *jt = new double [N2];
	for(int j=0; j<N2; j++) jt[j] = 0.0;

	cblas_dger(CblasColMajor, NBas, NBas, 1.0, jCa+i*NBas, 1, jCa+i*NBas, 1, jt, NBas);
	double factor = -2.0*occ*(1.0-occ)/smear;
	cblas_dger(CblasColMajor, N2, N2, factor, jt, 1, jt, 1, jHfull, N2);

	factor *= -0.5;
	cblas_dger(CblasColMajor, N2, N2, factor, jt, 1, jt_homo, 1, jHfull, N2);
	cblas_dger(CblasColMajor, N2, N2, factor, jt, 1, jt_lumo, 1, jHfull, N2);

	delete [] jt;
     }

     delete [] jt_homo;
     delete [] jt_lumo;
   }
  
   //mkl_set_num_threads(1);
 #pragma omp parallel
 {
   //int nthread = omp_get_num_threads();
   //int tid = omp_get_thread_num();
   //if(tid == 0){
   //  cout<<" nthread = " << nthread << endl;
   //}

   //are we private
   double *jTemp = new double[N2];
   double *jTemp1 = new double[N2];

   #pragma omp for schedule(dynamic)
   for(int mu=0; mu<NBas; mu++){
      int index = (2*NBas-mu+1)*mu/2*dim;
      for(int nu=mu; nu<NBas; nu++){
	 //VRcopy(jTemp, jHfull + (mu+nu*NBas)*N2, N2);
	 cblas_dcopy (N2, jHfull + (mu+nu*NBas)*N2, 1, jTemp, 1);
	 mkl_dimatcopy ('C', 'T', NBas, NBas, 1.0, jTemp, NBas, NBas);
	 VRadd(jTemp, jTemp, jHfull + (mu+nu*NBas)*N2, N2);

	 //VRcopy(jTemp1, jHfull + (nu+mu*NBas)*N2, N2);
	 cblas_dcopy (N2, jHfull + (nu+mu*NBas)*N2, 1, jTemp1, 1);
	 mkl_dimatcopy ('C', 'T', NBas, NBas, 1.0, jTemp1, NBas, NBas);
	 VRadd(jTemp1, jTemp1, jHfull + (nu+mu*NBas)*N2, N2);

	 VRadd(jTemp,jTemp, jTemp1, N2);

	 for(int i=0; i<NBas; i++)
	    jTemp[i*NBas+i] *= 0.5;

         for(int lam=0; lam<NBas; lam++){
	    //VRcopy(hess+index, jTemp+lam*NBas+lam, NBas-lam);
	    cblas_dcopy (NBas-lam, jTemp+lam*NBas+lam, 1, hess+index, 1);
	    index += NBas-lam;
	 }
      }
   }

   delete[] jTemp; 
   delete[] jTemp1;
 }

   delete [] jHfull; 

   //mkl_set_num_threads(nthread);
   mkl_dimatcopy ('C', 'T', dim, dim, -2.0, hess, dim, dim);

/*
   double cpu_duration = (clock() - startcputime) / (double)CLOCKS_PER_SEC;
   cout << "Finished in " << cpu_duration << " seconds [CPU Clock] " << endl;
   chrono::duration<double> wctduration = (chrono::system_clock::now() - wcts);
   cout << "Finished in " << wctduration.count() << " seconds [Wall Clock]" << endl;
*/


   return;
}



}
