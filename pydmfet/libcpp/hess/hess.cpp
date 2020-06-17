#include <iostream>
//#include <ctime>
//#include <chrono>
#include <math.h>
#include <vector>
#include <algorithm> 
#include <iomanip>
#include "config.h"
#include "blas/fblas.h"
//#include <rank_revealing_algorithms_intel_mkl.h>

//#include <pybind11/pybind11.h>

#define A(i,j)        *(A+(i)+N*(j))
#define B(i,j)        *(B+(i)+N*(j))

using namespace std;

#if defined __cplusplus
extern "C" {
#endif

int degen_subspac(vector<vector<int> > &sub, double* mo_energy, int norb, double e_tol);
int orb_in_subspac(int orb_i, vector<vector<int> > sub);
bool orb_pair_in_same_subspac(int orb_i, int orb_j, vector<vector<int> > sub);
void get_jt(double* jt, int orb_i, vector<vector<int> > sub, double* jCa, int NBas);
void get_jt_deg(double* jt, vector<vector<int> > sub, int index, double* jCa, int NBas);


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

void Transpose(double* A, double* B, int N)
{
  int i,j;
  if (N > 0) {
    for (i = 0; i < N; ++i)
      for (j = 0; j <= i; ++j) {
        B(i,j) = A(j,i);
        B(j,i) = A(i,j);
      }
    return;
  }
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


   const int I1 = 1;

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
            daxpy_ (&NBas, &Cmui, jCa+a*NBas, &I1, jt+index_munu, &I1);
            index_munu += NBas;
         }
      }
   } 

   double *jt_T = new double [N2*NOVa];
   #pragma omp parallel for schedule(static)
   for(int i=0; i<imax; i++){
      int index_munu = i*N2*NVa;
      for(int a=imax; a<amax; a++){
         //mkl_domatcopy ('C', 'T', NBas, NBas, 1.0, jt+index_munu, NBas, jt_T+index_munu, NBas);
         Transpose(jt+index_munu, jt_T+index_munu, NBas);
         index_munu += N2;
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
         double dia = 2.0/eps_ia;
         int ioff = N2*index_ia;
         //VRaxpy(jt_dia+ioff, dia, jt+ioff, jt_dia+ioff, N2);
         daxpy_ (&N2, &dia, jt+ioff, &I1, jt_dia+ioff, &I1);
         index_ia++;
      }
   }

   double *jt_T_dia = new double [N2*NOVa];
   #pragma omp parallel for schedule(static)
   for(int i=0; i<imax; i++){
      int index_ia = i*NVa;
      for(int a=imax; a<amax; a++){
         int ioff = N2*index_ia;
         //mkl_domatcopy ('C', 'T', NBas, NBas, 1.0, jt_dia+ioff, NBas, jt_T_dia+ioff, NBas);
         Transpose(jt_dia+ioff, jt_T_dia+ioff, NBas);
         index_ia++;
      }
   }

   const char NOTRANS='N';
   const char TRANS='T';
   const double D1 = 1.0;
   const double D0 = 0.0;

   double *jHfull = new double [N2*N2];
   //mkl_set_num_threads(nthread);
   //cout<<"mkl_max_threads = "<<mkl_get_max_threads()<<endl;
   //AtimsB(jHfull,jt,jt_dia,N2,N2,NOVa,N2,N2,N2,3);
   dgemm_ (&NOTRANS, &TRANS, &N2, &N2, &NOVa, &D1, jt, &N2, jt_dia, &N2, &D0, jHfull, &N2);

   delete [] jt;
   delete [] jt_dia;

   double *jHfull_cc = new double [N2*N2];
   dgemm_ (&NOTRANS, &TRANS, &N2, &N2, &NOVa, &D1, jt_T, &N2, jt_T_dia, &N2, &D0, jHfull_cc, &N2);

   delete [] jt_T;
   delete [] jt_T_dia;
  
   #pragma omp parallel for schedule(static)
   for(int i=0; i<N2; i++){
      int ioff = i*N2;
      //vdAdd(N2, jHfull+ioff, jHfull_cc+ioff, jHfull+ioff);
      VRadd(jHfull+ioff, jHfull+ioff, jHfull_cc+ioff, N2);
   } 
   delete [] jHfull_cc;


   //mkl_set_num_threads(1);
 #pragma omp parallel
 {
   //int nthread = omp_get_num_threads();
   //int tid = omp_get_thread_num();
   //if(tid == 0){
   //  cout<<" nthread = " << nthread << endl;
   //}

   double *jTemp = new double[N2];
   #pragma omp for schedule(dynamic)
   for(int mu=0; mu<NBas; mu++){
      int index = (2*NBas-mu+1)*mu/2*dim;
      for(int nu=mu; nu<NBas; nu++){
         if (mu != nu) {
           VRadd(jTemp, jHfull + (nu+mu*NBas)*N2, jHfull + (mu+nu*NBas)*N2, N2);
         }

         double *ptr = jTemp;
         if (mu == nu) ptr = jHfull + (mu+nu*NBas)*N2;

         for(int lam=0; lam<NBas; lam++){
            int nelem = NBas-lam;
            dcopy_ (&nelem, ptr+lam*NBas+lam, &I1, hess+index, &I1);
            index += NBas-lam;
         }
      }
   }

   delete[] jTemp; 
 }

   delete [] jHfull; 

   //mkl_set_num_threads(nthread);
   //mkl_dimatcopy ('C', 'T', dim, dim, -2.0, hess, dim, dim);

/*
   double cpu_duration = (clock() - startcputime) / (double)CLOCKS_PER_SEC;
   cout << "Finished in " << cpu_duration << " seconds [CPU Clock] " << endl;
   chrono::duration<double> wctduration = (chrono::system_clock::now() - wcts);
   cout << "Finished in " << wctduration.count() << " seconds [Wall Clock]" << endl;
*/


   return;
}




void calc_hess_dm_fast_frac(double* hess, double* jCa, double* orb_Ea, double* mo_occ, 
                            int dim, int NBas, int nthread, double smear, double e_tol, double occ_tol)
{
   //clock_t startcputime = clock();
   //auto wcts = chrono::system_clock::now();

   int N2 = NBas*NBas;
   int NOrb = NBas;  //assume no linear dependence

   int amax = NOrb;
   int imax = 0;
   for(int i=0; i<NOrb; i++){
      if(mo_occ[i] >= occ_tol) { imax++; }
   }

   int NOa = imax;
   int NVa = NOrb-1;
   //if(smear < 1e-8) NVa = amax-imax;
   int NOVa = NOa*NVa;


   vector<vector<int> > sub;
   int nsub = 0;
   nsub = degen_subspac(sub, orb_Ea, NOrb, e_tol);
   if(nsub > 0){
      cout<<"degenerate subspace detected:\n";
      cout<<"nsub = "<<nsub<<"\n";
      for(int i=0; i<nsub; i++){
         cout<<"subspace "<<i+1<<":\n";
         int sub_size = sub[i].size();
         for(int j=0; j<sub_size; j++){
            if(mo_occ[sub[i][j]] > occ_tol) {cout<<sub[i][j]+1<<", ";}
         }
         cout<<"\n";
      }
      cout<<endl;
   }


   double *jt = new double [N2*NOVa];
   for(int i=0; i<N2*NOVa; i++)
      jt[i] = 0.0;


   //mkl_set_dynamic( 0 );
   omp_set_num_threads(nthread);
   //mkl_set_num_threads(1);

   int a_start = 0;
   const int I1 = 1;
   //if(smear < 1e-8) a_start = imax;
   #pragma omp parallel for schedule(static)
   for(int i=0; i<imax; i++){
      int index_munu = i*N2*NVa;
      for(int a=a_start; a<amax; a++){
        if(a != i){
         for(int mu=0; mu<NBas; mu++){
            double Cmui=jCa[mu+i*NBas];
            //VRaxpy(jt+index_munu,Cmui,jCa+a*NBas,jt+index_munu,NBas);
            daxpy_ (&NBas, &Cmui, jCa+a*NBas, &I1, jt+index_munu, &I1);
            index_munu += NBas;
         }
        }
      }
   } 

   double *jt_T = new double [N2*NOVa];
   #pragma omp parallel for schedule(static)
   for(int i=0; i<imax; i++){
      int index_munu = i*N2*NVa;
      for(int a=a_start; a<amax; a++){
        if(a != i){
         //mkl_domatcopy ('C', 'T', NBas, NBas, 1.0, jt+index_munu, NBas, jt_T+index_munu, NBas);
         Transpose(jt+index_munu, jt_T+index_munu, NBas);
         index_munu += N2;
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
      for(int a=a_start; a<amax; a++){
        //double occ_a = mo_occ[a];
        if(a != i){
         double eps_ia = orb_Ea[i] - orb_Ea[a];
         double dia;
         if(orb_pair_in_same_subspac(i,a,sub)){
            //cout << "near degenerate orbitals detected: "<<i+1<<", "<<a+1<<endl;
            dia = 0.0;
         }
         else{
            dia = occ_i/eps_ia;
         }
         int ioff = N2*index_ia;
         //VRaxpy(jt_dia+ioff, dia, jt+ioff, jt_dia+ioff, N2);
         daxpy_ (&N2, &dia, jt+ioff, &I1, jt_dia+ioff, &I1);
         index_ia++;
        }
      }
   }


   double *jt_T_dia = new double [N2*NOVa];
   #pragma omp parallel for schedule(static)
   for(int i=0; i<imax; i++){
      int index_ia = i*NVa;
      for(int a=a_start; a<amax; a++){
        if(a != i){
         int ioff = N2*index_ia;
         //mkl_domatcopy ('C', 'T', NBas, NBas, 1.0, jt_dia+ioff, NBas, jt_T_dia+ioff, NBas);
         Transpose(jt_dia+ioff, jt_T_dia+ioff, NBas);
         index_ia++;
        }
      }
   }


   const char NOTRANS='N';
   const char TRANS='T';
   const double D1 = 1.0;
   const double D0 = 0.0;

   double *jHfull = new double [N2*N2];
   //mkl_set_num_threads(nthread);
   //cout<<"mkl_max_threads = "<<mkl_get_max_threads()<<endl;
   //AtimsB(jHfull,jt,jt_dia,N2,N2,NOVa,N2,N2,N2,3);
   dgemm_ (&NOTRANS, &TRANS, &N2, &N2, &NOVa, &D1, jt, &N2, jt_dia, &N2, &D0, jHfull, &N2);

   delete [] jt;
   delete [] jt_dia;


   double *jHfull_cc = new double [N2*N2];
   dgemm_ (&NOTRANS, &TRANS, &N2, &N2, &NOVa, &D1, jt_T, &N2, jt_T_dia, &N2, &D0, jHfull_cc, &N2);

   delete [] jt_T;
   delete [] jt_T_dia;

   #pragma omp parallel for schedule(static)
   for(int i=0; i<N2; i++){
      int ioff = i*N2;
      //vdAdd(N2, jHfull+ioff, jHfull_cc+ioff, jHfull+ioff);
      VRadd(jHfull+ioff, jHfull+ioff, jHfull_cc+ioff, N2);
   }
   delete [] jHfull_cc;


   //occupation number contribution
   //
   double pfunc = 0.0;
   for(int i=0; i<NOrb; i++){
      double occ = mo_occ[i]/2.0;
      if(occ < occ_tol/2.0 || 1.0-occ < occ_tol/2.0) continue;
      pfunc += occ*(1.0-occ);
   }

   if(smear > 1e-8 && pfunc > 1e-8){
     cout<<" Adding contributions from fluctuation of occupation numbers."<<endl;

     double *jt_sum = new double [N2];
     double *jt0_sum = new double [N2];
     for(int i=0; i<N2; i++) {
        jt_sum[i]=0.0;
        jt0_sum[i]=0.0;
     }

     //orbital energy part
     for(int i=0; i<NOrb; i++){
        double occ = mo_occ[i]/2.0;
        if(occ < occ_tol/2.0 || 1.0-occ < occ_tol/2.0) continue;

        double *jt = new double [N2];
        for(int j=0; j<N2; j++) jt[j] = 0.0;
        get_jt(jt, i, sub, jCa, NBas);

        double *jt0 = new double [N2];
        for(int j=0; j<N2; j++) jt0[j] = 0.0;
        dger_(&NBas, &NBas, &D1, jCa+i*NBas, &I1, jCa+i*NBas, &I1, jt0, &NBas);

        double fac = -2.0*occ*(1.0-occ)/smear;
        dger_(&N2, &N2, &fac, jt0, &I1, jt, &I1, jHfull, &N2);

        double fac2 = occ*(1.0-occ);
        daxpy_ (&N2, &fac2, jt, &I1, jt_sum, &I1);
        daxpy_ (&N2, &fac2, jt0, &I1, jt0_sum, &I1);

        delete [] jt;
        delete [] jt0;
     }

     //chemical potential part
     double fac = 2.0/smear/pfunc;
     dger_(&N2, &N2, &fac, jt0_sum, &I1, jt_sum, &I1, jHfull, &N2);

     delete [] jt_sum;
     delete [] jt0_sum;
   }


//Transform Hessian to up-triangle form
  #pragma omp parallel
  {

   double *jTemp = new double[N2];
   #pragma omp for schedule(dynamic)
   for(int mu=0; mu<NBas; mu++){
      int index = (2*NBas-mu+1)*mu/2*dim;
      for(int nu=mu; nu<NBas; nu++){
         if (mu != nu) {
           VRadd(jTemp, jHfull + (nu+mu*NBas)*N2, jHfull + (mu+nu*NBas)*N2, N2);
         }

         double *ptr = jTemp;
         if (mu == nu) ptr = jHfull + (mu+nu*NBas)*N2;

         for(int lam=0; lam<NBas; lam++){
            int nelem = NBas-lam;
            dcopy_ (&nelem, ptr+lam*NBas+lam, &I1, hess+index, &I1);
            index += NBas-lam;
         }
      }
   }

   delete[] jTemp;
  }

   delete [] jHfull; 


/*
   double cpu_duration = (clock() - startcputime) / (double)CLOCKS_PER_SEC;
   cout << "Finished in " << cpu_duration << " seconds [CPU Clock] " << endl;
   chrono::duration<double> wctduration = (chrono::system_clock::now() - wcts);
   cout << "Finished in " << wctduration.count() << " seconds [Wall Clock]" << endl;
*/


   return;
}


int degen_subspac(vector<vector<int> > &sub, double* mo_energy, int norb, double e_tol)
{
  double e_prev = -999999.0;
  double e_cur = 0.0;
  int nelem = 0;
  int nsub = 0;
  for (int i=0; i<norb; i++){
        e_cur = mo_energy[i];
        if(fabs(e_cur-e_prev) > e_tol){
           e_prev = e_cur;
           nelem = 0;
        }
        else{
           if(nelem == 0){
              vector<int> v;
              sub.push_back(v);
              nsub++;
              sub.back().push_back(i-1);
              nelem++;
           }
           sub.back().push_back(i);
           nelem++;

           e_prev = e_cur;
        }
  }

  return nsub;
}


int orb_in_subspac(int orb_i, vector<vector<int> > sub)
{
  if(sub.empty()) {return -1;}

  int index=0;
  for(vector<vector<int> >::iterator it=sub.begin(); it != sub.end(); ++it)
  {  //loop over subspaces 
     if(find( (*it).begin(), (*it).end(), orb_i) != (*it).end()) {return index;}
     index++;
  }

  return -1;
}


bool orb_pair_in_same_subspac(int orb_i, int orb_j, vector<vector<int> > sub)
{
  if(sub.empty()) {return false;}
  for(vector<vector<int> >::iterator it=sub.begin(); it != sub.end(); ++it)
  {  //loop over subspaces 
     if(find( (*it).begin(), (*it).end(), orb_i) != (*it).end())
     {
        if(find( (*it).begin(), (*it).end(), orb_j) != (*it).end()) {return true;}
     }
  }
  return false;
}


void get_jt(double* jt, int orb_i, vector<vector<int> > sub, double* jCa, int NBas)
{
  //jt is accummulated!
  int index = orb_in_subspac(orb_i, sub);
  if(index >= 0){
    get_jt_deg(jt, sub, index, jCa, NBas);
  }
  else{
    const double D1 = 1.0;
    const int I1 = 1;
    dger_(&NBas, &NBas, &D1, jCa+orb_i*NBas, &I1, jCa+orb_i*NBas, &I1, jt, &NBas);
  }

}


void get_jt_deg(double* jt, vector<vector<int> > sub, int index, double* jCa, int NBas)
{
  //jt is accummulated!
  int Nd = sub[index].size();
  const int I1 = 1;
  double alpha = 1.0/Nd;
  for(vector<int>::iterator it=sub[index].begin(); it != sub[index].end(); ++it)
  {
     int i = *it;
     dger_(&NBas, &NBas, &alpha, jCa+i*NBas, &I1, jCa+i*NBas, &I1, jt, &NBas);
  }

}

#if defined __cplusplus
} // end extern "C"
#endif

/*
namespace py = pybind11;

PYBIND11_MODULE(libhess, m) {

    m.def("calc_hess_dm_fast_frac", &calc_hess_dm_fast_frac);
    m.def("calc_hess_dm_fast", &calc_hess_dm_fast);

}
*/
