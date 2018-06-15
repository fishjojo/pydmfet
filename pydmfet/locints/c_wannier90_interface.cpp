#include <iostream>
#include <complex>
#include <cstring>

using namespace std;

extern "C" {

void wannier_setup_(char* seedname,int* mp_grid,int* num_kpts,
     double* real_lattice,double* recip_lattice,double* kpt_latt,int* num_bands_tot, 
     int* num_atoms,char atom_symbols[][20],double* atoms_cart,int* gamma_only,int* spinors,  
     int* nntot,int* nnlist,int* nncell,int* num_bands,int* num_wann, 
     double* proj_site,int* proj_l,int* proj_m,int* proj_radial,double* proj_z, 
     double* proj_x,double* proj_zona,int* exclude_bands,int* proj_s,double* proj_s_qaxis);


void wannier_run_(char* seedname,int* mp_grid,int* num_kpts, 
     double* real_lattice,double* recip_lattice,double* kpt_latt,int* num_bands, 
     int* num_wann,int* nntot,int* num_atoms,char atom_symbols[][20], 
     double* atoms_cart,int* gamma_only,complex<double>* M_matrix,complex<double>* A_matrix,double* eigenvalues, 
     complex<double>* U_matrix,complex<double>* U_matrix_opt,int* lwindow,double* wann_centres, 
     double* wann_spreads,double* spread);



void string_c2f(char* fstring, size_t fstring_len,
                      const char* cstring)
{
    size_t inlen = strlen(cstring);
    size_t cpylen = min(inlen, fstring_len);

    if (inlen > fstring_len)
    {
	cout<<"string length greater than 50!!!!"<<endl;
    }

    copy(cstring, cstring + cpylen, fstring);
    fill(fstring + cpylen, fstring + fstring_len, ' ');
}

void real_array_c2f(double* fa, double* ca, int M, int N)
{
    int index = 0;
    for (int j=0; j<N; j++)
	for (int i=0; i<M; i++){
	    fa[index] = ca[i*N+j];
	    index++;
	}

    return;
}

void wannier90_setup(char* seedname,int* mp_grid,int num_kpts,
     double* real_lattice,double* recip_lattice,double* kpt_latt,int num_bands_tot,
     int num_atoms,char** atom_symbols,double* atoms_cart,int gamma_only,int spinors,
     int* nntot,int* nnlist,int* nncell,int* num_bands,int* num_wann,
     double* proj_site,int* proj_l,int* proj_m,int* proj_radial,double* proj_z,
     double* proj_x,double* proj_zona,int* exclude_bands,int* proj_s,double* proj_s_qaxis)
{

     int len_name = 50;
     char fseedname[len_name];
     string_c2f(fseedname,len_name,seedname);

     char fatom_symbols[num_atoms][20];
     for (int i=0; i<num_atoms; i++){
	string_c2f(fatom_symbols[i],20,atom_symbols[i]);
     }

     //double f_real_lattice[9];
     //double f_recip_lattice[9];
     //real_array_c2f(f_real_lattice, real_lattice, 3,3);
     //real_array_c2f(f_recip_lattice,recip_lattice,3,3);

     wannier_setup_(fseedname,mp_grid,&num_kpts,
     		    real_lattice,recip_lattice,kpt_latt,&num_bands_tot, 
     		    &num_atoms,fatom_symbols,atoms_cart,&gamma_only,&spinors,
     		    nntot,nnlist,nncell,num_bands,num_wann, 
     		    proj_site,proj_l,proj_m,proj_radial,proj_z, 
     		    proj_x,proj_zona,exclude_bands,proj_s,proj_s_qaxis);

     return;
}


void wannier90_run(char* seedname,int* mp_grid,int num_kpts,
     double* real_lattice,double* recip_lattice,double* kpt_latt,int num_bands,
     int num_wann,int nntot,int num_atoms,char** atom_symbols,
     double* atoms_cart,int gamma_only,complex<double>* M_matrix,complex<double>* A_matrix,double* eigenvalues,
     complex<double>* U_matrix,complex<double>* U_matrix_opt,int* lwindow,double* wann_centres,
     double* wann_spreads,double* spread)
{
     int len_name = 50;
     char fseedname[len_name];
     string_c2f(fseedname,len_name,seedname);

     char fatom_symbols[num_atoms][20];
     for (int i=0; i<num_atoms; i++){
        string_c2f(fatom_symbols[i],20,atom_symbols[i]);
     }

     wannier_run_(fseedname, mp_grid, &num_kpts,
                  real_lattice, recip_lattice, kpt_latt, &num_bands,
     	          &num_wann, &nntot, &num_atoms, fatom_symbols,
                  atoms_cart, &gamma_only, M_matrix, A_matrix, eigenvalues,
                  U_matrix, U_matrix_opt, lwindow, wann_centres,
     		  wann_spreads, spread);


     return;
}

}
