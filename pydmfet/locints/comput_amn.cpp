#include <iostream>
#include <omp.h>
#include <math.h>

using namespace std;
extern "C" {

/***************************************/
void vrload(double* a, double num, int n)
{
/*
  int i=0;
  for(i=0; i<n-3; i+=4){
    a[i] = num;
    a[i+1] = num;
    a[i+2] = num;
    a[i+3] = num;
  }
*/
  for(int j=0; j<n; j++){
    a[j] = num;
  }

  return;
}

void vradd(double* c,double* a,double* b,int n)
{
/*
  int i=0;
  for(i=0; i<n-3; i+=4){
    c[i] = a[i] + b[i];
    c[i+1] = a[i+1] + b[i+1];
    c[i+2] = a[i+2] + b[i+2];
    c[i+3] = a[i+3] + b[i+3];
  }
*/
  for(int j=0; j<n; j++){
    c[j] = a[j] + b[j];
  }

  return;
}

void vrsub(double* c,double* a,double* b,int n)
{
/*
  int i=0;
  for(i=0; i<n-3; i+=4){
    c[i] = a[i] - b[i];
    c[i+1] = a[i+1] - b[i+1];
    c[i+2] = a[i+2] - b[i+2];
    c[i+3] = a[i+3] - b[i+3];
  }
*/
  for(int j=0; j<n; j++){
    c[j] = a[j] - b[j];
  }

  return;
}

double vrnorm(double* a, int n)
{
    double enorm = 0.0;
    for(int i=0; i<n; i++)
	enorm += a[i]*a[i];

    enorm = sqrt(enorm);
    return enorm;
}

void vrscale(double* a, double num, int n)
{
    for(int i=0; i<n; i++)
	a[i] *= num;

    return;
}

double vrdot(double* a, double* b, int n)
{

    double out = 0.0;
    for(int i=0; i<n; i++)
	out += a[i] * b[i];

    return out;
}


void elemt_scale(double*c, double* a, int n)
{
/*
  int i=0;
  for(i=0; i<n-3; i+=4){
    c[i] *= a[i];
    c[i+1] *= a[i+1];
    c[i+2] *= a[i+2];
    c[i+3] *= a[i+3];
  }
*/
  for(int j=0; j<n; j++){
    c[j] *= a[j];
  }

  return;
}

void vrcross(double* c,double*a, double* b)
{

    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
    return;
}

/*************************************************/



void g_r_radial(double* radial, int r, double zona, double* r_rel, int ngs)
{

    double r_norm, fac;
    if(r==1)
    {
	for(int i=0; i<ngs; i++){
	    r_norm = vrnorm(r_rel+i*3, 3);
            fac = 2.0*pow(zona, 1.5);
            radial[i] = fac*exp(-1.0*zona*r_norm);
	}
    }
    else if(r==2)
    {
	for(int i=0; i<ngs; i++){
	    r_norm = vrnorm(r_rel+i*3, 3);
            fac = 0.5/sqrt(2.0)*pow(zona, 1.5);
            radial[i] = fac*(2.0-zona*r_norm)*exp(-0.5*zona*r_norm);
	}
    }
    else if(r==3)
    {
	for(int i=0; i<ngs; i++){
            r_norm = vrnorm(r_rel+i*3, 3);
            fac = sqrt(4.0/27.0)*pow(zona, 1.5);
            radial[i] = fac*(1.0-2.0/3.0*zona*r_norm+2.0/27.0*zona*zona*r_norm*r_norm)*exp(-1.0/3.0*zona*r_norm);
	}
    }
    else{
        cout<<"NYI"<<endl;
	throw exception();
    }

  return;
}

void cos_theta(double* out, double* r_rel,double* zaxis, int ngs)
{

    double z_norm = vrnorm(zaxis,3);
    double dot,ab;
    for(int i=0;i<ngs;i++){
        dot = vrdot(r_rel+i*3,zaxis,3);
        ab = vrnorm(r_rel+i*3,3) * z_norm;
        if(ab == 0.0)
            out[i] = 0.0;
        else
            out[i] = dot/ab;
    }

    return;
}


void sin_theta(double* out, double* r_rel,double* zaxis, int ngs)
{

    double *cos = (double*)malloc(sizeof(double)*ngs);
    cos_theta(cos,r_rel,zaxis,ngs);

    for(int i=0; i<ngs; i++)
	out[i] = sqrt(1.0 - cos[i]*cos[i]);

    free(cos);
    return;
}

void sin_cos_phi(double* sin_phi,double* cos_phi,
		 double* r_rel,double* xy_perp,double* xaxis,double* yaxis,int ngs)
{
    double x_norm = vrnorm(xaxis,3);
    double tmp[3],dot,ab,r_proj_xy[3];

    for(int i=0;i<ngs;i++)
    {
	vrcross(tmp,xy_perp, r_rel+i*3);
        vrcross(r_proj_xy, tmp, xy_perp);
        dot = vrdot(r_proj_xy, xaxis,3);
        ab = vrnorm(r_proj_xy,3) * x_norm;
        if(ab == 0.0)
            cos_phi[i] = 0.0;
        else
            cos_phi[i] = dot/ab;

        sin_phi[i] = sqrt(1.0-cos_phi[i]*cos_phi[i]);
        dot = vrdot(r_proj_xy, yaxis,3);
        if(dot < 0.0)
            sin_phi[i] *= -1.0;
    }

    return;
}


void g_r_ang(double* ang,int l,int m,double* r_rel,double* zaxis,double* xaxis,int ngs)
{

    if(l==0){
	vrload(ang, 1.0/sqrt(4.0*M_PI), ngs);
        return;
    }

    double yaxis[3],xy_perp[3];
    vrcross(yaxis,zaxis,xaxis);
    double norm = vrnorm(yaxis,3);
    vrscale(yaxis,1.0/norm, 3);
    vrcross(xy_perp,xaxis,yaxis);

    if(l==1)
    {
        double fac = sqrt(0.75*M_PI);
        if(m==1){
	    cos_theta(ang,r_rel,zaxis,ngs);
	}
        else if(m==2 || m==3){
	    double *sin_phi = (double*)malloc(sizeof(double)*ngs);
	    double *cos_phi = (double*)malloc(sizeof(double)*ngs);
	    sin_cos_phi(sin_phi,cos_phi,r_rel,xy_perp,xaxis,yaxis,ngs);
	    sin_theta(ang,r_rel,zaxis,ngs);

	    if(m==2)
	        elemt_scale(ang,cos_phi,ngs);
	    else if(m==3)
	        elemt_scale(ang,sin_phi,ngs);
	    else
		throw exception();

	    free(sin_phi);
	    free(cos_phi);
	}
	else{
	    cout<<"wrong lm"<<endl;
	    throw exception();
	}

	vrscale(ang,fac,ngs);
    }
    else{
        cout<<"NYI"<<endl;
	throw exception();
    }

    return;
}


void batch_comput_g_r(double* g_r, double* coords, int ngrid, int nbands, 
		      double* R, int* lmr, double* zaxis, double* xaxis, double* zona)
{

    double *r_rel = (double*)malloc(sizeof(double)*ngrid*3);
    double *ang = (double*)malloc(sizeof(double)*ngrid);

    for(int i=0; i<nbands; i++)
    {
	double *radial = g_r + i*ngrid;

        int l=lmr[i*3];
        int m=lmr[i*3+1];
        int r=lmr[i*3+2];
        double *zax = zaxis+i*3;
        double *xax = xaxis+i*3;

	for(int igrid=0; igrid<ngrid; igrid++)
        {   vrsub(r_rel+igrid*3, coords+igrid*3, R+i*3, 3);}

        g_r_radial(radial, r,zona[i],r_rel, ngrid);
        if(l<0)
	{
            if(l == -1)
	    {
		double* s = (double*)malloc(sizeof(double)*ngrid);
		double* px = (double*)malloc(sizeof(double)*ngrid);
                g_r_ang(s,0,1,r_rel,zax,xax,ngrid);
                g_r_ang(px,1,2,r_rel,zax,xax,ngrid);
                double fac = 1.0/sqrt(2.0);
                if(m == 1){
		    vradd(ang,s,px,ngrid);
		    vrscale(ang,fac,ngrid);
		}
                else if(m == 2){
		    vrsub(ang,s,px,ngrid);
		    vrscale(ang,fac,ngrid);
                }
		free(s);
		free(px);
	    }
            else if(l == -2)
	    {
		double* s = (double*)malloc(sizeof(double)*ngrid);
                double* px = (double*)malloc(sizeof(double)*ngrid);
                g_r_ang(s,0,1,r_rel,zax,xax,ngrid);
		vrscale(s, 1.0/sqrt(3.0),ngrid);

                g_r_ang(px,1,2,r_rel,zax,xax,ngrid);

                if(m == 1){
		    double* py = (double*)malloc(sizeof(double)*ngrid);
                    g_r_ang(py,1,3,r_rel,zax,xax,ngrid);
		    vrscale(px, -1.0/sqrt(6.0), ngrid);
		    vrscale(py, 1.0/sqrt(2.0), ngrid);
		    vradd(ang,s,px,ngrid);
		    vradd(ang,ang,py,ngrid);
		    free(py);
		}
                else if(m == 2){
		    double* py = (double*)malloc(sizeof(double)*ngrid);
                    g_r_ang(py,1,3,r_rel,zax,xax,ngrid);
		    vrscale(px, -1.0/sqrt(6.0), ngrid);
		    vrscale(py, -1.0/sqrt(2.0), ngrid);
		    vradd(ang,s,px,ngrid);
                    vradd(ang,ang,py,ngrid);
                    free(py);
		}
                else if(m == 3){
		    vrscale(px, 2.0/sqrt(6.0), ngrid);
		    vradd(ang,s,px,ngrid);
		}
		free(s); free(px);
	    }
            else if(l == -3)
	    {
		double* s = (double*)malloc(sizeof(double)*ngrid);
                double* px = (double*)malloc(sizeof(double)*ngrid);
		double* py = (double*)malloc(sizeof(double)*ngrid);
                double* pz = (double*)malloc(sizeof(double)*ngrid);
                g_r_ang(s,0,1,r_rel,zax,xax,ngrid);
                g_r_ang(px,1,2,r_rel,zax,xax,ngrid);
                g_r_ang(py,1,3,r_rel,zax,xax,ngrid);
                g_r_ang(pz,1,1,r_rel,zax,xax,ngrid);

                if(m == 1){
		    vradd(ang,s,px,ngrid);
		    vradd(ang,ang,py,ngrid);
		    vradd(ang,ang,pz,ngrid);
		}
                else if(m == 2){
		    vradd(ang,s,px,ngrid);
		    vrsub(ang,ang,py,ngrid);
		    vrsub(ang,ang,pz,ngrid);
		}
                else if(m == 3){
		    vrsub(ang,s,px,ngrid);
		    vradd(ang,ang,py,ngrid);
		    vrsub(ang,ang,pz,ngrid);
		}
                else if(m == 4){
		    vrsub(ang,s,px,ngrid);
		    vrsub(ang,ang,py,ngrid);
                    vradd(ang,ang,pz,ngrid);
		}

		vrscale(ang,0.5,ngrid);
	    }
            else
	    {
                cout<<"NYI\n"<<endl;
		throw exception();
	    }
	}
        else
	{
            g_r_ang(ang,l,m,r_rel,zax,xax,ngrid);
	}

	elemt_scale(radial,ang,ngrid);
    }

    free(r_rel); free(ang);

    return;
}


void comput_g_r(int blk_size, int nblks,double* g_r, double* coords, int ngs, int nbands,
                double* Rc,int* lmr, double* zaxis, double* xaxis,double* zona)
{

#pragma omp parallel
{
    #pragma omp for schedule(dynamic)
    for(int ibat=0; ibat<nblks; ibat++)
    {
	int batch_ngrid = blk_size;
	if(ibat == nblks-1) batch_ngrid = ngs-(nblks-1)*blk_size; 

	double *batch_coords = coords + ibat*blk_size*3;
	double *batch_g_r = g_r + ibat*nbands*blk_size;
	batch_comput_g_r(batch_g_r, batch_coords, batch_ngrid, nbands,
                 Rc, lmr, zaxis, xaxis, zona);

    }
}


}

}
