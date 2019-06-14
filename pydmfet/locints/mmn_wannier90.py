import os
from pyscf import lib,pbc
from pyscf.lib import param
from pyscf.pbc.lib.kpts_helper import gamma_point
import numpy as np
import ctypes
import math
from multiprocessing import Pool
import time,gc
from pydmfet import tools

lib_wannier90 = np.ctypeslib.load_library('libwannier90',  os.path.dirname(__file__))
libmisc = np.ctypeslib.load_library('libmisc', os.path.dirname(__file__))
num_nnmax = 12

sqrt = math.sqrt


def assign_wf2atm(wf,filename=None,num_wf=None,natom=None):

    if filename is None: filename = wf.seedname+'_centres.xyz'
    if num_wf is None: num_wf = wf.num_wann
    if natom is None: natom = wf.num_atoms

    wf_c = np.zeros((num_wf,3))
    atm_cart = np.zeros((natom,3))
    with open( filename, 'r' ) as f:
        tmp = next(f)
        tmp = next(f)
        for i in range(num_wf):
            a,x,y,z = [tmp for tmp in next(f).split()]
            wf_c[i,0] = float(x)
            wf_c[i,1] = float(y)
            wf_c[i,2] = float(z)
        for i in range(natom):
            a,x,y,z = [tmp for tmp in next(f).split()]
            atm_cart[i,0] = float(x)
            atm_cart[i,1] = float(y)
            atm_cart[i,2] = float(z)

    dist_table = np.zeros((natom,num_wf))
    for i in range(natom):
        for j in range(num_wf):
            dist_table[i,j] = np.linalg.norm(atm_cart[i] - wf_c[j])

    #print dist_table
    aoslice = wf.cell.aoslice_by_atom()
    assign_table = []
    assign_list = []
    for i in range(natom):
        seq = dist_table[i].argsort()
        nao = aoslice[i,3] - aoslice[i,2]
        assign_table.append(seq[:nao])
        for j in range(nao):
            assign_list.append(seq[j])

    print (assign_table)
    print (assign_list)

    for i in range(num_wf):
        exist = False
        for j in range(num_wf):
            if(assign_list[j] == i):
                exist = True

        assert(exist == True)

    return assign_list


def read_umat(wannier,filename):

    nk = 0
    nw = 0
    nw1 = 0
    kpts = None
    umat = None
    with open( filename, 'r' ) as f:
        tmp = next(f)
        nk,nw,nw1 = [int(x) for x in next(f).split()]
        tmp = next(f)
        kpts = np.zeros((nk,3))
        umat = np.zeros((nk,nw,nw), dtype = np.complex128)
        for k in range(nk):
            kpts[k] = [float(x) for x in next(f).split()]
            for j in range(nw):
                for i in range(nw):
                    u_r, u_i = [float(x) for x in next(f).split()]
                    umat[k,i,j] = u_r + 1j*u_i
            if(k<nk-1):
                tmp = next(f)

    return umat

def write_eig(wannier, filename, nband=None):

    if nband is None: nband = wannier.nband
    mf = wannier.mf
    cell = wannier.cell
    ene = mf.mo_energy

    nk = 1
    cof_hartree2ev = param.HARTREE2EV
    with open( filename, 'w' ) as fout:
        for k in range(nk):
            for i in range(nband):
                fout.write('%d  %d  %16.10f\n' % (i+1, k+1, ene[i]*cof_hartree2ev) )


def write_bloch_u_r(wannier, bands = None, nband=None, kpts = np.zeros((1,3))):

    if bands is None: bands = wannier.bands
    if nband is None: nband = wannier.nband
    cell = wannier.cell
    gs = cell.gs
    ngx, ngy, ngz = 2*np.asarray(gs)+1

    coords_zyx = cell.gen_uniform_grids(gs)
    ngs = len(coords_zyx)

    assert(ngs == ngx*ngy*ngz)

    coords = np.zeros((ngs,3))
    index = 0
    for ix in range(ngx):
        for iy in range(ngy):
            for iz in range(ngz):
                coords[ix+iy*ngx+iz*ngx*ngy] = coords_zyx[index]
                index += 1


    mf = wannier.mf
    mo_coeff = mf.mo_coeff
    nk = len(kpts)
    #nao = cell.nao_nr()
    #mo_coeff = mf.mo_coeff.reshape((nk,nao,nao))

    mydf = mf.with_df
    ni = mydf._numint
    aoR = ni.eval_ao(cell, coords, kpts, non0tab=None)

    phase_k = np.zeros((nk,ngs), dtype = np.complex128)
    for i in range(nk):
        for j in range(ngs):
            kr = np.dot(kpts[i], coords[j])
            phase_k[i,j] = np.exp(-1j*kr)

    u_r = np.zeros((nk,nband,ngs), dtype = np.complex128)
    for k, aoR_k in enumerate(aoR):
        u_r[k] = lib.dot(mo_coeff[:,bands].T,aoR_k.T*phase_k[k])

    #bohr2ang = math.pow(param.BOHR, -1.5)
    #u_r *= bohr2ang

    for k in range(nk):
        filename = 'UNK'+'{:05}'.format(k+1)+'.1'
        with open( filename, 'w' ) as fout:
            fout.write('%d %d %d %d %d\n' % (ngx,ngy,ngz,k+1,nband) )
            for i in range(nband):
                for j in range(ngs):
                    fout.write('%16.10f   %16.10f\n' % (u_r[k,i,j].real, u_r[k,i,j].imag) )

def calc_iso_cutoff(wannier, umat=None, bands = None, nband=None, kpts=np.zeros((1,3))):

    if bands is None: bands = wannier.bands
    if nband is None: nband = wannier.nband
    if umat  is None: umat = wannier.u_mat
    cell = wannier.cell
    gs = cell.gs
    ngx, ngy, ngz = 2*np.asarray(gs)+1

    coords_zyx = cell.gen_uniform_grids(gs)
    ngs = len(coords_zyx)

    assert(ngs == ngx*ngy*ngz)

    coords = coords_zyx

    mf = wannier.mf
    mo_coeff = mf.mo_coeff
    nk = len(kpts)
    #nao = cell.nao_nr()
    #mo_coeff = mf.mo_coeff.reshape((nk,nao,nao))

    mydf = mf.with_df
    ni = mydf._numint

    aoR = ni.eval_ao(cell, coords, kpts, non0tab=None)

    u_r = np.zeros((nband,ngs), dtype = np.complex128)
    wf_coeff = np.dot(mf.mo_coeff,umat[0].real)
    #gamma point k = 0
    for k, aoR_k in enumerate(aoR):
        u_r = lib.dot(wf_coeff[:,bands].T,aoR_k.T)

    weight = cell.vol / ngs

    thresh = 0.8
    c =  np.zeros((nband))
    for i in range(nband):
            u2_r = u_r[i] * u_r[i].conj()
            u2_r = u2_r.real
            nelec = weight * u2_r.sum()
            if abs(nelec - 1.0)>0.0001:
                print ('k=',k,'  iband=',i, '  nelec=',nelec)

            u2_r *= -1.0
            u2_r.sort()
            u2_r *= -1.0
            rho = 0.0
            index  = -1
            for j in range(ngs):
                rho += weight * u2_r[j]
                if(rho > thresh):
                    index = j
                    break

            c[i] = sqrt(0.5*(u2_r[j] + u2_r[j-1]))
            print (c[i])


    
def get_nnkpts(nntot, nnlist, nncell):

    nk = nnlist.shape[0]
    nnkpts = np.zeros((nk,nntot,5), dtype=np.int32)   

    for k in range(nk):
        for i in range(nntot):
            nnkpts[k,i,0] = k+1
            nnkpts[k,i,1] = nnlist[k,i]
            nnkpts[k,i,2:5] = nncell[0:3,k,i] 

    return nnkpts

def get_bpts(wannier, nnkpts=None,  kpts = np.zeros((1,3)) ):

    if nnkpts is None: nnkpts = wannier.nnkpts
    nk = len(kpts)
    nb = nnkpts.shape[1]

    cell = wannier.cell
    b = cell.reciprocal_vectors()

    bpts = np.zeros((nk,nb,3))
    for k in range(nk):
        for i in range(nb):
            kk = nnkpts[k,i,0]-1
            k2 = nnkpts[k,i,1]-1
            kb = kpts[k2] + np.dot(nnkpts[k,i,2:5], b) 
            bpts[k,i,:] = kb - kpts[kk]

    return bpts


def write_Amn(wannier, filename, nk, nband = None, Amn = None):

    if Amn is None: Amn = wannier.Amn
    if nband is None: nband = wannier.nband

    with open( filename, 'w' ) as fout:
        fout.write('Amn file\n')
        fout.write('%d   %d   %d\n' % (nband, nk, nband))

        for k in range(nk):
            for n in range(nband):
                for m in range(nband):
                    fout.write('%d  %d  %d  %16.10f   %16.10f\n' % (m+1, n+1, k+1, Amn[k,m,n].real, Amn[k,m,n].imag) )



def write_Mmn(wannier, filename, nk, nband = None, Mmn = None, nnkpts=None):

    if Mmn is None: Mmn = wannier.Mmn
    if nnkpts is None: nnkpts = wannier.nnkpts
    if nband is None: nband = wannier.nband

    nntot = nnkpts.shape[1]

    with open( filename, 'w' ) as fout:

        fout.write('Mmn file\n')
        fout.write('%d   %d   %d\n' % (nband, nk, nntot))

        for k in range(nk):
          for i in range(nntot):
            fout.write('%d %d %d %d %d\n' % tuple(nnkpts[k,i]) )
            for n in range(nband):
                for m in range(nband):
                    fout.write('%16.10f   %16.10f\n' % (Mmn[k,i,m,n].real, Mmn[k,i,m,n].imag) ) 


def comput_Mmn(wannier, bands = None, nband = None, bpts=None, kpts = np.zeros((1,3)) ):

    #only works for Gamma point

    if bands is None: bands = wannier.bands
    if nband is None: nband = wannier.nband
    if bpts is None: bpts = wannier.bpts
    mf = wannier.mf
    mo_coeff = mf.mo_coeff

    cell = wannier.cell
    mydf = mf.with_df
    gs = mydf.gs

    coords = cell.gen_uniform_grids(gs)
    ngs = len(coords)

    ni = mydf._numint

    weight = cell.vol / ngs

    aoR = ni.eval_ao(cell, coords, kpts, non0tab=None)

    
    nk = len(kpts) 
    nntot = bpts.shape[1]
    phase_b = np.zeros((nk,nntot,ngs), dtype = np.complex128)
    for k in range(nk):
        for i in range(nntot):
            for j in range(ngs):
                br = np.dot(bpts[k,i], coords[j])
                phase_b[k,i,j] = np.exp(-1j*br)

    nbas = cell.nao_nr()
    Mmunu = np.zeros((nk,nntot,nbas,nbas), dtype = np.complex128)
    Mmn = np.zeros((nk,nntot,nband,nband), dtype = np.complex128)
    for k, aoR_k in enumerate(aoR):
        for i in range(nntot):
            Mmunu[k,i] = weight * lib.dot(aoR_k.T.conj()*phase_b[k,i], aoR_k)

    for k in range(nk):
        for i in range(nntot):
            Mmn[k,i] = lib.dot(lib.dot(mo_coeff[:,bands].T.conj(), Mmunu[k,i]), mo_coeff[:,bands])

    return Mmn

def read_proj_data(filename):

    nband = 0
    with open( filename, 'r' ) as f:
        tmp = next(f)
        nband = int(next(f))
        R = np.zeros((nband,3))
        lmr = np.zeros((nband,3),dtype=np.int32)
        zaxis = np.zeros((nband,3))
        xaxis = np.zeros((nband,3))
        zona = np.zeros(nband)
        for i in range(nband):
            tmp = [float(x) for x in next(f).split()]
            R[i,0:3] = tmp[0:3]
            lmr[i,0:3] = [int(x) for x in tmp[3:6]]
            tmp =  [float(x) for x in next(f).split()]
            zaxis[i,0:3] = tmp[0:3]
            xaxis[i,0:3] = tmp[3:6]
            zona[i] = tmp[6]
            zona[i] *= param.BOHR

        return (R,lmr,zaxis,xaxis,zona) 


def comput_g_r_para(i, R,lmr,zaxis,xaxis,zona,coords):

        ngs = len(coords)
        g_r = np.zeros((ngs))

        l=lmr[i,0]
        m=lmr[i,1]
        r=lmr[i,2]
        zax = zaxis[i]
        xax = xaxis[i]  

        r_rel = coords - R[i]
        radial = g_r_radial(r,zona[i],r_rel)
        ang = None
        if(l<0):
            if(l == -1):
                s = g_r_ang(0,1,r_rel,zax,xax)
                px = g_r_ang(1,2,r_rel,zax,xax)
                fac = 1.0/sqrt(2.0)
                if(m == 1):
                    ang = fac*(s+px)
                elif(m == 2):
                    ang = fac*(s-px)

            elif(l == -2):
                s = g_r_ang(0,1,r_rel,zax,xax)
                px = g_r_ang(1,2,r_rel,zax,xax)
                if(m == 1):
                    py = g_r_ang(1,3,r_rel,zax,xax)
                    ang = 1.0/sqrt(3.0)*s - 1.0/sqrt(6.0)*px + 1.0/sqrt(2.0)*py
                elif(m == 2):
                    py = g_r_ang(1,3,r_rel,zax,xax)
                    ang = 1.0/sqrt(3.0)*s - 1.0/sqrt(6.0)*px - 1.0/sqrt(2.0)*py
                elif(m == 3):
                    ang = 1.0/sqrt(3.0)*s + 2.0/sqrt(6.0)*px

            elif(l == -3):
                s = g_r_ang(0,1,r_rel,zax,xax)
                px = g_r_ang(1,2,r_rel,zax,xax)
                py = g_r_ang(1,3,r_rel,zax,xax)
                pz = g_r_ang(1,1,r_rel,zax,xax)
                
                if(m == 1):
                    ang = 0.5*(s+px+py+pz)
                elif(m == 2):
                    ang = 0.5*(s+px-py-pz)
                elif(m == 3):
                    ang = 0.5*(s-px+py-pz)
                elif(m == 4):
                    ang = 0.5*(s-px-py+pz)

            else:
                raise Exception("NYI")
        else:
            ang = g_r_ang(l,m,r_rel,zax,xax)

        g_r = radial * ang

        del radial, ang, r_rel
        return g_r



def comput_g_r(R,lmr,zaxis,xaxis,zona,nband,coords):

    ngs = len(coords)
    g_r = np.zeros((nband,ngs))
   
    for i in range(nband):
        l=lmr[i,0]
        m=lmr[i,1]
        r=lmr[i,2]
        zax = zaxis[i]
        xax = xaxis[i]  

        r_rel = coords - R[i]
        radial = g_r_radial(r,zona[i],r_rel)
        ang = None
        if(l<0):
            if(l == -1):
                s = g_r_ang(0,1,r_rel,zax,xax)
                px = g_r_ang(1,2,r_rel,zax,xax)
                fac = 1.0/sqrt(2.0)
                if(m == 1):
                    ang = fac*(s+px)
                elif(m == 2):
                    ang = fac*(s-px)

            elif(l == -2):
                s = g_r_ang(0,1,r_rel,zax,xax)
                px = g_r_ang(1,2,r_rel,zax,xax)
                if(m == 1):
                    py = g_r_ang(1,3,r_rel,zax,xax)
                    ang = 1.0/sqrt(3.0)*s - 1.0/sqrt(6.0)*px + 1.0/sqrt(2.0)*py
                elif(m == 2):
                    py = g_r_ang(1,3,r_rel,zax,xax)
                    ang = 1.0/sqrt(3.0)*s - 1.0/sqrt(6.0)*px - 1.0/sqrt(2.0)*py
                elif(m == 3):
                    ang = 1.0/sqrt(3.0)*s + 2.0/sqrt(6.0)*px

            elif(l == -3):
                s = g_r_ang(0,1,r_rel,zax,xax)
                px = g_r_ang(1,2,r_rel,zax,xax)
                py = g_r_ang(1,3,r_rel,zax,xax)
                pz = g_r_ang(1,1,r_rel,zax,xax)
                
                if(m == 1):
                    ang = 0.5*(s+px+py+pz)
                elif(m == 2):
                    ang = 0.5*(s+px-py-pz)
                elif(m == 3):
                    ang = 0.5*(s-px+py-pz)
                elif(m == 4):
                    ang = 0.5*(s-px-py+pz)

            else:
                raise Exception("NYI")
        else:
            ang = g_r_ang(l,m,r_rel,zax,xax)

        g_r[i] = radial * ang

    return g_r

def g_r_ang(l,m,r_rel,zaxis,xaxis):

    ngs = len(r_rel)
    ang = np.zeros(ngs)
    if(l==0):
        ang[:] = 1.0/sqrt(4.0*np.pi)
        return ang

    yaxis = np.cross(zaxis,xaxis)
    yaxis = yaxis*(1.0/np.linalg.norm(yaxis))
    xy_perp = np.cross(xaxis,yaxis)

    if(l==1):
        fac = sqrt(0.75*np.pi)
        if(m==1):
            ang = fac*cos_theta(r_rel,zaxis)
        elif(m==2):
            ang = fac*sin_theta(r_rel,zaxis)*cos_phi(r_rel,xy_perp,xaxis,yaxis)
        elif(m==3):
            ang = fac*sin_theta(r_rel,zaxis)*sin_phi(r_rel,xy_perp,xaxis,yaxis)
    else:
        raise Exception("NYI")

    return ang

def sin_phi(r_rel,xy_perp,xaxis,yaxis):

    return sin_cos_phi(r_rel,xy_perp,xaxis,yaxis)[0]

def cos_phi(r_rel,xy_perp,xaxis,yaxis):

    return sin_cos_phi(r_rel,xy_perp,xaxis,yaxis)[1]

def sin_cos_phi(r_rel,xy_perp,xaxis,yaxis):

    ngs = len(r_rel)
    sin_phi = np.zeros(ngs)
    cos_phi = np.zeros(ngs)
    x_norm = np.linalg.norm(xaxis)

    for i in range(ngs):
        r_proj_xy = np.cross(np.cross(xy_perp, r_rel[i]), xy_perp)
        dot = np.dot(r_proj_xy, xaxis)
        ab = np.linalg.norm(r_proj_xy) * x_norm
        if(ab == 0.0):
            cos_phi[i] = 0.0
        else:
            cos_phi[i] = dot/ab

        sin_phi[i] = sqrt(1.0-cos_phi[i]**2)
        dot = np.dot(r_proj_xy, yaxis)
        if(dot < 0.0):
            sin_phi[i] *= -1.0

    return(sin_phi, cos_phi)


def sin_theta(r_rel,zaxis):

    cos = cos_theta(r_rel,zaxis)
    sin_theta = np.sqrt(1.0-np.square(cos))
    del cos
    return sin_theta

def cos_theta(r_rel,zaxis):

    ngs = len(r_rel)
    cos_theta = np.zeros(ngs)
    z_norm = np.linalg.norm(zaxis)

    for i in range(ngs):
        dot = np.dot(r_rel[i],zaxis)
        ab = np.linalg.norm(r_rel[i]) * z_norm
        if(ab == 0.0):
            cos_theta[i] = 0.0
        else:
            cos_theta[i] = dot/ab

    return cos_theta


def g_r_radial(r,zona,r_rel):

    ngs = len(r_rel)
    radial = np.zeros(ngs)

    r_norm = np.zeros(ngs)
    for i in range(ngs):
        r_norm[i] = np.linalg.norm(r_rel[i])

    if(r==1):
        fac = 2.0*math.pow(zona, 1.5)
        radial = fac*np.exp(-1.0*zona*r_norm)
    elif(r==2):
        fac = 0.5/sqrt(2.0)*math.pow(zona, 1.5)
        radial = fac*(2.0-zona*r_norm)*np.exp(-0.5*zona*r_norm)
    elif(r==3):
        fac = sqrt(4.0/27.0)*math.pow(zona, 1.5)
        radial = fac*(1.0-2.0/3.0*zona*r_norm+2.0/27.0*zona**2*np.square(r_norm))*np.exp(-1.0/3.0*zona*r_norm)
    else:
        raise Exception("NYI")

    del r_norm
    return radial

def comput_Amn(wf, Rc=None,lmr=None,zaxis=None,xaxis=None,zona=None, bands = None, nband = None, kpts = np.zeros((1,3)),max_memory=None):

    if Rc is None: Rc = wf.Rc
    if lmr is None: lmr = wf.lmr
    if zaxis is None: zaxis = wf.zaxis
    if xaxis is None: xaxis = wf.xaxis
    if zona is None: zona = wf.zona
    if bands is None: bands = wf.bands
    if nband is None: nband = wf.nband
    if max_memory is None: max_memory = wf.max_memory

    nk = len(kpts)
    mf = wf.mf
    mo_coeff = mf.mo_coeff

    cell = wf.cell
    mydf = mf.with_df
    gs = mydf.gs

    coords = cell.gen_uniform_grids(gs)
    ngs = len(coords)

    ni = mydf._numint

    weight = cell.vol / ngs

    aoR = ni.eval_ao(cell, coords, kpts, non0tab=None)

    a = cell.lattice_vectors()
    R = np.dot(Rc,a)
    #g_r = comput_g_r(R,lmr,zaxis,xaxis,zona,nband,coords)
    
    #parallel block 
    g_r = np.ndarray((nband*ngs))

    from pyscf import lib
    max_mem = wf.max_memory - lib.current_memory()[0]
    print ('available mem (Mb) = ', max_mem)

    blk_size = min(ngs,(max_mem*1e6/8-nband*ngs*3)/12/16)
    nblks = ngs//blk_size
    if(nblks <= 1):
        nblks = lib.num_threads()
    blk_size = ngs//nblks
    #g_r_blks = np.concatenate(tuple[g_r_split[i] for i in range(nblks)],axis=0)


    libmisc.comput_g_r(ctypes.c_int(blk_size), ctypes.c_int(nblks), \
                       g_r.ctypes.data_as(ctypes.c_void_p), coords.ctypes.data_as(ctypes.c_void_p),ctypes.c_int(ngs), ctypes.c_int(nband), \
                       R.ctypes.data_as(ctypes.c_void_p),lmr.ctypes.data_as(ctypes.c_void_p), zaxis.ctypes.data_as(ctypes.c_void_p), \
                       xaxis.ctypes.data_as(ctypes.c_void_p),zona.ctypes.data_as(ctypes.c_void_p))


    g_r_split = np.split(g_r,[i*nband*blk_size for i in range(1,nblks)])
    for i in range(nblks):
        g_r_split[i] = np.reshape(g_r_split[i],(nband,-1))

    g_r = np.concatenate([g_r_split[i] for i in range(nblks)],axis=1)


    
    '''
    pool = Pool()
    results = [pool.apply_async(comput_g_r_para, (i, R,lmr,zaxis,xaxis,zona,coords,)) for i in range(nband) ]
    pool.close()
    pool.join()
    for i in range(nband):
        g_r[i] = results[i].get()
    '''
    '''
    #assume 8 proc
    blk_size = ngs//8
    tmp = np.split(coords,[blk_size,blk_size*2,blk_size*3,blk_size*4,blk_size*5,blk_size*6,blk_size*7])
    for i in range(nband):
        pool = Pool(8,maxtasksperchild=1)
        results = [pool.apply_async(comput_g_r_para, (i, R,lmr,zaxis,xaxis,zona,coords_blk,)) for blk,coords_blk in enumerate(tmp) ]
        pool.close()
        pool.join()
        for blk in range(8):
            index1 = blk*blk_size
            index2 = index1+blk_size
            if(blk == 7):
                index2 = ngs
            g_r[i][index1:index2] = results[blk].get()

    del tmp
    #end parallel block
    '''   

    print ('available mem (Mb) = ', wf.max_memory - lib.current_memory()[0])
 
    nao = cell.nao_nr()
    Amun = np.zeros((nk,nao,nband), dtype = np.complex128)
    Amn = np.zeros((nk,nband,nband), dtype = np.complex128)
    for k, aoR_k in enumerate(aoR):
        Amun[k] = weight * lib.dot(aoR_k.T.conj(), g_r.T)

    for k in range(nk):
        Amn[k] = lib.dot(mo_coeff[:,bands].T.conj(), Amun[k])

    return Amn


def wannier90_setup(wf, seedname=None, mp_grid_dim=None, num_kpts=None, real_lattice=None, recip_lattice=None, \
                    kpt_latt=None, num_bands_tot=None, num_atoms=None, atom_symbols=None, atoms_cart=None, \
                    gamma_only=None, spinors=False):

    if seedname is None: seedname = wf.seedname
    if mp_grid_dim is None: mp_grid_dim = wf.mp_grid_dim
    if num_kpts is None: num_kpts = wf.num_kpts
    if real_lattice is None: real_lattice = wf.real_lattice
    if recip_lattice is None: recip_lattice = wf.recip_lattice
    if kpt_latt is None: kpt_latt = wf.kpt_latt
    if num_bands_tot is None: num_bands_tot = wf.num_bands_tot
    if num_atoms is None: num_atoms = wf.num_atoms
    if atom_symbols is None: atom_symbols = wf.atom_symbols
    if atoms_cart is None: atoms_cart = wf.atoms_cart
    if gamma_only is None: gamma_only = wf.gamma_only

    real_lattice = real_lattice * param.BOHR  #unit: Angstrom
    recip_lattice = recip_lattice / param.BOHR #unit: 1/Angstrom
    real_lattice_T = real_lattice.T
    recip_lattice_T = recip_lattice.T

    atoms_cart = atoms_cart * param.BOHR #unit: Angstrom

    #output
    nnlist = np.ndarray((num_kpts,num_nnmax), dtype = np.int32, order='F')
    nncell = np.ndarray((3,num_kpts,num_nnmax), dtype = np.int32, order='F')
    nntot = np.ndarray((1), dtype = np.int32)
    num_bands = np.ndarray((1), dtype = np.int32)
    num_wann = np.ndarray((1), dtype = np.int32)
    proj_site = np.ndarray((num_bands_tot,3))
    proj_l = np.ndarray((num_bands_tot), dtype = np.int32)
    proj_m = np.ndarray((num_bands_tot), dtype = np.int32)
    proj_radial = np.ndarray((num_bands_tot), dtype = np.int32)
    proj_z = np.ndarray((num_bands_tot,3))
    proj_x = np.ndarray((num_bands_tot,3))
    proj_zona = np.ndarray((num_bands_tot))
    exclude_bands = np.ndarray((num_bands_tot), dtype = np.int32)
    proj_s = np.ndarray((num_bands_tot), dtype = np.int32)
    proj_s_qaxis = np.ndarray((num_bands_tot,3))

    arr = (ctypes.c_char_p * num_atoms)()
    arr[:] = atom_symbols
    #fn_setup = getattr(lib_wannier90, 'wannier90_setup')
    lib_wannier90.wannier90_setup(ctypes.c_char_p(seedname), mp_grid_dim.ctypes.data_as(ctypes.c_void_p),ctypes.c_int(num_kpts), \
             real_lattice_T.ctypes.data_as(ctypes.c_void_p),recip_lattice_T.ctypes.data_as(ctypes.c_void_p), \
             kpt_latt.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(num_bands_tot), \
             ctypes.c_int(num_atoms), arr, atoms_cart.ctypes.data_as(ctypes.c_void_p), \
             ctypes.c_int(gamma_only), ctypes.c_int(spinors), \
             nntot.ctypes.data_as(ctypes.c_void_p),nnlist.ctypes.data_as(ctypes.c_void_p),\
             nncell.ctypes.data_as(ctypes.c_void_p),num_bands.ctypes.data_as(ctypes.c_void_p),num_wann.ctypes.data_as(ctypes.c_void_p), \
             proj_site.ctypes.data_as(ctypes.c_void_p),proj_l.ctypes.data_as(ctypes.c_void_p),proj_m.ctypes.data_as(ctypes.c_void_p), \
             proj_radial.ctypes.data_as(ctypes.c_void_p), proj_z.ctypes.data_as(ctypes.c_void_p), \
             proj_x.ctypes.data_as(ctypes.c_void_p),proj_zona.ctypes.data_as(ctypes.c_void_p), \
             exclude_bands.ctypes.data_as(ctypes.c_void_p),proj_s.ctypes.data_as(ctypes.c_void_p), \
             proj_s_qaxis.ctypes.data_as(ctypes.c_void_p))

    nntot = nntot[0]
    num_bands = num_bands[0]
    num_wann = num_wann[0]

    proj_zona *= param.BOHR

    return (nnlist,nncell,nntot,num_bands,num_wann,\
            proj_site,proj_l,proj_m,proj_radial,\
            proj_z,proj_x,proj_zona,exclude_bands,\
            proj_s,proj_s_qaxis)


def wannier90_run(wf, Mmn=None, Amn=None, eigenvalues=None, \
                  seedname=None, mp_grid_dim=None, num_kpts=None, real_lattice=None, recip_lattice=None, \
                  kpt_latt=None, num_bands=None,num_wann=None,nntot=None,num_atoms=None, atom_symbols=None, atoms_cart=None, \
                  gamma_only=None):

    if Mmn is None: Mmn = wf.Mmn
    if Amn is None: Amn = wf.Amn
    if eigenvalues is None: eigenvalues = wf.eig

    if seedname is None: seedname = wf.seedname
    if mp_grid_dim is None: mp_grid_dim = wf.mp_grid_dim
    if num_kpts is None: num_kpts = wf.num_kpts
    if real_lattice is None: real_lattice = wf.real_lattice
    if recip_lattice is None: recip_lattice = wf.recip_lattice
    if kpt_latt is None: kpt_latt = wf.kpt_latt
    if num_bands is None: num_bands = wf.num_bands
    if num_wann is None: num_wann = wf.num_wann
    if nntot is None: nntot = wf.nntot
    if num_atoms is None: num_atoms = wf.num_atoms
    if atom_symbols is None: atom_symbols = wf.atom_symbols
    if atoms_cart is None: atoms_cart = wf.atoms_cart
    if gamma_only is None: gamma_only = wf.gamma_only

    real_lattice = real_lattice * param.BOHR  #unit: Angstrom
    recip_lattice = recip_lattice / param.BOHR #unit: 1/Angstrom
    real_lattice_T = real_lattice.T
    recip_lattice_T = recip_lattice.T

    atoms_cart = atoms_cart * param.BOHR #unit: Angstrom

    arr = (ctypes.c_char_p * num_atoms)()
    arr[:] = atom_symbols

    M_matrix = np.zeros((num_kpts, nntot, num_bands, num_bands), dtype=np.complex128)
    for k in range(num_kpts):
        for i in range(nntot):
            M_matrix[k,i] = Mmn[k,i].T

    A_matrix = np.zeros((num_kpts,num_wann,num_bands),dtype=np.complex128)
    for k in range(num_kpts):
        A_matrix[k] = Amn[k].T


    #output
    U_matrix = np.ndarray((num_kpts,num_wann,num_wann), dtype = np.complex128)
    U_matrix_opt = np.ndarray((num_kpts,num_wann,num_bands), dtype = np.complex128)
    lwindow = np.ndarray((num_kpts,num_bands), dtype=np.int32)
    wann_centres = np.ndarray((num_wann,3))
    wann_spreads = np.ndarray((num_wann))
    spread = np.ndarray((3))

    lib_wannier90.wannier90_run(ctypes.c_char_p(seedname),mp_grid_dim.ctypes.data_as(ctypes.c_void_p),ctypes.c_int(num_kpts), \
                                real_lattice_T.ctypes.data_as(ctypes.c_void_p),recip_lattice_T.ctypes.data_as(ctypes.c_void_p), \
                                kpt_latt.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(num_bands), ctypes.c_int(num_wann), \
                                ctypes.c_int(nntot),ctypes.c_int(num_atoms), \
                                arr,atoms_cart.ctypes.data_as(ctypes.c_void_p),\
                                ctypes.c_int(gamma_only),\
                                M_matrix.ctypes.data_as(ctypes.c_void_p),A_matrix.ctypes.data_as(ctypes.c_void_p),\
                                eigenvalues.ctypes.data_as(ctypes.c_void_p),\
                                U_matrix.ctypes.data_as(ctypes.c_void_p),U_matrix_opt.ctypes.data_as(ctypes.c_void_p),\
                                lwindow.ctypes.data_as(ctypes.c_void_p),wann_centres.ctypes.data_as(ctypes.c_void_p),\
                                wann_spreads.ctypes.data_as(ctypes.c_void_p),spread.ctypes.data_as(ctypes.c_void_p))
    for k in range(num_kpts):
        U_matrix[k] = U_matrix[k].T

    #print U_matrix

    return U_matrix

class wannier2:

    def __init__(self, the_mf, mp_grid_dim, seedname, bands=None, plot=False, reorder=False, max_memory=None):

        self.mp_grid_dim = np.asarray(mp_grid_dim, dtype=np.int32)
        self.num_kpts = mp_grid_dim[0]*mp_grid_dim[1]*mp_grid_dim[2]

        self.seedname = seedname
        self.plot = plot
        self.reorder = reorder

        self.mf = the_mf
        self.cell = self.mf.cell
        self.real_lattice = self.cell.lattice_vectors()
        self.recip_lattice = self.cell.reciprocal_vectors()
        
        self.max_memory = max_memory
        if self.max_memory is None: self.max_memory = self.cell.max_memory

        self.num_bands_tot = self.cell.nao_nr() 
        self.bands = bands
        if self.bands is None: self.bands = np.arange(self.num_bands_tot)

        self.num_atoms = self.cell.natm
        self.atom_symbols = [x[0] for x in self.cell._atom]
        self.atoms_cart = np.asarray([x[1] for x in self.cell._atom])

        self.kpts = None
        if isinstance(self.mf, pbc.scf.khf.KSCF) : self.kpts = self.mf.kpts
        elif isinstance(self.mf, pbc.scf.hf.SCF) : self.kpts = self.mf.kpt
        self.kpts = self.kpts.reshape((-1,3))
        self.kpt_latt = self.cell.get_scaled_kpts(self.kpts)

        assert(len(self.kpt_latt) == self.num_kpts)

        self.gamma_only = False
        if gamma_point(self.kpts) : self.gamma_only = True

        eig = self.mf.mo_energy
        eig *= param.HARTREE2EV #in eV
        self.eig = eig.reshape((-1,self.num_bands_tot))

        self.nnlist,self.nncell,self.nntot,self.num_bands,self.num_wann,\
        self.proj_site,self.proj_l,self.proj_m,self.proj_radial,\
        self.proj_z,self.proj_x,self.proj_zona,self.exclude_bands,\
        self.proj_s,self.proj_s_qaxis = self.wannier90_setup()

        #print self.proj_site
        #print self.proj_l
        #print self.proj_m
        #print self.proj_radial

    def kernel(self):

        nnkpts = get_nnkpts(self.nntot, self.nnlist, self.nncell)
        self.bpts = get_bpts(self, nnkpts, self.kpts)

        if(self.plot):
            write_bloch_u_r(self, self.bands, self.num_bands_tot, self.kpts)

        t0 = (time.clock(),time.time())
        self.Mmn = comput_Mmn(self, self.bands, self.num_bands, self.bpts, self.kpts)
        #write_Mmn(self, self.seedname+'.mmn', self.num_kpts, self.num_bands, self.Mmn, nnkpts)
        t1 = tools.timer("Mmn",t0)

        self.lmr = np.column_stack((self.proj_l,self.proj_m,self.proj_radial))

        t0 = (time.clock(),time.time()) 
        self.Amn = comput_Amn(self, self.proj_site, self.lmr, self.proj_z, self.proj_x, self.proj_zona, self.bands, self.num_bands, self.kpts)
        t1 = tools.timer("Amn",t0)

        u_mat = self.wannier90_run()
        self.u_mat = u_mat
        if (self.reorder):
            seq = self.assign_wf2atm()
            self.u_mat = u_mat[:,:,seq]

        #calc_iso_cutoff(self, self.u_mat, self.bands, self.num_bands, self.kpts)

    wannier90_setup = wannier90_setup
    wannier90_run = wannier90_run
    assign_wf2atm = assign_wf2atm

class wannier:

    def __init__(self, the_mf, nnkpts, seedname,bands, has_proj_data=False, kpts = np.zeros((1,3))):

        self.mf = the_mf
        self.cell = self.mf.cell
        self.nnkpts = nnkpts
        self.bands = bands
        self.nband = len(self.bands)
        self.kpts = kpts
        self.file_mmn = seedname+'.mmn'
        self.file_amn = seedname+'.amn'
        self.file_ene = seedname+'.eig'

        self.bpts = self.get_bpts()
        self.Mmn = self.comput_Mmn()
        self.write_Mmn(self.file_mmn, len(self.kpts))
        self.write_bloch_u_r()
        self.write_eig(self.file_ene)

        self.Rc = None 
        self.lmr = None
        self.zaxis = None
        self.xaxis = None
        self.zona = None
        self.Amn = None
        if(has_proj_data):
            self.file_proj_data = seedname+'.proj'
            self.Rc,self.lmr,self.zaxis,self.xaxis,self.zona = read_proj_data(self.file_proj_data)
            self.Amn = self.comput_Amn()
            self.write_Amn(self.file_amn, len(self.kpts))

    get_bpts = get_bpts
    comput_Mmn = comput_Mmn
    write_Mmn = write_Mmn
    comput_Amn = comput_Amn
    write_Amn = write_Amn
    write_bloch_u_r = write_bloch_u_r
    write_eig = write_eig
