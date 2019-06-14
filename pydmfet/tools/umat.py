import numpy as np
from pyscf import dft

def read_vemb(filename, nx, ny, nz):

    vemb = np.zeros((nx*ny*nz))

    file = open(filename, 'r')
    data = file.readlines()
    for i in range(nx*ny*nz):
        vemb[i] = float(data[i].split()[0])

    print ('vemb')
    print (vemb)
    return vemb

def get_uniform_grid(ax,ay,az,nx,ny,nz):

    x = np.linspace(0, ax ,nx)
    y = np.linspace(0, ay ,ny)
    z = np.linspace(0, az ,nz)

    ngrid = nx*ny*nz
    coord = np.empty((ngrid,3))

    i = 0
    for iz in range(nz):
        zz = z[iz]
        for iy in range(ny):
            yy = y[iy]
            for ix in range(nx):
                xx = x[ix]
                coord[i,0] = xx
                coord[i,1] = yy
                coord[i,2] = zz
                i += 1
    print ('coord')
    print (coord)
    return coord


def calc_umat(mol, vemb, coord, w):

    nbas = mol.nao_nr()
    ngrid = len(vemb)

    ao_r = dft.numint.eval_ao(mol, coord)
    v_ao_r = np.empty((ngrid,nbas))
    for igrid in range(ngrid):
        v_ao_r[igrid,:] = vemb[igrid] * ao_r[igrid,:]

    umat = w * np.dot(v_ao_r.T, ao_r)

    return umat


def read_umat_from_uniform_vemb(mol,ax,ay,az,nx,ny,nz,filename, unit='a'):

    b2a = 0.52917721067
    if(unit == 'a'):
        ax /= b2a
        ay /= b2a
        az /= b2a

    volume = ax*ay*az
    w = volume/(nx*ny*nz)
    print ('volume = ', volume)
    print ('weight = ', w)

    vemb = read_vemb(filename, nx, ny, nz)
    coord = get_uniform_grid(ax,ay,az,nx,ny,nz)
    umat = calc_umat(mol, vemb, coord, w)

    return umat

