from pyscf import lo
import numpy as np
import copy
from pyscf.tools import molden

#generate molden file
def mo_molden(mol,mo_coeff,filename):
    with open( filename, 'w' ) as thefile:
            molden.header( mol, thefile )
            molden.orbital_coeff( mol, thefile, mo_coeff )


#for dfet
def add_ghost(atoms, ghost):

    atom_new = ''
    if isinstance(atoms, (bytes, str)):
        j = 0
        for dat in atoms.split('\n'):
            dat = dat.strip()
            if dat and dat[0] != '#':
                if(ghost[j] == 1):
                    dat =  'GHOST-'+dat
                atom_new = atom_new + dat +'\n'
            j+=1
    else:
        atom_new = copy.copy(atoms)
        natm = len(atom_new)
        for i in range(natm):
            if(ghost[i] == 1):
                tmp = list(atom_new[i])
                tmp[0] = 'GHOST-'+tmp[0]
                atom_new[i] = tuple(tmp)
                
    return atom_new

#for oniom
def frag_atom(atom, frag):

    atom_new = ''
    j = 0
    for dat in atom.split('\n'):
        dat = dat.strip()
        if dat and dat[0] != '#':
            if(frag[j] == 1):
                atom_new = atom_new + dat +'\n'
        j+=1

    return atom_new

#for dfet cut one bond
def add_ghost_part_atom(atom, bound_atom_index, replace_symbol, is_frag=True):

    atom_new = ''
    j = 1
    for dat in atom.split('\n'):
        dat = dat.strip()
        if dat and dat[0] != '#':
            if(j < bound_atom_index):
                if(is_frag == False):
                    dat =  'GHOST-'+dat
            elif(j == bound_atom_index):
                dat = replace_symbol + ' ' + dat[2:]
            elif(j > bound_atom_index):
                if(is_frag == True):
                    dat =  'GHOST-'+dat
            atom_new = atom_new + dat +'\n'
        j+=1

    return atom_new

#for oniom cut one bond
def frag_atom_part_atom(atom, bound_atom_index, replace_symbol):

    atom_new = ''
    j = 1
    for dat in atom.split('\n'):
        dat = dat.strip()
        if dat and dat[0] != '#':
            if(j < bound_atom_index):
                atom_new = atom_new + dat +'\n'
            elif(j == bound_atom_index):
                dat = replace_symbol + ' ' + dat[2:]
                atom_new = atom_new + dat
        j+=1

    return atom_new

def frag_atom_part_atom_hmod(atom, bound_atom_index, replace_symbol, bond_length):

    atom_new = ''
    j = 1
    for dat in atom.split('\n'):
        dat = dat.strip()
        if dat and dat[0] != '#':
            if(j < bound_atom_index[1]):
                atom_new = atom_new + dat +'\n'
                if(j==bound_atom_index[0]):
                    xyz1 = np.fromstring(dat[2:], dtype=float, sep=' ')
            elif(j == bound_atom_index[1]):
                xyz2 = np.fromstring(dat[2:], dtype=float, sep=' ')
                vec = xyz2 - xyz1
                coord_h = xyz1 + bond_length/np.linalg.norm(vec) * vec
                dat = replace_symbol + ' ' + str(coord_h[0]) + ' ' + str(coord_h[1]) + ' ' + str(coord_h[2])
                #dat = replace_symbol + ' ' + dat[2:]
                atom_new = atom_new + dat
        j+=1

    return atom_new


def frag_atom_part_atom_hmod_mult(atom, atom1, atom2, bound_atom_index, replace_symbol, bond_length):

    nbound = atom1.size
    xyz1 = []
    atom_new = ''
    j = 1
    for dat in atom.split('\n'):
        dat = dat.strip()
        if dat and dat[0] != '#':
            if(j <= bound_atom_index[0]):
                atom_new = atom_new + dat +'\n'
                for k in range(nbound):
                    if(j==atom1[k]):
                        xyz1.append(np.fromstring(dat[2:], dtype=float, sep=' '))
            elif(j <= bound_atom_index[1]):
                for k in range(nbound):
                    if(j==atom2[k]):
                        xyz2 = np.fromstring(dat[2:], dtype=float, sep=' ')
                        vec = xyz2 - xyz1[k]
                        coord_h = xyz1[k] + bond_length/np.linalg.norm(vec) * vec
                        dat = replace_symbol + ' ' + str(coord_h[0]) + ' ' + str(coord_h[1]) + ' ' + str(coord_h[2])
                        atom_new = atom_new + dat +'\n'
        j+=1

    return atom_new




def localize_dens(mf,nfrag_atm, norb, method = 'pipek'):

    mol = mf.mol
    nocc = mol.nelectron // 2
    if(method == 'pipek'):
        mo = lo.pipek.PM(mol).kernel(mf.mo_coeff[:,:nocc], verbose=4)
    elif(method == 'boys'):
        mo = lo.boys.Boys(mol).kernel(mf.mo_coeff[:,:nocc], verbose=4)

    nbas = mol.nao_nr()
    P_frag = np.zeros((nbas,nbas))
    P_env = np.zeros((nbas,nbas))

    pop_list = []
    for i in range(nocc):
        pop = lo.pipek.atomic_pops(mol, mo[:,i:i+1], method='mulliken')
        pop_list.append(pop)
        '''
        if_frag = False
        for j in range(nfrag_atm):
            if(pop[j,0,0] > pop_min):
                if_frag = True
                print "add mo ",i+1, " to frag"
                P_frag += np.dot(mo[:,i:i+1], mo[:,i:i+1].T) *2.0
                break

        if(if_frag == False):
            P_env += np.dot(mo[:,i:i+1], mo[:,i:i+1].T) *2.0
        '''

    max_list = np.zeros((nocc))
    for i in range(nocc):
        max_list[i] = np.amax(pop_list[i][:nfrag_atm,0,0])

    print (max_list)
    max_list*=-1.0
    ind = max_list.argsort()
    print (ind)
    mo[:,:nocc] = mo[:,ind]

    for i in range(norb):
        P_frag += np.dot(mo[:,i:i+1], mo[:,i:i+1].T) *2.0
    for i in range(norb, nocc):
        P_env += np.dot(mo[:,i:i+1], mo[:,i:i+1].T) *2.0



    return (P_frag, P_env)




def print_eomcc_t1(t1, thresh = 0.1):

    #nocc = t1.shape[0]
    #nvirt = t1.shape[1]

    ind = np.unravel_index(np.argsort(-np.absolute(t1),axis=None) ,t1.shape)
    ind2 = np.absolute(t1[ind]) > thresh


    if(sum(ind2) > 0):
        print (ind[0][ind2]+1)
        print ('| | |')
        print (ind[1][ind2])
        print (t1[ind[0][ind2],ind[1][ind2]])


