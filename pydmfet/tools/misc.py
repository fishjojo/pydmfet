from pyscf import lo
import numpy as np
import copy

#for dfet
def add_ghost(atoms, ghost):

    atom_new = ''
    if isinstance(atoms, (str, unicode)):
        j = 0
        for dat in atoms.split('\n'):
            dat = dat.strip()
            if dat and dat[0] != '#':
                if(ghost[j] == 1):
                    dat =  'ghost'+dat
                atom_new = atom_new + dat +'\n'
            j+=1
    else:
	atom_new = copy.copy(atoms)
	natm = len(atom_new)
	for i in range(natm):
	    if(ghost[i] == 1):
		tmp = list(atom_new[i])
	        tmp[0] = 'ghost'+tmp[0]
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
		    dat =  'ghost'+dat
            elif(j == bound_atom_index):
		dat = replace_symbol + ' ' + dat[2:]
	    elif(j > bound_atom_index):
		if(is_frag == True):
                    dat =  'ghost'+dat
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

    print max_list
    max_list*=-1.0
    ind = max_list.argsort()
    print ind
    mo[:,:nocc] = mo[:,ind]

    for i in range(norb):
	P_frag += np.dot(mo[:,i:i+1], mo[:,i:i+1].T) *2.0
    for i in range(norb, nocc):
	P_env += np.dot(mo[:,i:i+1], mo[:,i:i+1].T) *2.0



    return (P_frag, P_env)
