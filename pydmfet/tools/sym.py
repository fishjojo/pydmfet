import numpy as np

def h_chain_sym_tab(nat):

    '''
    symmetry table for embedding potential:
    H chain cut in the center
    minimal basis

    deprecated! use the following function instead
    '''

    sym_tab = np.zeros([nat,nat],dtype=int)
    for i in range(nat//2):
        sym_tab[i,i] = i
        sym_tab[nat-i-1,nat-i-1] = i
    val = nat//2
    for i in range(nat//2-1):
        for j in range(i+1, nat//2):
            sym_tab[i,j] = val
            sym_tab[nat-j-1,nat-i-1] = val
            val += 1

    for i in range(nat//2):
        for j in range(nat//2,nat):
            if sym_tab[i,j] == 0:
                sym_tab[i,j] = val
                sym_tab[nat-j-1,nat-i-1] = val
                val += 1

    sym_tab = np.tril(sym_tab.T,-1) + sym_tab

    return sym_tab

def h_lattice_sym_tab(atm_ind):

    '''
    symmetry table for embedding potential:
    H lattice cut in the center
    minimal basis
    '''

    nrow = atm_ind.shape[0]
    ncol = atm_ind.shape[1]
    nat = nrow * ncol

    sym_tab = np.zeros([nat,nat],dtype=int)
    sym_tab[:,:] = -1

    val = 0
    #diagonal elements
    for i in range(nrow//2+nrow%2):
        for j in range(ncol//2+ncol%2):
            ind = atm_ind[i,j]
            sym_tab[ind,ind] = val
            ind = atm_ind[i,ncol-j-1]
            sym_tab[ind,ind] = val
            ind = atm_ind[nrow-i-1,j]
            sym_tab[ind,ind] = val
            ind = atm_ind[nrow-i-1,ncol-j-1]
            sym_tab[ind,ind] = val

            val += 1

    #off-diagonal elements
    for i in range(nrow):
        for j in range(ncol):
            for k in range(nrow):
                for l in range(ncol):
                    if i==k and j==l: 
                        continue

                    inc = False

                    ind_1 = atm_ind[i,j]
                    ind_2 = atm_ind[k,l]
                    if sym_tab[ind_1,ind_2] == -1:
                        sym_tab[ind_1,ind_2] = val
                        sym_tab[ind_2,ind_1] = val
                        inc = True

                    ind_1 = atm_ind[i,ncol-j-1]
                    ind_2 = atm_ind[k,ncol-l-1]
                    if sym_tab[ind_1,ind_2] == -1:
                        if not inc: raise RuntimeError("bug")
                        sym_tab[ind_1,ind_2] = val
                        sym_tab[ind_2,ind_1] = val

                    ind_1 = atm_ind[nrow-i-1,j]
                    ind_2 = atm_ind[nrow-k-1,l]
                    if sym_tab[ind_1,ind_2] == -1:
                        if not inc: raise RuntimeError("bug")
                        sym_tab[ind_1,ind_2] = val
                        sym_tab[ind_2,ind_1] = val

                    ind_1 = atm_ind[nrow-i-1,ncol-j-1]
                    ind_2 = atm_ind[nrow-k-1,ncol-l-1]
                    if sym_tab[ind_1,ind_2] == -1:
                        if not inc: raise RuntimeError("bug")
                        sym_tab[ind_1,ind_2] = val
                        sym_tab[ind_2,ind_1] = val

                    if inc: val += 1

    return sym_tab
