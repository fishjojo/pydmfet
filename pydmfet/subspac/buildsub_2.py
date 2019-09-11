import math
import numpy as np
from .buildsub import fix_virt
from pydmfet import tools

def construct_subspace2(mo_coeff, mo_occ, mol_frag, ints, impurityOrbs, \
                        dim_imp=None, dim_bath=None, threshold=1e-13, occ_tol=1e-8):

    #loc2sub sequence: occ_imp, occ_bath, frac, virt_imp

    occ = mo_occ.copy()
    for i in range(occ.size):
        if(occ[i] > 2-occ_tol):
            occ[i] = 2.0
        else:
            occ[i] = 0.0

    mocc = mo_coeff[:,occ>0]
    rdm1_loc = np.dot(mocc * occ[occ>0], mocc.T.conj())

    numImpOrbs = np.sum(impurityOrbs)
    numTotalOrbs = len(impurityOrbs)
    isImp = np.outer(impurityOrbs,impurityOrbs) == 1
    imp1RDM = np.reshape(rdm1_loc[isImp], (numImpOrbs, numImpOrbs))

    eigenvals_imp, eigenvecs_imp = fix_virt(ints, mol_frag, imp1RDM, numImpOrbs,numTotalOrbs, threshold)

    tmp = []
    for i in range(numImpOrbs):
        if(eigenvals_imp[i] < threshold):
            eigenvals_imp[i] = 0.0
        elif(eigenvals_imp[i] > 2.0 - threshold):
            eigenvals_imp[i] = 2.0
            tmp.append(i)

    if(len(tmp) == 0):
        tmp.append(numImpOrbs-1)

    last_imp_orb = -1
    for i in range(tmp[0],-1,-1):
        if(eigenvals_imp[i] > 1.999):
            last_imp_orb = i
            break

    if(last_imp_orb == -1):
        tokeep_imp = np.sum( -np.maximum( -eigenvals_imp, eigenvals_imp - 2.0 ) > threshold )
    else:
        tokeep_imp = last_imp_orb + 1

    tools.VecPrint(eigenvals_imp, "occ_imp") 
    #print "occ_imp"
    #print eigenvals_imp

    isocc = np.zeros(occ.size,dtype=int)
    isocc[occ>0] = 1
    nocc = np.sum(isocc)
    embeddingOrbs = 1 - impurityOrbs
    nbath = np.sum(embeddingOrbs)
    isbath = np.outer(embeddingOrbs,isocc) == 1
    mocc = np.reshape(mo_coeff[isbath],(nbath,nocc))
    s = 2.0*np.dot(mocc.T, mocc)
    eigenvals_bath, eigenvecs_bath = np.linalg.eigh(s)

    eigenvecs_bath = math.sqrt(2.0)*np.dot(mocc,eigenvecs_bath)
    for i in range(nocc):
        #eigenvecs_bath[:,i] = eigenvecs_bath[:,i]/math.sqrt(eigenvals_bath[i])
        fac = np.dot(eigenvecs_bath[:,i],eigenvecs_bath[:,i])
        eigenvecs_bath[:,i] = eigenvecs_bath[:,i]/math.sqrt(fac)


    idx = np.maximum(-eigenvals_bath[:], eigenvals_bath[:] - 2.0 ).argsort()
    eigenvals_bath = eigenvals_bath[idx]
    eigenvecs_bath[:,:] = eigenvecs_bath[:,idx]

    tools.VecPrint(eigenvals_bath, "occ_bath")
    #print "occ_bath"
    #print eigenvals_bath

    tokeep_bath = tokeep_imp
    if(dim_bath is not None):
        tokeep_bath = min(dim_bath, nocc)
        tokeep_imp = min(dim_bath, numImpOrbs)
    if(dim_imp is not None):
        tokeep_imp = min(dim_imp,numImpOrbs)

    if(tokeep_imp < numImpOrbs):
        frozenEigVals_imp = -eigenvals_imp[tokeep_imp:]
        frozenEigVecs_imp = eigenvecs_imp[:,tokeep_imp:]
        idx = frozenEigVals_imp.argsort()
        eigenvecs_imp[:,tokeep_imp:] = frozenEigVecs_imp[:,idx]
        frozenEigVals_imp = -frozenEigVals_imp[idx]
        eigenvals_imp[tokeep_imp:] = frozenEigVals_imp

    frozenEigVals_bath = -eigenvals_bath[tokeep_bath:]
    frozenEigVecs_bath = eigenvecs_bath[:,tokeep_bath:]
    idx = frozenEigVals_bath.argsort()
    eigenvecs_bath[:,tokeep_bath:] = frozenEigVecs_bath[:,idx]
    frozenEigVals_bath = -frozenEigVals_bath[idx]
    eigenvals_bath[tokeep_bath:] = frozenEigVals_bath


    loc2sub = eigenvecs_bath
    for counter in range(numImpOrbs):
        loc2sub = np.insert(loc2sub, counter, 0.0, axis=1) #Stack columns with zeros in the beginning
    counter = 0
    for counter2 in range(numTotalOrbs):
        if ( impurityOrbs[counter2] ):
            loc2sub = np.insert(loc2sub, counter2, 0.0, axis=0) #Stack rows with zeros on locations of the impurity orbitals
            counter += 1
    assert( counter == numImpOrbs )

    for icol in range(numImpOrbs):
        counter = 0
        for irow in range(numTotalOrbs):
            if( impurityOrbs[irow]):
                loc2sub[irow][icol] = eigenvecs_imp[counter][icol]
                counter+=1
        assert( counter == numImpOrbs )


    isfrac = np.logical_and(mo_occ>occ_tol, mo_occ<2.0-occ_tol)
    nfrac = np.sum(isfrac)
    mo_frac = mo_coeff[:,isfrac].copy()
    occ_frac = mo_occ[isfrac].copy()


    Occupations = np.hstack((eigenvals_imp, eigenvals_bath))

    _loc2sub = np.ndarray((numTotalOrbs,tokeep_imp+nocc), dtype=np.double)
    _Occupations = np.ndarray((tokeep_imp+nocc), dtype=np.double)

    _loc2sub[:,:tokeep_imp] = loc2sub[:,:tokeep_imp]
    _loc2sub[:,tokeep_imp:tokeep_imp+nocc] = loc2sub[:,numImpOrbs:numImpOrbs+nocc]
    _Occupations[:tokeep_imp] = Occupations[:tokeep_imp]
    _Occupations[tokeep_imp:tokeep_imp+nocc] = Occupations[numImpOrbs:numImpOrbs+nocc]

    icount = 0
    for i in range(nfrac):
        tmp = np.dot(_loc2sub.T, mo_frac[:,i])
        tmp1 = np.dot(_loc2sub, np.diag(tmp))
        for j in range(_loc2sub.shape[-1]):
            mo_frac[:,i] = mo_frac[:,i] - tmp1[:,j]
        norm = np.linalg.norm(mo_frac[:,i])
        if(norm>1e-8):
            mo_frac[:,i] = mo_frac[:,i]/norm
            _loc2sub = np.insert(_loc2sub,tokeep_imp+tokeep_bath, mo_frac[:,i], axis=1)
            _Occupations = np.insert(_Occupations,tokeep_imp+tokeep_bath, occ_frac[i])
            eigenvals_bath = np.append(eigenvals_bath,[occ_frac[i]])
            icount += 1

    nfrac = icount

    occ_virt = eigenvals_imp[tokeep_imp:numImpOrbs]
    mo_virt = loc2sub[:,tokeep_imp:numImpOrbs]
    nvirt = numImpOrbs - tokeep_imp
    icount = 0
    for i in range(nvirt):
        tmp = np.dot(_loc2sub.T,mo_virt[:,i])
        tmp1 = np.dot(_loc2sub, np.diag(tmp))
        for j in range(_loc2sub.shape[-1]):
            mo_virt[:,i] = mo_virt[:,i] - tmp1[:,j]
        norm = np.linalg.norm(mo_virt[:,i])
        if(norm>1e-8):
            mo_virt[:,i] = mo_virt[:,i]/norm
            _loc2sub = np.insert(_loc2sub,tokeep_imp+tokeep_bath+nfrac, mo_virt[:,i], axis=1)
            _Occupations = np.insert(_Occupations,tokeep_imp+tokeep_bath+nfrac, occ_virt[i])
            icount += 1

    nvirt = icount
        
    _Occupations[:tokeep_imp+tokeep_bath+nfrac] = 0.0

    dim = tokeep_imp+nocc+nfrac+nvirt
    print ("check orthogonality:")
    print ('norm: ', np.linalg.norm(np.dot(_loc2sub.T, _loc2sub)-np.eye(dim)))
    print ('max: ', np.amax(np.absolute(np.dot(_loc2sub.T, _loc2sub)-np.eye(dim))))
    #assert(np.amax(np.absolute(np.dot(_loc2sub.T, _loc2sub)-np.eye(dim)))<1e-9)


    return (tokeep_imp, tokeep_bath+nfrac, _Occupations, _loc2sub, eigenvals_imp, eigenvals_bath)
 


