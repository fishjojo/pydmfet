import math
import numpy as np
from pydmfet import qcwrap,tools,libgen


def subocc_to_dens_part(P_ref, occ_imp, occ_bath, dim_imp, dim_bath):

    dim = dim_imp + dim_bath

    P_imp = np.zeros([dim,dim], dtype=float)
    P_bath = np.zeros([dim,dim], dtype=float)


    for i in range(dim):
        if(i < dim_imp):
            P_imp[i][i] = occ_imp[i]
        else:
            index = i-dim_imp
            if(index >= dim_imp):
                break
            if(occ_imp[index] > 0.8):
                P_imp[i][i] = 2.0 - occ_imp[index]
                P_imp[i][index] = P_ref[i][index] #can't determine sign #math.sqrt(2.0*occ_imp[index] - occ_imp[index]**2)
                P_imp[index][i] = P_imp[i][index]
            else:
                P_imp[index][index] = 0.0
    '''
    for i in range(dim):
        if(i < dim_bath):
            index = i+dim_imp
            if(occ_imp[i] <= 0.8 or occ_bath[i] > 1.9):
                P_bath[index][index] = occ_bath[i]
                P_bath[i][index] = P_ref[i][index]  #math.sqrt(2.0*occ_bath[i]-occ_bath[i]**2)
                P_bath[index][i] = P_bath[i][index]
                P_bath[i][i] = 2.0 - occ_bath[i]
    '''
    return (P_imp, P_bath)


def mulliken_partition_loc(frag_orbs,dm_loc):

    dim = dm_loc.shape[0]

    P_frag = np.zeros([dim,dim],dtype=float)
    P_env = np.zeros([dim,dim],dtype=float)

    for i in range(dim):
        for j in range(dim):
            if(frag_orbs[i] == 1 and frag_orbs[j] == 1): 
                P_frag[i,j] = dm_loc[i,j]
            elif(frag_orbs[i] == 0 and frag_orbs[j] == 0):
                P_env[i,j] = dm_loc[i,j]
            else:
                P_frag[i,j] = 0.5*dm_loc[i,j]
                P_env[i,j]  = 0.5*dm_loc[i,j]

    print ("Ne_frag_loc = ", np.sum(np.diag(P_frag)))
    print ("Ne_env_loc = ", np.sum(np.diag(P_env)))

    return (P_frag,P_env)

def loc_fullP_to_fragP(frag_orbs,mo_coeff,NOcc,NOrb):

    weight_frag = []
    weight_env = []
    for i in range(NOcc):
        sum_frag = 0.0
        sum_env = 0.0
        for j in range(NOrb):
            if(frag_orbs[j] == 1):
                sum_frag += mo_coeff[j,i] * mo_coeff[j,i]
            else:
                sum_env += mo_coeff[j,i] * mo_coeff[j,i]
        weight_frag.append(sum_frag)
        weight_env.append(sum_env)

    print (2.0*np.sum(weight_frag))
    print (2.0*np.sum(weight_env))

    P_frag = np.zeros([NOrb,NOrb],dtype=float)
    P_env = np.zeros([NOrb,NOrb],dtype=float)

    index = 0
    dim = mo_coeff.shape[0]
    mo_frag = np.zeros((dim,NOcc))
    mo_occ = np.zeros((NOcc))
    for i in range(NOcc):
        P_tmp = 2.0*np.outer(mo_coeff[:,i], mo_coeff[:,i])
        print (weight_frag[i], weight_env[i])
        if(weight_frag[i] >= weight_env[i]):
            P_frag = P_frag + P_tmp

            mo_frag[:,index] = mo_coeff[:,i]
            mo_occ[index] = 2.0
            index +=1
        else:
            P_env = P_env + P_tmp

    print ("Ne_frag_loc = ", np.sum(np.diag(P_frag)))
    print ("Ne_env_loc = ", np.sum(np.diag(P_env)))

    return (P_frag, P_env, mo_frag, mo_occ)


def fullP_to_fragP(obj, subTEI, Nelec,P_ref, dim, dim_imp, mf_method):

    loc2sub = obj.loc2sub
    core1PDM_loc = obj.core1PDM_loc

    fock_sub = obj.ints.fock_sub( loc2sub, dim, core1PDM_loc)

    energy, OneDM, mo_coeff = qcwrap.pyscf_rhf.scf( fock_sub, subTEI, dim, Nelec, P_ref, mf_method)

    P_imp = np.zeros([dim, dim],dtype = float)
    P_bath = np.zeros([dim, dim],dtype = float)

    NOcc = Nelec//2
    for i in range(NOcc):
        isimp = classify_orb(mo_coeff[:,i],dim_imp)
        P_tmp = 2.0*np.outer(mo_coeff[:,i], mo_coeff[:,i])
        if isimp :
            P_imp = P_imp + P_tmp
        else :
            P_bath = P_bath + P_tmp

    
    print ("Ne_imp = ", np.sum(np.diag(P_imp)))
    print ("Ne_bath = ",  np.sum(np.diag(P_bath)))
    print ("|P_imp + P_bath - P_ref| = ", np.linalg.norm(P_imp+P_bath-P_ref))

    return (P_imp,P_bath)

def classify_orb(orb,dim_imp):

    sum_imp = 0.0
    for i in range(dim_imp):
        sum_imp += orb[i]*orb[i]

    print (sum_imp)

    sum_bath = 1.0-sum_imp

    if(sum_imp > sum_bath):
        return True
    else:
        return False


def build_Pimp(occ,loc2sub,tokeep_imp,dim):

    occ_loc = np.zeros([dim],dtype=float)
    occ_loc[:tokeep_imp] = occ[:tokeep_imp]

    Pimp_loc = np.dot( np.dot( loc2sub, np.diag( occ_loc ) ), loc2sub.T )

    return Pimp_loc

def build_core(occ,loc2sub,idx_imp):

        core_cutoff = 0.01

        occ_frag = np.zeros( len(occ) ,dtype = float)

        NOrb_imp = 0
        for cnt in range(len(occ)):
            if ( occ[ cnt ] < core_cutoff ):
                occ[ cnt ] = 0.0
            elif ( occ[ cnt ] > 2.0 - core_cutoff ):
                occ[ cnt ] = 2.0
                if(cnt < idx_imp):
                    NOrb_imp += 1
                    occ_frag[cnt] = 2.0
            else:
                print ("environment orbital occupation = ", occ[ cnt ])
                raise Exception("subspace construction failed!")

        Nelec_core = int(round(np.sum( occ )))
        core1PDM_loc = np.dot( np.dot( loc2sub, np.diag( occ ) ), loc2sub.T )
        frag_core1PDM_loc = np.dot( np.dot( loc2sub, np.diag( occ_frag ) ), loc2sub.T )

        return (core1PDM_loc, Nelec_core, NOrb_imp, frag_core1PDM_loc)


def construct_subspace(ints,mol_frag,mol_env,OneDM, impurityOrbs, threshold=1e-13, dim_bath = None, dim_imp = None):
    '''
    Subspace construction
    OneDM is in local orbital representation
    '''

    numImpOrbs   = np.sum( impurityOrbs )
    numTotalOrbs = len( impurityOrbs )

    impOrbs = impurityOrbs.copy()
    impOrbs = np.matrix(impOrbs)
    if (impOrbs.shape[0] > 1):
        impOrbs = impOrbs.T
    isImp = np.dot( impOrbs.T , impOrbs ) == 1
    imp1RDM = np.reshape( OneDM[ isImp ], ( numImpOrbs , numImpOrbs ) )

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
        if(eigenvals_imp[i] > 1.99):
            last_imp_orb = i
            break

    if(last_imp_orb == -1):
        tokeep_imp = np.sum( -np.maximum( -eigenvals_imp, eigenvals_imp - 2.0 ) > threshold )
    else:
        tokeep_imp = last_imp_orb + 1

    print ("occ_imp")
    print (eigenvals_imp)
    #print eigenvecs_imp


###############################################

    embeddingOrbs = 1 - impurityOrbs
    embeddingOrbs = np.matrix( embeddingOrbs )
    if (embeddingOrbs.shape[0] > 1):
        embeddingOrbs = embeddingOrbs.T # Now certainly row-like matrix (shape = 1 x len(vector))
    isEmbedding = np.dot( embeddingOrbs.T , embeddingOrbs ) == 1
    numEmbedOrbs = np.sum( embeddingOrbs )
    embedding1RDM = np.reshape( OneDM[ isEmbedding ], ( numEmbedOrbs , numEmbedOrbs ) )

    eigenvals_bath, eigenvecs_bath = fix_virt(ints, mol_frag, embedding1RDM, numEmbedOrbs,numTotalOrbs, threshold)

    for i in range(numEmbedOrbs):
        if(eigenvals_bath[i] < threshold):
            eigenvals_bath[i] = 0.0
        elif(eigenvals_bath[i] > 2.0-threshold):
            eigenvals_bath[i] = 2.0

    #if (tokeep_bath > tokeep_imp):
    #    print "Throwing out ", tokeep_bath - tokeep_imp, "bath orbitals"
    #    tokeep_bath = tokeep_imp

    tokeep_bath = tokeep_imp #keep all bath orbitals
    if(dim_bath is not None):
        tokeep_bath = min(dim_bath, numTotalOrbs - numImpOrbs)
        tokeep_imp = min(dim_bath,numImpOrbs)

    if(dim_imp is not None):
        tokeep_imp = min(dim_imp,numImpOrbs)

    print ("occ_bath")
    print (eigenvals_bath)
    #print eigenvecs_bath

    #tokeep_imp = numImpOrbs  #keep all imp orbitals in the active space
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

    #print eigenvals_bath
    #print eigenvecs_bath

    loc2sub = eigenvecs_bath
    for counter in range(0, numImpOrbs):
        loc2sub = np.insert(loc2sub, counter, 0.0, axis=1) #Stack columns with zeros in the beginning
    counter = 0
    for counter2 in range(0, numTotalOrbs):
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


    Occupations = np.hstack(( eigenvals_imp, eigenvals_bath ))
    #print "Occupations"
    #print Occupations

    Occupations[:tokeep_imp] = 0.0
    Occupations[numImpOrbs:numImpOrbs+tokeep_bath] = 0.0
    #print Occupations

    # Orthonormality is guaranteed due to (1) stacking with zeros and (2) orthonormality eigenvecs for symmetric matrix
    assert( np.linalg.norm( np.dot(loc2sub.T, loc2sub) - np.identity(numTotalOrbs) ) < 1e-12 )

    _loc2sub = loc2sub.copy()
    _Occupations = Occupations.copy()
    
    if(tokeep_imp < numImpOrbs):
        _loc2sub[:,tokeep_imp:tokeep_imp+tokeep_bath] = loc2sub[:,numImpOrbs:numImpOrbs+tokeep_bath]
        _loc2sub[:,tokeep_imp+tokeep_bath:tokeep_bath+numImpOrbs] = loc2sub[:,tokeep_imp:numImpOrbs]

        _Occupations[tokeep_imp:tokeep_imp+tokeep_bath] = Occupations[numImpOrbs:numImpOrbs+tokeep_bath]
        _Occupations[tokeep_imp+tokeep_bath:tokeep_bath+numImpOrbs] = Occupations[tokeep_imp:numImpOrbs]
    

    return ( tokeep_imp, tokeep_bath, _Occupations, _loc2sub, eigenvals_imp, eigenvals_bath)



def fix_virt(ints, mol, imp1RDM, numImpOrbs,numTotalOrbs, thresh):

    eigenvals_imp, eigenvecs_imp = np.linalg.eigh( imp1RDM )
    idx = np.argmax(abs(eigenvecs_imp), axis=0)
    eigenvecs_imp[:,eigenvecs_imp[idx,np.arange(numImpOrbs)]<0] *= -1

    nvirt = 0
    for i in range(numImpOrbs):
        if(abs(eigenvals_imp[i]) < thresh or eigenvals_imp[i]<0.0):
            nvirt += 1
    nocc = numImpOrbs - nvirt
    eigenvals_imp[:nocc] = eigenvals_imp[nvirt:].copy()
    eigenvals_imp[nocc:] = 0.0

    tmp = np.pad(eigenvecs_imp[:,:nvirt], ((0,numTotalOrbs-numImpOrbs),(0,0)), 'constant', constant_values=0 )
    subKin = libgen.frag_kin_sub( ints, tmp, nvirt )
    subVnuc1 = libgen.frag_vnuc_sub(mol, ints, tmp, nvirt)
    mo_energy, mo_coeff = np.linalg.eigh(subKin+subVnuc1)
    #print mo_energy

    virt_imp = np.dot(tmp[:numImpOrbs,:],mo_coeff)
    eigenvecs_imp = np.concatenate((eigenvecs_imp[:,nvirt:], virt_imp), axis=1)

    idx = np.maximum( -eigenvals_imp[:nocc], eigenvals_imp[:nocc] - 2.0 ).argsort()
    eigenvals_imp[:nocc] = eigenvals_imp[idx]
    eigenvecs_imp[:,:nocc] = eigenvecs_imp[:,idx]

    idx = np.argmax(abs(eigenvecs_imp), axis=0)
    eigenvecs_imp[:,eigenvecs_imp[idx,np.arange(numImpOrbs)]<0] *= -1

    one = np.dot(eigenvecs_imp.T,eigenvecs_imp)
    assert(np.linalg.norm(one-np.eye(numImpOrbs)) <1e-9)
################################################################
    '''
    deg_info = []
    i = 0
    while i< nocc-1:
        n_deg = 0
        for j in range(i+1, nocc):
            gap = abs(eigenvals_imp[i] - eigenvals_imp[j])
            if(gap < 1e-10):
                n_deg += 1
        deg_info.append((i,n_deg+1))
        i += n_deg+1

    for i, pair in enumerate(deg_info):
        print i, pair
        if(pair[1] > 1):
            tmp = np.pad(eigenvecs_imp[:,pair[0]:pair[0]+pair[1]], ((0,numTotalOrbs-numImpOrbs),(0,0)), 'constant', constant_values=0 )
            subKin = libgen.frag_kin_sub( ints, tmp, pair[1] )
            subVnuc1 = libgen.frag_vnuc_sub(mol, ints, tmp, pair[1])
            mo_energy, mo_coeff = np.linalg.eigh(subKin+subVnuc1)
            tools.MatPrint( subKin+subVnuc1, "Hamilton")
            print mo_energy
            tmp1 = np.dot(tmp[:numImpOrbs,:],mo_coeff)
            eigenvecs_imp[:,pair[0]:pair[0]+pair[1]] = tmp1.copy()

    one = np.dot(eigenvecs_imp.T,eigenvecs_imp)
    assert(np.linalg.norm(one-np.eye(numImpOrbs)) <1e-9)
    '''
    return (eigenvals_imp, eigenvecs_imp)
