import numpy as np

def fullP_to_fragP(P,dim_imp, dim):

    eigenvals, eigenvecs = np.linalg.eigh( P ) 
    tmp = -eigenvals
    idx = tmp.argsort()

    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:,idx]

    P_imp = np.zeros([dim, dim],dtype = float)
    P_bath = np.zeros([dim, dim],dtype = float)

    for i in range(dim):
	if(eigenvals[i] > 1.99):
	    isimp = classify_orb(eigenvecs[:,i],dim_imp)
	    if isimp :
		P_imp = P_imp + 2.0*np.outer(eigenvecs[:,i], eigenvecs[:,i]) 
	    else :
		P_bath = P_bath + 2.0*np.outer(eigenvecs[:,i], eigenvecs[:,i])
	elif(eigenvals[i] > 0.01):
	    print "fatal error"
	    assert(0==1)

    return (P_imp,P_bath)

def classify_orb(orb,dim_imp):

    sum_imp = 0.0
    for i in range(dim_imp):
	sum_imp += orb[i]*orb[i]

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

	Ne_imp = 0
        for cnt in range(len(occ)):
            if ( occ[ cnt ] < core_cutoff ):
                occ[ cnt ] = 0.0
            elif ( occ[ cnt ] > 2.0 - core_cutoff ):
                occ[ cnt ] = 2.0
		if(cnt < idx_imp):
		    Ne_imp += 1
            else:
                print "environment orbital occupation = ", occ[ cnt ]
                print "subspace construction failed!"
                assert( 0 == 1 )

        Nelec_core = int(round(np.sum( occ )))
        core1PDM_loc = np.dot( np.dot( loc2sub, np.diag( occ ) ), loc2sub.T )
        return (core1PDM_loc, Nelec_core, Ne_imp)


def construct_subspace(OneDM, impurityOrbs, threshold=1e-13):
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

    eigenvals_imp, eigenvecs_imp = np.linalg.eigh( imp1RDM )
    idx = np.maximum( -eigenvals_imp, eigenvals_imp - 2.0 ).argsort()
    tokeep_imp = np.sum( -np.maximum( -eigenvals_imp, eigenvals_imp - 2.0 )[idx] > threshold )

    eigenvals_imp = eigenvals_imp[idx]
    eigenvecs_imp = eigenvecs_imp[:,idx]

    #print eigenvals_imp
    #print eigenvecs_imp

    if(tokeep_imp < numImpOrbs):
        frozenEigVals_imp = -eigenvals_imp[tokeep_imp:]
        frozenEigVecs_imp = eigenvecs_imp[:,tokeep_imp:]
        idx = frozenEigVals_imp.argsort()
        eigenvecs_imp[:,tokeep_imp:] = frozenEigVecs_imp[:,idx]
        frozenEigVals_imp = -frozenEigVals_imp[idx]
        eigenvals_imp[tokeep_imp:] = frozenEigVals_imp

    #print eigenvals_imp
    #print eigenvecs_imp

    embeddingOrbs = 1 - impurityOrbs
    embeddingOrbs = np.matrix( embeddingOrbs )
    if (embeddingOrbs.shape[0] > 1):
        embeddingOrbs = embeddingOrbs.T # Now certainly row-like matrix (shape = 1 x len(vector))
    isEmbedding = np.dot( embeddingOrbs.T , embeddingOrbs ) == 1
    numEmbedOrbs = np.sum( embeddingOrbs )
    embedding1RDM = np.reshape( OneDM[ isEmbedding ], ( numEmbedOrbs , numEmbedOrbs ) )

    eigenvals_bath, eigenvecs_bath = np.linalg.eigh( embedding1RDM )
    idx = np.maximum( -eigenvals_bath, eigenvals_bath - 2.0 ).argsort() # Occupation numbers closest to 1 come first
    tokeep_bath = np.sum( -np.maximum( -eigenvals_bath, eigenvals_bath - 2.0 )[idx] > threshold )
    if (tokeep_bath > tokeep_imp):
        print "Throwing out ", tokeep_bath - tokeep_imp, "bath orbitals"
        tokeep_bath = tokeep_imp

    eigenvals_bath = eigenvals_bath[idx]
    eigenvecs_bath = eigenvecs_bath[:,idx]

    #print eigenvals_bath
    #print eigenvecs_bath


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

    #print loc2sub
    Occupations = np.hstack(( eigenvals_imp, eigenvals_bath ))
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

    return ( tokeep_imp, tokeep_bath, _Occupations, _loc2sub,eigenvals_imp, eigenvals_bath )
