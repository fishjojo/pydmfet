import numpy as np

def find_efermi(eigenvals, smear_sigma, NAlpha, Norb):

    maxit = 200
    toll = 1.0e-12
    step=max(2.0*smear_sigma,1.0)
    i_homo = NAlpha-1

    convgd = False
    while True:
        if i_homo == Norb:
            raise Exception("failed to find fermi energy!")

        e_homo = eigenvals[i_homo]
        emed = e_homo
        emax = emed + step
        emin = emed - step

        attempts=0
        while True:
            attempts += 1

            fmax = fzero(eigenvals, emax, smear_sigma, NAlpha, Norb)[0]
            fmed = fzero(eigenvals, emed, smear_sigma, NAlpha, Norb)[0]
            fmin = fzero(eigenvals, emin, smear_sigma, NAlpha, Norb)[0]

            if (fmax*fmin < 0.0):
                break
            elif(attempts > maxit):
                raise Exception("fail!")
            else:
                emax += step
                emin -= step

        attempts=0
        mo_occ = None
        while True:
            attempts += 1
            if(fmax*fmed > 0.0):
                emax = emed
                fmax = fmed
            else:
                emin = emed
                fmin = fmed

            if(attempts < 15 or abs(fmax-fmin) < 0.0):
                emed=0.5*(emin+emax)
            else:
                emed=-fmin*(emax-emin)/(fmax-fmin)+emin
    
            fmed, mo_occ = fzero(eigenvals, emed, smear_sigma, NAlpha, Norb)

            if(abs(fmed) < toll or abs(emin-emax) < toll*0.1 ):
                convgd = True
                break

            if(attempts > maxit):
                #raise Exception("fail 2!")
                i_homo += 1
                break

        if convgd:
            break

    return emed, mo_occ


def fzero(eigenvals, efermi, smear_sigma, NAlpha, Norb):

    mo_occ = np.zeros((Norb))
    for i in range(Norb):
        e_i = eigenvals[i]
        expo = (e_i-efermi)/smear_sigma
        if(expo > 100.0):
            mo_occ[i] = 0.0
        else:
            mo_occ[i] = 1.0/(1.0 + np.exp(expo) )

        if(mo_occ[i] < np.finfo(float).eps):
            mo_occ[i] = 0.0

        if(mo_occ[i] >1.0): mo_occ[i] = 1.0
        elif(mo_occ[i] <0.0): mo_occ[i] = 0.0


    ne = np.sum(mo_occ)
    zero = NAlpha - ne
    return (zero, mo_occ)


def entropy_corr(mo_occ, smear_sigma=0.0):

    if mo_occ is None:
        return 0.0

    toll = 1e-8
    S = 0.0
    if(smear_sigma >= 1e-8):
        nmo = mo_occ.size
        for i in range(nmo):
            occ_i = mo_occ[i]/2.0 #closed shell
            if(occ_i > toll and occ_i < 1.0-toll):
                S += occ_i * np.log(occ_i) + (1.0-occ_i) * np.log(1.0-occ_i)
            else:
                S += 0.0

    energy = 2.0*S*smear_sigma
    print ('entropy correction = ',energy)

    return energy

