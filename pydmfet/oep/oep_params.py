


class OEPparams:

    def __init__(self, oep_method = 'WY', opt_method = 'BFGS', ftol = 1e-6, gtol = 1e-4, maxit = 50):

        self.oep_method = oep_method
        self.opt_method = opt_method
	self.ftol = ftol
        self.gtol = gtol
        self.maxit = maxit


