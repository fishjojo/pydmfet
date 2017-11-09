
class OEPparams:

    def __init__(self, algorithm = '2011', oep_method = 'WY', opt_method = 'BFGS', ftol = 1e-6, gtol = 1e-5, maxit = 50, diffP_tol = 1e-6, outer_maxit = 50, oep_print = 0):

	self.algorithm = algorithm
        self.oep_method = oep_method
        self.opt_method = opt_method
	self.ftol = ftol
        self.gtol = gtol
        self.maxit = maxit
	self.outer_maxit = outer_maxit
	self.diffP_tol = diffP_tol

	self.oep_print = oep_print
