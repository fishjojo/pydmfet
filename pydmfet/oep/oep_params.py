
class OEPparams:

    def __init__(self, algorithm = '2011', oep_method = 'WY', \
		 opt_method = 'L-BFGS-B', ftol = 1e-10, gtol = 1e-5, maxit = 200, \
		 diffP_tol = 1e-5, outer_maxit = 200, l2_lambda = 0.0, oep_print = 0, svd_thresh=1e-2):

	self.algorithm = algorithm
        self.oep_method = oep_method
        self.opt_method = opt_method
	self.ftol = ftol
        self.gtol = gtol
        self.maxit = maxit
	self.outer_maxit = outer_maxit
	self.diffP_tol = diffP_tol

	self.oep_print = oep_print

	self.l2_lambda = l2_lambda

	self.svd_thresh = svd_thresh 
