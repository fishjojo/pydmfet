
class OEPparams:

    def __init__(self, algorithm = '2011', oep_method = 'WY', opt_method = 'L-BFGS-B', \
                 diffP_tol = 1e-5, outer_maxit = 50, l2_lambda = 0.0, oep_print = 0, \
                 options = {'maxiter':200, 'ftol':1e-8, 'gtol':1e-5, 'disp':True, 'svd_thresh':1e-2}, \
                 umat_init_method = 'zero'):

        self.algorithm = algorithm
        self.oep_method = oep_method
        self.opt_method = opt_method
        self.outer_maxit = outer_maxit
        self.diffP_tol = diffP_tol
        self.options = options

        self.l2_lambda = l2_lambda
        self.oep_print = oep_print
        self.umat_init_method = umat_init_method

