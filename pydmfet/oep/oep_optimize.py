from scipy import optimize
from pydmfet.opt import newton

class OEP_Optimize():

    def __init__(self, opt_method, options, x0, func, func_args, jac=True, hess=None):

        self.method    = opt_method
        self.options   = options
        self.x0        = x0
        self.func      = func
        self.func_args = func_args
        self.jac       = jac
        self.hess      = hess


    def kernel(self):

        if self.method.lower() == "newton":
            #in-house newton method
            return newton(self.func, self.x0, args=self.func_args, options=self.options)

        #scipy.optimize functions
        if self.hess is not None:
            res = optimize.minimize(self.func, self.x0, args=self.func_args, method=self.method, jac=self.jac, hess=self.hess, \
                                    options=self.options)
        else:
            res = optimize.minimize(self.func, self.x0, args=self.func_args, method=self.method, jac=self.jac, \
                                    options=self.options)

        return res.x
