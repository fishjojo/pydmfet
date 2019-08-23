from scipy import optimize
from pydmfet.opt import newton
import numpy

class Memoize_Jac_Hess(object):

    def __init__(self, fun):
        self.fun = fun
        self.jac = None
        self.hess = None
        self.x = None

    def __call__(self, x, *args):
        self.x = numpy.asarray(x).copy()
        res = self.fun(x, *args)
        self.jac = res[1] if len(res) > 1 else None
        self.hess = res[2] if len(res) > 2 else None
        return res[0]

    def derivative(self, x, *args):
        if self.jac is not None and numpy.all(x == self.x):
            return self.jac
        else:
            self(x, *args)
            return self.jac

    def hessian(self, x, *args):
        if self.hess is not None and numpy.all(x == self.x):
            return self.hess
        else:
            self(x, *args)
            return self.hess



class OEP_Optimize():

    def __init__(self, opt_method, options, x0, func, func_args, jac=True, hess=None):

        self.method    = opt_method
        self.options   = options
        self.x0        = x0
        self.func      = func
        self.func_args = func_args
        self.jac       = jac
        self.hess      = hess


    def kernel(self, const_shift = None):

        if self.method.lower() == "newton":
            #in-house newton method
            return newton(self.func, self.x0, self.func_args, self.jac, self.hess, self.options, const_shift)

        #scipy.optimize functions
        if self.hess is not None:
            res = optimize.minimize(self.func, self.x0, args=self.func_args, method=self.method, jac=self.jac, hess=self.hess, \
                                    options=self.options)
        else:
            res = optimize.minimize(self.func, self.x0, args=self.func_args, method=self.method, jac=self.jac, \
                                    options=self.options)

        return res.x
