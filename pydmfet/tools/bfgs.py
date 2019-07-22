import numpy as np
from numpy import array, zeros, int32
from scipy.optimize import MemoizeJac

def minimize(func, x0, args=(), method='L-BFGS-B', jac=None, hess=None, callback=None, options=None):

    if(jac != True): raise ValueError("jac has to be True")

    x0 = np.asarray(x0, dtype=np.double)

    if not isinstance(args, tuple):
        args = (args,)

    meth = method.lower()

    if options is None:
        options = {}

    func = MemoizeJac(func)
    jac = func.derivative

    if meth == 'l-bfgs-b':
        return _minimize_lbfgsb(func, x0, args, jac, callback=callback, **options)




def _minimize_lbfgsb(fun, x0, args=(), jac=None, bounds=None,
                     disp=None, maxcor=10, ftol=2.2204460492503131e-09,
                     gtol=1e-5, eps=1e-8, maxfun=15000, maxiter=15000,
                     iprint=-1, callback=None, maxls=20, **unknown_options):

    m = maxcor
    epsilon = eps #useless
    pgtol = gtol

    x0 = asarray(x0).ravel()
    n, = x0.shape

    if bounds is None:
        bounds = [(None, None)] * n

    bounds = [(None if l == -np.inf else l, None if u == np.inf else u) for l, u in bounds]


    n_function_evals, fun = wrap_function(fun, ())

    if jac is not None:
        def func_and_grad(x):
            f = fun(x, *args)
            g = jac(x, *args)
            return f, g


    x = array(x0, np.double)
    f = array(0.0, np.double)
    g = zeros((n,), np.double)
    wa = zeros(2*m*n + 5*n + 11*m*m + 8*m, np.double)
    iwa = zeros(3*n, int32)
    task = zeros(1, 'S60')
    csave = zeros(1, 'S60')
    lsave = zeros(4, int32)
    isave = zeros(44, int32)
    dsave = zeros(29, np.double)


    task[:] = 'START'

    n_iterations = 0

    while 1:
        # x, f, g, wa, iwa, task, csave, lsave, isave, dsave = \
        _lbfgsb.setulb(m, x, low_bnd, upper_bnd, nbd, f, g, factr,
                       pgtol, wa, iwa, task, iprint, csave, lsave,
                       isave, dsave, maxls)
        task_str = task.tostring()
        if task_str.startswith(b'FG'):
            # The minimization routine wants f and g at the current x.
            # Note that interruptions due to maxfun are postponed
            # until the completion of the current minimization iteration.
            # Overwrite f and g:
            f, g = func_and_grad(x)
        elif task_str.startswith(b'NEW_X'):
            # new iteration
            n_iterations += 1
            if callback is not None:
                callback(np.copy(x))

            if n_iterations >= maxiter:
                task[:] = 'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'
            elif n_function_evals[0] > maxfun:
                task[:] = ('STOP: TOTAL NO. of f AND g EVALUATIONS '
                           'EXCEEDS LIMIT')
        else:
            break


def lbfgs_core(task):

    task_str = task.tostring()
    if task_str.startswith(b'START'):
        task[:] = 'FG'
        return
