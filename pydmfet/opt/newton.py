import numpy as np
from pydmfet.libcpp import invert_mat_sigular_thresh
import copy

def backtrack(func,x,args,f,g,hess,svd_thresh,alpha=2.0,tau=0.5,c=0.1):

    info=0
    size = g.size

    invers_hess = invert_mat_sigular_thresh(hess,svd_thresh)

    jX = np.zeros(size,dtype=np.double)
    jX = np.dot(invers_hess,g)
    #X_len = np.linalg.norm(jX)
    jX = -1.0*jX

    m = np.dot(jX,g)
    t = -c*m

    jY = np.zeros(size,dtype=np.double)
    x_new = np.zeros(size,dtype=np.double)

    alpha_loc = copy.copy(alpha)
    while(1):
	jY = jX.copy()
	print "line search step size = " , alpha_loc
	jY = alpha_loc * jY

	x_new = x + jY
        f_new, g_new = func(x_new, *args)

	max_jY = np.amax(np.absolute(jY))
        if(max_jY < 1e-8):
	    info =1
	    print "step size less than 1e-8"
	    break

	obj = alpha_loc*t
        print " line search debug obj = ",obj
        alpha_loc *= tau
	if(f-f_new >= obj):
	    break

    if(info == 0):
        return x_new
    else:
	return None



def newton(func,x0,args=(),gtol=1e-4,maxit=50,svd_thresh=0.01):

    x0 = np.asarray(x0, dtype=np.double)
    if not isinstance(args, tuple):
        args = (args,)

    x = x0.copy()
    it = 0
    while(1 and it<maxit):
	it += 1
	#call func
	f,g,hess = func(x,*args,calc_hess=True)
	gmax = np.amax(np.absolute(g))
	if(gmax<gtol):
	    break

	#line search
	x = backtrack(func,x,args,f,g,hess,svd_thresh)
	if(x is None):
	    raise Exception(" Line search failed!")

    return x

