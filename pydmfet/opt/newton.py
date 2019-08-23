import numpy as np
from pydmfet.libcpp import invert_mat_sigular_thresh
import copy
from scipy.optimize import line_search
from scipy.optimize.optimize import  MemoizeJac


def line_scipy(func,xk,args,jac,fk,gk,hess,const_shift,svd_thresh):

    size = gk.size
    invers_hess = invert_mat_sigular_thresh(hess,svd_thresh)
    pk = -np.dot(invers_hess,gk)

    proj = np.eye(size)-np.outer(const_shift,const_shift)
    pk = np.dot(proj,pk)

    alpha,fc,gc,new_fval,old_fval,new_slope=line_search(func, jac, xk, pk, gfk=gk, old_fval=fk, args=args,maxiter=20)

    if(alpha is None) : return None

    print ("line search step size = " , alpha)
    x_new = xk + alpha*pk
    return x_new


def backtrack(func,x,args,jac,f,g,hess,const_shift,svd_thresh,alpha=2.0,tau=0.5,c=0.1):

    info=0
    size = g.size

    invers_hess = invert_mat_sigular_thresh(hess,svd_thresh)
    #invers_hess = np.linalg.pinv(hess,rcond=1e-8)
    jX = -np.dot(invers_hess,g)

    proj = np.eye(size)-np.outer(const_shift,const_shift)
    jX = np.dot(proj,jX)


    m = np.dot(jX,g)
    t = -c*m

    x_new = np.zeros(size,dtype=np.double)

    alpha_loc = copy.copy(alpha)
    inc = False
    it = 0
    gmax_0 = np.amax(np.absolute(g))
    gmax_1 = 0.0
    while(1):
        it += 1

        if it == 2:
            if gmax_0/gmax_1 < 0.01:
                alpha_loc *= gmax_0/gmax_1*100.0

        #print "line search step size = " , alpha_loc
        jY = alpha_loc * jX

        max_move = np.amax(np.absolute(jY))
        if(max_move > 5.0):
            alpha_loc *= 5.0/max_move
            jY = alpha_loc * jX

        print ("line search step size = " , alpha_loc)

        x_new = x + jY
        f_new = func(x_new, *args)
        g_new = jac(x_new, *args)

        if it == 1:
            gmax_1 = np.amax(np.absolute(g_new))

        max_jY = np.amax(np.absolute(jY))
        if(max_jY < 1e-8):
            info =1
            print ("step size less than 1e-8")
            break

        obj = alpha_loc*t
        print (" line search debug obj = ",obj)
        if(f-f_new < obj):
            alpha_loc *= tau
            #if(inc == True):
                #x_new = x + alpha_loc * jX
                #break
        elif(f-f_new >= obj):
            break
            #if(inc == False and it != 1):
                #break
            #alpha_loc /= tau
            #inc = True

    if(info == 0):
        return x_new
    else:
        return None



def newton(func,x0,args,jac,hessian,options, const_shift):

    x0 = np.asarray(x0, dtype=np.double)
    if not isinstance(args, tuple):
        args = (args,)

    x = x0.copy()
    alpha = 1.0
    it = 0
    f_old=1e6

    maxit = options["maxiter"]
    gtol  = options["gtol"]
    ftol  = options["ftol"]
    svd_thresh = options["svd_thresh"]
    while(1 and it<maxit):
        it += 1

        #call func
        f = func(x,*args)
        g = jac(x,*args)
        hess = hessian(x,*args)

        gmax = np.amax(np.absolute(g))
        if(gmax<gtol or abs(f-f_old)<ftol):
            break

        #line search
        svd_loc = 0.01
        if(gmax<1e-2): svd_loc = max(1e-3, svd_thresh)
        if(gmax<1e-3): svd_loc = max(1e-4, svd_thresh)
        if(gmax<1e-4): svd_loc = max(1e-5, svd_thresh)
        if(gmax<1e-5): svd_loc = max(1e-6, svd_thresh)
        svd_loc = svd_thresh
        #if(it==1): alpha_loc = 0.5
        #else: alpha_loc = alpha
        x = backtrack(func,x,args,jac,f,g,hess,const_shift,svd_loc,alpha=alpha)
        #x=line_scipy(func,x,args,jac,f,g,hess,const_shift,svd_loc)
        if(x is None):
            raise Exception(" Line search failed!")

        f_old = copy.copy(f)

        if(it==maxit):
            print ("max iteration reached in newton opt!")

    return x

