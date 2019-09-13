from .oep_params import *
from .func_wuyang import *
from .func_leastsq import *
from .init_dens_par import *
from . import oep_main

def OEP(embedobj,params,*args):

    return oep_main.OEP(embedobj,params, *args)
