from .oep_params import *
from .func_wuyang import *
from . import oep_main

def OEP(embedobj,params,*args):

    return oep_main.OEP(embedobj,params, *args)
