from pydmfet.oep import oep_main, oepparam
from pydmfet.oep.oepparam import *

def OEP(embedobj,params,*args):

    oep_method = params.oep_method

    if(oep_method == 'WY'):

        return oep_main.OEP(embedobj,params, *args)
