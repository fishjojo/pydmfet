from pydmfet.oep import oepwy, oepparam
from pydmfet.oep.oepparam import *

def OEP(embedobj,params,*args):

    oep_method = params.oep_method

    if(oep_method == 'WY'):

        return oepwy.OEPWY(embedobj,params, *args)
