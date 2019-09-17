import numpy as np
from argparse import ArgumentParser

def read_umat(nrow,filename):

    f = open(filename, "r")

    mat = np.ndarray((nrow,nrow),dtype = np.double)

    vec = []
    index = 0
    nline = 0
    for line in f:
        nline += 1
        elemts = line.split()
        if(nline%(nrow+1) == 1):
            if len(vec)>0:
                x = np.array(vec)
                x = x.astype(np.double)
                _nrow = x.size/6
                assert(_nrow==nrow)
                x = np.reshape(x,(nrow,-1))
                mat[:,index:index+6] = x.copy()
                index += 6
                vec = []
            continue
        else:
            for i in elemts[1:]:
                vec.append(i)

    x = np.array(vec)
    x = x.astype(np.double)
    x = np.reshape(x,(nrow,-1))
    mat[:,index:] = x.copy()


    f.close()

    return mat

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("n", help="nrow", type=int)
    parser.add_argument("-f", "--filename", dest="filename")
    args = parser.parse_args()

    read_umat(args.n, args.filename)
