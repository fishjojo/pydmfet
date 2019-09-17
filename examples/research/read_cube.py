import sys
import numpy as np

def read_cube(filename, first):

    file = open(filename, 'r')

    dens = np.empty((10000,100))

    data = file.readlines()
    index = first-1
    for i in range(10000):
        for j in range(17):
            x = np.array(data[index].split())
            y = x.astype(float)
            dens[i,j*6:min(j*6+6,100)] = y
            index += 1

    return dens


def write_dens(filename,  dens):

    file = open(filename, 'w')
    
    for i in range(10000):
        for j in range(17):
            if(j != 16):
                file.write("  %.6E %.6E %.6E %.6E %.6E %.6E\n" % (dens[i,j*6] ,dens[i,j*6+1] ,dens[i,j*6+2] ,dens[i,j*6+3] ,dens[i,j*6+4] ,dens[i,j*6+5]) )
            else:
                file.write("  %.6E %.6E %.6E %.6E \n" % (dens[i,j*6] ,dens[i,j*6+1] ,dens[i,j*6+2] ,dens[i,j*6+3]) )


def main():

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    first = int(sys.argv[3])

    dens1 = read_cube(file1, first)
    dens2 = read_cube(file2, first)

    diff_dens = dens1-dens2

    write_dens("diff_dens.cube",diff_dens)

if __name__ == "__main__":
    main()
