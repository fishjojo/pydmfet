import sys
import numpy as np

def read_cube(filename, first):

    file = open(filename, 'r')

    dens = np.empty((10000,100))

    data = file.readlines()
    index = first-1
    for i in range(10000):
        for j in range(17):
	    if(j != 16):
		for k in range(6):
	            dens[i,j*6:min(j*6+6,100)] = float(data[index].split()[k])
	    else:
		for k in range(4):
		    dens[i,j*6:min(j*6+6,100)] = float(data[index].split()[k])

	    index += 1


    return dens


def write_dens(filename,  dens):

    file = open(filename, 'w')
    
    for i in range(10000):
	for j in range(17):
	    if(j != 16):
	        file.write("%.6g %.6g %.6g %.6g %.6g %.6g\n" % (dens[i,j*6] ,dens[i,j*6+1] ,dens[i,j*6+2] ,dens[i,j*6+3] ,dens[i,j*6+4] ,dens[i,j*6+5]) )
	    else:
		file.write("%.6g %.6g %.6g %.6g \n" % (dens[i,j*6] ,dens[i,j*6+1] ,dens[i,j*6+2] ,dens[i,j*6+3]) )


def main():

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    #file3 = sys.argv[3]
    first = int(sys.argv[3])


    dens1 = read_cube(file1, first)
    dens2 = read_cube(file2, first)
    #dens3 = read_cube(file3, first)

    diffP = dens1 - dens2# - dens3
    print np.max(diffP)
    write_dens("diffP.cube",diffP)

if __name__ == "__main__":
    main()
