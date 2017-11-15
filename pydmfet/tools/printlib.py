import numpy as np


def MatPrint(mat, Title, NRow=None, NCol=None, NLDA=None, NColPerLine = 6):

    if(NRow == None):
	NRow = mat.shape[0]

    if(NCol == None):
	NCol = mat.shape[1]

    if(NLDA == None):
	NLDA = NRow

    if (NRow < 1 or NCol < 1 or NLDA < NRow) :
	print "NRow = ", NRow,  ", NCol = ", NCol, ", NLDA = ", NLDA
	raise Exception('wrong matrix dimension!')

    NColPerLine = min(NCol, NColPerLine)


    PreCol = " "

    print " ",Title
    for ColFirst in range(0,NCol,NColPerLine): 
        ColLast = min(ColFirst+NColPerLine,NCol)

        #Print column numbers
	print PreCol,

        for j in range(ColFirst, ColLast):
	    print '{:12d}'.format(j+1),

	print ""

	#Print all the rows for the current columns
        for i in range(NRow):
            print '{:5d}'.format(i+1),
            for j in range(ColFirst, ColLast):
        	print '{:12.7f}'.format(mat[i,j]),
	    print ""
