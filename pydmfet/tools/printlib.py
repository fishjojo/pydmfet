import numpy as np


def VecPrint(vec, Title, NColPerLine = 5):

    print (" ",Title)

    NElem = vec.size
    NRow = NElem//NColPerLine

    indx = 0
    for iRow in range(NRow):
        for iCol in range(NColPerLine):
            print ("{a:12.6e} ".format(a=vec[indx]), end='')
            indx += 1
        print ("")

    if(NElem%NColPerLine >0):
        NCol = NElem%NColPerLine
        for iCol in range(NCol):
            print ("{a:12.6e} ".format(a=vec[indx]), end='')
            indx += 1
        print ("")


def MatPrint(mat, Title, NRow=None, NCol=None, NLDA=None, NColPerLine = 6):

    if(NRow == None):
        NRow = mat.shape[0]

    if(NCol == None):
        NCol = mat.shape[1]

    if(NLDA == None):
        NLDA = NRow

    if (NRow < 1 or NCol < 1 or NLDA < NRow) :
        print ("NRow = ", NRow,  ", NCol = ", NCol, ", NLDA = ", NLDA)
        raise Exception('wrong matrix dimension!')

    NColPerLine = min(NCol, NColPerLine)


    PreCol = " "

    print (" ",Title)
    for ColFirst in range(0,NCol,NColPerLine): 
        ColLast = min(ColFirst+NColPerLine,NCol)

        #Print column numbers
        print (PreCol,end='')

        for j in range(ColFirst, ColLast):
            print ("{a:12d} ".format(a=j+1), end='')

        print ("")

        #Print all the rows for the current columns
        for i in range(NRow):
            print ("{a:5d} ".format(a=i+1), end='')
            for j in range(ColFirst, ColLast):
                print ("{a:12.7f} ".format(a=mat[i,j]), end='')
            print ("")
