'''
Created on Nov 28, 2017

@author: mark.lambrecht

Module Description:
    Extracts the specified column "i" from the input matrix "matrix". "i" is zero-based.
Inputs:
    matrix - 2D matrix from which to extract column
    i      - desired column index
Outputs:
    column "i" of matrix "matrix
'''


def column(matrix, i):
    return [row[i] for row in matrix]
