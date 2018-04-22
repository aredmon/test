"""
****************************************************
Module Name: Munkres
Author(s): Larry Gariepy
Version: 1.0
Date: 12/14/2015

    Module Description:
		This module contains an implementation of the Munkres optimal assignment algorithm (sometimes called the Hungarian assignment algorithm, 
		though there are several variants).

		Some of the comment descriptions are cited from Bob Pilgrim's website at Murray State U., 
		http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html, and this web page has a visual description
		of how the algorithm works.    
	Class Descriptions:
       -N/A
	Important Function Descriptions:
		-munkres():
				Given a cost matrix, run the optimal assignment algorithm and produce a matrix of assignments.  
					inputs: 1) cost_matrix - an mxn matrix of cost values for associating m workers to n tasks
					                        (RF objects to IR objects, etc.)
					outputs: assignment matrix of 0's and 1's where a '1' in the (i,j) entry indicates the optimal solution
					         includes the assignment of element i to element j.  Note, if m!=n, then there will not be an assignment of 
					         every object.  Rather, the solution will contain a number of assignments equal to min(m,n)


****************************************************
"""
import numpy as np

def munkres(cost_matrix):
	"""Execute the Munkres optimal assignment algorithm for the cost_matrix, and return an assignment_matrix to reflect the assignments"""
	(m,n) = cost_matrix.shape
	rot_flag = False
	
	if (m > n):
		cost_matrix = cost_matrix.T
		(m,n) = cost_matrix.shape
		rot_flag = True
	
	
	
	star = np.zeros( (m,n) , dtype = bool)
	prime = np.zeros( (m,n) , dtype = bool)
	colCov = np.zeros( n , dtype = bool)
	rowCov = np.zeros( m , dtype = bool)
	unCov = np.zeros( (m,n) , dtype = bool)
	step = 1
	itercount = 0
	
	while itercount < 100:
		itercount = itercount + 1
		## MUNKRES STEP 1: For each row of the matrix, find the smallest element and subtract it from every element in its row.
		##                 Go to step 2.
		if (step == 1):
			row_min = cost_matrix.min(1).reshape( (m,1) )  # take the minimum along the row dimension
			cost_matrix = cost_matrix - row_min
			step = 2
			
		## MUNKRES STEP 2: Find a zero (Z) in the resulting matrix.  If there is no starred zero in its row or column, star Z.  Repeat for each element in the matrix.
		##                 Go to step 3.
		elif (step == 2):
			zero = (cost_matrix == 0)  # save the location of the zeros
			(r,c) = zero.nonzero() # get indices of zeros -- yes, despite the use of `nonzero`
			__, r1 = np.unique(r, return_index = True)
			__, c1 = np.unique(c, return_index = True)
	
			ndx_star = np.intersect1d(r1,c1)
			star[r[ndx_star],c[ndx_star]] = True  # This has the effect of selecting a set of zeros such that every row and column with a zero has exactly one starred zero; 
			                                      # in the case of multiple zeros in a row/column, the first such zero is selected without loss of generality
			step = 3
			
		## MUNKRES STEP 3: Cover each column containign a starred zero.  If K columns are covered, the starred zeros describe a complete set of unique assignments (DONE).
		##                 If < K columns are covered, go to Step 4.
		elif (step == 3):
			#print("STEP3: star = ",star)
			colCov = np.any(star, axis = 0)  # logical matrix with True for a column containing a starred zero (referencing `star`)
			if sum(colCov) >= m:
				#print("Munkres: found ",m," assignments; DONE")
				break
			step = 4
			
		## MUNKRES STEP 4: Find a noncovered zero and prime it.  If there is no starred zero in the row containing the primed zero, Go to Step 5.
		##                 Otherwise, cover this row and uncover the column containing the starred zero.  Continue in this manner until there are no uncoverd zeros left.  
		##                 Save the smallest uncovered value and Go to Step 6.
		elif (step == 4):
			itercount2 = 0
			while itercount2 < 100:
				itercount2 = itercount2 + 1
				unCov = np.zeros( (m,n) , dtype = bool)
				
				#print("STEP4: unCov = ",unCov)
				## only need to update unCov if both np.invert(rowCov) and np.invert(colCov) have at least one True value
				unCov[np.ix_(np.invert(rowCov), np.invert(colCov))] = zero[np.ix_(np.invert(rowCov), np.invert(colCov))]
				if unCov.any(): # if there are uncovered zeros
					[row,col] = unCov.nonzero()
					if row.size > 1:  ## get indices of the first uncovered zero
						row = row[0]
						col = col[0]
					row = int(row) # need to explicitly convert to int types to avoid indexing problems of 1-element ndarrays later
					col = int(col)
					prime[row,col] = True
					
					Z = np.array([row, col]).reshape( (1,2) )
					
				else:  # If no uncovered zeros
					temp = cost_matrix[np.ix_(np.invert(rowCov), np.invert(colCov))]
					minNonCov = temp.min()
					step = 6
					break
				

				if np.any(star[row,:]):
					rowCov[row] = True
					colCov[star[row,:]] = False
				else:
					step = 5
					break
			if itercount2 >= 100:
				print("Number of iterations exceeded: B")
				return
			
		## MUNKRES STEP 5: Construct a series of alternating primed and starred zeros as follows:
		##                 Let Z0 represent the uncovered primed zero found in Step 4.  Let Z1 denote the starred zero in the column of Z0 (if any).  
		##                 Let Z2 denote the primed zero in the row of Z1 (there must always be one).  
		##                 Continue until the series terminates at a primed zero that has no starred zero in its column.
		##                 Unstar each starred zero of the series, star each primed zero of the series, erase all primes, and uncover every line in the matrix.
		##                 Go to Step 3.
		elif (step == 5):
			k = 0
			itercount2 = 0
			while itercount2 < 100: # while last primed Z(k) has starred zero in the same column
				itercount2 = itercount2 + 1
				#print ("STEP5 R: star = ", star)
				colZ = star[:,Z[k,1]]
				if np.any(colZ):
					Z = np.vstack([Z, np.append(colZ.nonzero()[0],Z[k,1])]) ## let Z(k+1)_denote the starred zero
					# 	Z(k+1, :) = [find(colZ), Z(k,2)];  ## NOTE: Here is the MATLAB syntax for the above, which is probably more readable; 
					rowZ = prime[Z[k+1,0], :]
					if np.any(rowZ):
						Z = np.vstack([Z,np.append([Z[k+1,0],],rowZ.nonzero()[0])])  # This is dynamically growing the Z array
						# Z[k+2,:] = np.append([Z[k+1,0],],rowZ.nonzero()[0]) ## Again, the MATLAB version
						k = k + 2
						
				else:
					# unstar each starred zero in Z
					star[Z[1:k:2,0],Z[1:k:2,1]] = False
					# star each primed zero in Z
					star[Z[0:k+1:2,0],Z[0:k+1:2,1]] = True
					# erase all primes
					prime = np.zeros( (m,n) , dtype = bool)
					# uncover all rows/columns
					colCov = np.zeros( n , dtype = bool)
					rowCov = np.zeros( m , dtype = bool)
					step = 3
					break
			if (itercount2 >= 100):
				print("Number of iterations exceeded: C")
				return
				
		
		## MUNKRES STEP 6: Add the value found in Step 4 to every element of each covered row, and subtract it from every element of each uncovered column.  
		##                 Return to Step 4 without altering any stars, primes, or covered lines.
		elif (step == 6):
			cost_matrix[rowCov,:] = cost_matrix[rowCov,:] + minNonCov  ## Add minimum uncovered value to covered rows
			cost_matrix[:,np.invert(colCov)] = cost_matrix[:,np.invert(colCov)] - minNonCov
			zero = (cost_matrix == 0)
			#print("STEP6: cost_matrix = ", cost_matrix)
			step = 4
		
	if itercount >= 100:
		print("Number of iterations exceeded: A")
		return
		
	if rot_flag:
		return star.T
	else:
		return star
	
	
def test_munkres():
	posA = np.array([[1, 2, 3], [0, -1, 3], [2, -1, 0], [1, 1, 1]])
	posB = posA + np.array([-3, 3, -3])
	cost_matrix = np.array([[np.linalg.norm(arr1-arr2,2)**2 for arr1 in posA] for arr2 in posB ])
	
	posA = np.array([[1,0],[0,1],[-1,0],[0,-1]]);
	posA_delta = np.array([[-1,0],[0,-1],[1,0],[0,1],]);
	offset = np.array([10.01, 10.01])
	posB = posA_delta + offset
	
	(r,c) = posA.shape
	cost_matrix = np.zeros( (r,r) )
	for i in range(r):
		for j in range(r):
			cost_matrix[i][j] = np.linalg.norm(posA[i,:] - posB[j,:],2)**2
	
	
	print("posA = \n", posA)
	print("posB = \n", posB)
	print("cost_matrix = \n",cost_matrix)
	print(munkres(cost_matrix))
	print("after: cost_matrix = ", cost_matrix)
	#print(munkres(cost_matrix.T))
	

if (__name__ == "__main__"):
	test_munkres()

