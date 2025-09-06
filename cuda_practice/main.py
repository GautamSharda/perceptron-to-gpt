# MATMUL assuming: square matrices with 1D representations.
# A = [[1,2,3],
#     [4,5,6],
#     [7,8,9]]
A_flat = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# A_order = [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9]
# B = [[1,2,3],
#     [4,5,6],
#     [7,8,9]]
B_flat = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# B_order = [1, 4, 7, 2, 5, 8, 3, 6, 9, 1, 4, 7, 2, 5, 8, 3, 6, 9, 1, 4, 7, 2, 5, 8, 3, 6, 9]
def matmul(A_flat, B_flat, width):
    P = []
    # Since I want to fill out the product matrix row-wise, by definition, the first matrix should be the outer loop.
    for vi_a in range(width): # Iterating through rows of A. A[vi_a] is the row vector in A.
        for vi_b in range(width): # Iterating through cols of B. B[vi_b] is the column vector in B.
            dp = 0
            for i in range(width): # Iterator for columns in A[vi_a] and rows in B[vi_b].
                # I want element (vi_a, i) in A and (i, vi_b) in B.
                # Translating to 1D: index = row*width + col -- index in A = vi_a*width + i and index in B = i*width + vi_b
                dp += A_flat[vi_a*width + i] * B_flat[i*width + vi_b]
            P.append(dp)   
    return P
import math
print(matmul(A_flat, B_flat, int(math.sqrt(len(A_flat)))))