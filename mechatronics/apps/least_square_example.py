import numpy as np

from mechatronics.estimation import calculate_least_squared_solution, calculate_weighted_least_squared
from mechatronics.estimation import recursive_least_squares

# Generate the H and Y matrices for the Measurement Dataset
Hmatrix = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]])
Ymatrix = np.array([[65, 65, 81, 92, 97]]).T
sol = calculate_least_squared_solution(Hmatrix, Ymatrix)

print(sol)

Rmatrix = np.diag([1, 2, 3, 2, 1])
sol, cvar = calculate_weighted_least_squared(Hmatrix, Rmatrix, Ymatrix)

print(sol)
print(cvar)

Xmatrix_prev = np.array([[8.0], [52.0]])
Pmatrix_prev = np.diag([1, 1])
Hmatrix = np.array([[2, 1]])
Rmatrix = np.array([[1]])
Ymatrix = np.array([[65]])
sol, cvar = recursive_least_squares(Xmatrix_prev, Pmatrix_prev, Hmatrix, Rmatrix, Ymatrix)
print(sol)
print(cvar)