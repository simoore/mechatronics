import numpy as np

from mechatronics.estimation import calculate_least_squared_solution, calculate_weighted_least_squared

# Generate the H and Y matrices for the Measurement Dataset
Hmatrix = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]])
Ymatrix = np.array([[65, 65, 81, 92, 97]]).T
sol = calculate_least_squared_solution(Hmatrix, Ymatrix)

print(sol)

Rmatrix = np.diag([1, 2, 3, 2, 1])
sol, cvar = calculate_weighted_least_squared(Hmatrix, Rmatrix, Ymatrix)

print(sol)
print(cvar)