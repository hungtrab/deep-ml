import numpy as np
import math

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    def solve_quadratic(a, b, c):
        delta = b**2 - 4*a*c
        x1 = (-b + math.sqrt(max(delta, 0))) / (2*a)
        x2 = (-b - math.sqrt(max(delta, 0))) / (2*a)
        return x1, x2

    # Step 1: Compute A^T A
    B = A.T @ A

    # Step 2: Solve characteristic equation
    # lambda^2 - trace(B)*lambda + det(B) = 0
    l1, l2 = solve_quadratic(
        1,
        -(B[0, 0] + B[1, 1]),
        B[0, 0]*B[1, 1] - B[0, 1]**2
    )

    s1, s2 = math.sqrt(max(l1, 0)), math.sqrt(max(l2, 0))
    S = np.array([s1, s2])

    # Step 3: One Jacobi rotation to diagonalize B
    theta = 0.5 * math.atan2(2 * B[0, 1], B[0, 0] - B[1, 1])

    c, s = math.cos(theta), math.sin(theta)
    V = np.array([
        [c, -s],
        [s,  c]
    ])

    # Ensure singular values and vectors have matching order
    projected = np.diag(V.T @ B @ V)
    if projected[0] < projected[1]:
        V = V[:, [1, 0]]

    # Step 4: Compute U = A V Sigma^-1
    U = np.zeros((2, 2))
    eps = 1e-12

    for i in range(2):
        if S[i] > eps:
            U[:, i] = (A @ V[:, i]) / S[i]

    # Handle rank-deficient matrices
    if S[0] <= eps:
        U = np.eye(2)
    elif S[1] <= eps:
        U[:, 0] /= np.linalg.norm(U[:, 0])
        U[:, 1] = np.array([-U[1, 0], U[0, 0]])

    return U, S, V.T