import numpy as np
from scipy.linalg import get_lapack_funcs
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import tensorflow as tf


# BiCGSTAB = biconjugate gradient stabilized method
# Solves non-symmetric and indefinite systems

class BiCGSTABSparse:
    def __init__(self, A_sparse):
        """
        Initializes the GMRESSparse class with a sparse matrix A.

        Args:
            A_sparse (scipy.sparse.spmatrix): The input sparse matrix A.
        """
        if A_sparse.shape[0] != A_sparse.shape[1]:
            print("A is not a square matrix!")
        self.n = A_sparse.shape[0]
        self.machine_tol = 1.0e-17
        A_sp = A_sparse.copy()
        self.A_sparse = A_sp.astype(np.float32)

    def multiply_A(self,x):
        #return np.matmul(self.A,x)
        return self.A_sparse.dot(x)

    def multiply_A_sparse(self, x):
        """
        Multiplies the sparse matrix A by vector x.

        Args:
            x (np.array): The input vector.

        Returns:
            np.array: The result of A * x.
        """
        return self.A_sparse.dot(x)

    def norm(self, x):
        """
         Computes the norm of vector x.

         Args:
             x (np.array): The input vector.

         Returns:
             float: The norm of x.
         """
        return np.linalg.norm(x)

    def dot(self, x, y):
        """
        Computes the dot product of vectors x and y.

        Args:
            x (np.array): The first vector.
            y (np.array): The second vector.

        Returns:
            float: The dot product of x and y.
        """
        return np.dot(x, y)

    def multi_norm(self, xs):
        """
        Computes the norms of multiple vectors.

        Args:
            xs (np.array): The input vectors.

        Returns:
            np.array: The norms of the input vectors.
        """
        norms = np.zeros(xs.shape[0])
        for i in range(xs.shape[0]):
            norms[i] = self.norm(xs[i])
        return norms

    def multi_dot(self, xs, ys):
        """
        Computes the dot products of multiple pairs of vectors.

        Args:
            xs (np.array): The first set of vectors.
            ys (np.array): The second set of vectors.

        Returns:
            np.array: The dot products of the input vector pairs.
        """
        dots = np.zeros(xs.shape[0])
        for i in range(xs.shape[0]):
            dots[i] = self.dot(xs[i], ys[i])
        return dots

    def get_zero_rows(self):
        """
        Gets the rows of the sparse matrix A that are entirely zero.

        Returns:
            np.array: The indices of the zero rows.
        """
        return np.where(self.A_sparse.getnnz(1) == 0)[0]
        
    def bicgstab(self, b, x0, max_iter=1000, rtol=1e-10, atol=0.0):
        """
        Solves the linear system Ax = b using the BiConjugate Gradient Stabilized (BiCGSTAB) method.

        Args:
            b (np.array): The right-hand side vector.
            x0 (np.array): The initial guess for the solution.
            rtol (float): The relative tolerance for convergence.
            atol (float): The absolute tolerance for convergence.
            max_iter (int): The maximum number of iterations.

        Returns:
            tuple: The solution vector x and the array of residuals.
        """
        n = len(b)
        bnrm2 = np.linalg.norm(b)
        if bnrm2 == 0:
            return x0, [0]

        atol = max(float(atol), float(rtol) * float(bnrm2))

        x = x0.copy()
        r = b - self.multiply_A(x)
        rhat = r.copy()
        p = r.copy()
        phat = p.copy()
        rho_old = 1.0
        alpha = 1.0
        omega = 1.0

        residual_norm = np.linalg.norm(r)
        res = []
        res.append(residual_norm)
        
        # if initial guess is within the residual tolerance
        if residual_norm <= rtol * bnrm2 + atol:
                print(f"BiCGSTAB converged in {1} iterations with residual {residual_norm}.")
                return x, res

        for iteration in range(max_iter):
            rho_new = np.dot(rhat, r)
            beta = (rho_new / rho_old) * (alpha / omega)
            p = r + beta * (p - omega * phat)
            phat = self.multiply_A(p)

            # Compute alpha
            alpha = rho_new / np.dot(rhat, phat)
            s = r - alpha * phat

            # Check for convergence before applying second correction
            residual_norm = np.linalg.norm(s)
            res.append(residual_norm)

            if residual_norm <= rtol * bnrm2 + atol:
                print(f"BiCGSTAB converged in {iteration + 1} iterations with residual {residual_norm}.")
                return x + alpha * p, res

            # Compute omega (for second correction)
            t = self.multiply_A(s)
            omega = np.dot(t, s) / np.dot(t, t)
            x += alpha * p + omega * s

            # Update residual for next iteration
            r = s - omega * t

            rho_old = rho_new

        print(f"BiCGSTAB stopped after {max_iter} iterations with residual {residual_norm}.")
        return x, res

            