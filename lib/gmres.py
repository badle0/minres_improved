import numpy as np
from scipy.linalg import get_lapack_funcs
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import tensorflow as tf

class GMRESSparse:
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

    def gmres(self, b, x0, max_iter=100, rtol=1e-10, atol = 0., restart=None):
        """
        Solves the linear system Ax = b using the GMRES algorithm.

        Args:
            b (np.array): The right-hand side vector.
            x0 (np.array): The initial guess for the solution.
            tol (float): The tolerance for convergence.
            max_iter (int): The maximum number of iterations.
            restart (int): Number of iterations between restarts.

        Returns:
            tuple: The solution vector x and the array of residuals.
        """
        n = len(b)
        bnrm2 = np.linalg.norm(b)
        atol = max(float(atol), float(rtol) * float(bnrm2))

        if bnrm2 == 0:
            return x0, 0

        eps = np.finfo(x0.dtype.char).eps

        if restart is None:
            restart = 20
        restart = min(restart, n)
        x = x0.copy()

        lartg = get_lapack_funcs('lartg', dtype=x.dtype)

        v = np.zeros((n, restart + 1))
        h = np.zeros((restart + 1, restart))
        givens = np.zeros([restart, 2])
        res = []

        # legacy iteration count
        inner_iter = 0

        for it in range(max_iter):
            if it == 0:
                r = b - self.multiply_A(x) if x.any() else b.copy
                rnorm_initial = self.norm(r)
                if rnorm_initial < atol:
                    res = res + [rnorm_initial]
                    return x0, res

            v[0, :] = r
            tmp = np.linalg.pinv(v[0, :])
            v[0, :] *= (1 / tmp)
            S = np.zeros(restart + 1)
            S[0] = tmp

            for j in range(restart):
                av = self.multiply_A(v[j, :])
                w = av

                # Modified Gram-Schmidt
                h0 = self.norm(w)
                for i in range(j + 1):
                    tmp = self.dot(v[i, :], w)
                    h[i, j] = tmp
                    w -= tmp*v[i, :]

                h1 = self.norm(w)
                h[i, i + 1] = h1
                v[i + 1, :] = w[:]

                # Exact solution indicator
                if h1 <= eps*h0:
                    h[i, i+1] = 0
                else:
                    v[i + 1, :] *= (1 / h1)

                # apply past Givens rotations to current h column
                for k in range(j):
                    c, s = givens[k, 0], givens[k, 1]
                    n0, n1 = h[j, [k, k + 1]]
                    h[j, [k, k + 1]] = [c * n0 + s * n1, -s.conj() * n0 + c * n1]

                # get and apply current rotation to h and S
                c, s, mag = lartg(h[j, j], h[j, j + 1])
                givens[j, :] = [c, s]
                h[j, [j, j + 1]] = mag, 0

                # S[j+1] component is always 0
                tmp = -np.conjugate(s) * S[j]
                S[[j, j + 1]] = [c * S[j], tmp]
                presid = np.abs(tmp)
                inner_iter += 1

            # Solve h(j, j) upper triangular system and allow pseudo-solve
            # singular cases as in (but without the f2py copies):
            # y = trsv(h[:j+1, :j+1].T, S[:j+1])

            if h[j, j] == 0:
                S[j] = 0

            y = np.zeros([j + 1], dtype=x.dtype)
            y[:] = S[:j + 1]
            for k in range(j, 0, -1):
                if y[k] != 0:
                    y[k] /= h[k, k]
                    tmp = y[k]
                    y[:k] -= tmp * h[k, :k]
            if y[0] != 0:
                y[0] /= h[0, 0]

            x += self.dot(v[:j + 1, :], y)

            r = b - self.multiply_A(x)
            rnorm = self.norm(r)
            res.append(rnorm)
            if rnorm <= atol:
                print(f"GMRES converged in {it + 1} iterations with residual {rnorm}.")
                return x, res

        print(f"GMRES stopped after {max_iter} iterations with residual {rnorm}.")
        return x, res