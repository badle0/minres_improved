import numpy as np
from scipy.linalg import get_lapack_funcs
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import tensorflow as tf
from scipy.sparse.linalg._isolve.utils import make_system



# Solves non-symmetric and indefinite systems

class GeneralizedConjugateResidual:
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
    
    # based off of this link and following standard scipy implementation:
    # https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=7dcb4567b950c8260d9df8f6473de99cf97a69a3
    def gcr(self, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None,
        callback=None):
        """Use Generalized Conjugate Residual iteration to solve ``Ax = b``.

        Parameters
        ----------
        b : ndarray
            Right-hand side of the linear system.
        x0 : ndarray
            Starting guess for the solution.
        rtol, atol : float, optional
            Parameters for the convergence test. For convergence,
            ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
            The default is ``atol=0.`` and ``rtol=1e-5``.
        maxiter : integer
            Maximum number of iterations. Iteration will stop after maxiter
            steps even if the specified tolerance has not been achieved.
        M : {sparse array, ndarray, LinearOperator}
            Preconditioner for `A`. It should approximate the
            inverse of `A`.
        callback : function
            User-supplied function to call after each iteration. It is called
            as ``callback(xk)``, where ``xk`` is the current solution vector.

        Returns
        -------
        x : ndarray
            The converged solution.
        info : integer
            Provides convergence information:
                0  : successful exit
                >0 : convergence to tolerance not achieved, number of iterations
                <0 : parameter breakdown
        """
        A, M, x, b, postprocess = make_system(self.A_sparse, M, x0, b)
        n = len(b)
        bnrm2 = np.linalg.norm(b)
        atol = max(float(atol), float(rtol) * float(bnrm2))

        matvec = A.matvec
        psolve = M.matvec

        if bnrm2 == 0:
            return postprocess(b), 0

        if maxiter is None:
            maxiter = n * 10

        dotprod = np.vdot if np.iscomplexobj(x) else np.dot

        r = b - matvec(x) # step 1
        res_norm = np.linalg.norm(r)

        p_list = []
        Ap_list = []

        for k in range(maxiter):
            if res_norm < atol:
                print(f"Converged in {k} iterations")
                return postprocess(x), 0

            z = psolve(r) # search direction p
            Az = matvec(z)# Ap

            # not run on first iteration
            for i in range(len(p_list)):
                Ap_i = Ap_list[i]
                denom = dotprod(Ap_i, Ap_i)
                if denom == 0:
                    return postprocess(x), -10  # breakdown
                beta = dotprod(Az, Ap_i) / denom # step 6
                z -= beta * p_list[i] # step 7 p...
                Az -= beta * Ap_i # step 7 Ap...

            denom = dotprod(Az, Az) # step 3 denom
            if denom == 0:
                return postprocess(x), -11  # breakdown

            alpha = dotprod(r, Az) / denom # step 3
            x += alpha * z # step 4
            r -= alpha * Az# step 5

            p_list.append(z) # add to p list
            Ap_list.append(Az)

            res_norm = np.linalg.norm(r)

            if callback:
                callback(x)

        return postprocess(x), maxiter
