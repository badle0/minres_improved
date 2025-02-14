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
            if rho_old == 0:
                print(f"ERROR: Rho breakdown @ iter={iteration}, rho_old=0")
                return x, res
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
    
    
    
    # def bicgstab_sp(self, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None,
    #          callback=None):
    #     """Use BIConjugate Gradient STABilized iteration to solve ``Ax = b``.

    #     Parameters
    #     ----------
    #     A : {sparse array, ndarray, LinearOperator}
    #         The real or complex N-by-N matrix of the linear system.
    #         Alternatively, `A` can be a linear operator which can
    #         produce ``Ax`` and ``A^T x`` using, e.g.,
    #         ``scipy.sparse.linalg.LinearOperator``.
    #     b : ndarray
    #         Right hand side of the linear system. Has shape (N,) or (N,1).
    #     x0 : ndarray
    #         Starting guess for the solution.
    #     rtol, atol : float, optional
    #         Parameters for the convergence test. For convergence,
    #         ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
    #         The default is ``atol=0.`` and ``rtol=1e-5``.
    #     maxiter : integer
    #         Maximum number of iterations.  Iteration will stop after maxiter
    #         steps even if the specified tolerance has not been achieved.
    #     M : {sparse array, ndarray, LinearOperator}
    #         Preconditioner for `A`. It should approximate the
    #         inverse of `A` (see Notes). Effective preconditioning dramatically improves the
    #         rate of convergence, which implies that fewer iterations are needed
    #         to reach a given error tolerance.
    #     callback : function
    #         User-supplied function to call after each iteration.  It is called
    #         as ``callback(xk)``, where ``xk`` is the current solution vector.

    #     Returns
    #     -------
    #     x : ndarray
    #         The converged solution.
    #     info : integer
    #         Provides convergence information:
    #             0  : successful exit
    #             >0 : convergence to tolerance not achieved, number of iterations
    #             <0 : parameter breakdown

    #     Notes
    #     -----
    #     The preconditioner `M` should be a matrix such that ``M @ A`` has a smaller
    #     condition number than `A`, see [1]_ .

    #     References
    #     ----------
    #     .. [1] "Preconditioner", Wikipedia, 
    #         https://en.wikipedia.org/wiki/Preconditioner
    #     .. [2] "Biconjugate gradient stabilized method", 
    #         Wikipedia, https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method

    #     Examples
    #     --------
    #     >>> import numpy as np
    #     >>> from scipy.sparse import csc_array
    #     >>> from scipy.sparse.linalg import bicgstab
    #     >>> R = np.array([[4, 2, 0, 1],
    #     ...               [3, 0, 0, 2],
    #     ...               [0, 1, 1, 1],
    #     ...               [0, 2, 1, 0]])
    #     >>> A = csc_array(R)
    #     >>> b = np.array([-1, -0.5, -1, 2])
    #     >>> x, exit_code = bicgstab(A, b, atol=1e-5)
    #     >>> print(exit_code)  # 0 indicates successful convergence
    #     0
    #     >>> np.allclose(A.dot(x), b)
    #     True
    #     """
    #     A, M, x, b, postprocess = make_system(A, M, x0, b) 
    #     bnrm2 = np.linalg.norm(b)

    #     atol, _ = _get_atol_rtol('bicgstab', bnrm2, atol, rtol) 

    #     if bnrm2 == 0:
    #         return postprocess(b), 0 

    #     n = len(b)

    #     dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    #     if maxiter is None:
    #         maxiter = n*10

    #     matvec = A.matvec
    #     psolve = M.matvec

    #     # These values make no sense but coming from original Fortran code
    #     # sqrt might have been meant instead.
    #     rhotol = np.finfo(x.dtype.char).eps**2
    #     omegatol = rhotol

    #     # Dummy values to initialize vars, silence linter warnings
    #     rho_prev, omega, alpha, p, v = None, None, None, None, None

    #     r = b - matvec(x) if x.any() else b.copy()
    #     rtilde = r.copy()

    #     for iteration in range(maxiter):
    #         if np.linalg.norm(r) < atol:  # Are we done?
    #             return postprocess(x), 0

    #         rho = dotprod(rtilde, r)
    #         if np.abs(rho) < rhotol:  # rho breakdown
    #             return postprocess(x), -10

    #         if iteration > 0:
    #             if np.abs(omega) < omegatol:  # omega breakdown
    #                 return postprocess(x), -11

    #             beta = (rho / rho_prev) * (alpha / omega)
    #             p -= omega*v
    #             p *= beta
    #             p += r
    #         else:  # First spin
    #             s = np.empty_like(r)
    #             p = r.copy()

    #         phat = psolve(p)
    #         v = matvec(phat)
    #         rv = dotprod(rtilde, v)
    #         if rv == 0:
    #             return postprocess(x), -11
    #         alpha = rho / rv
    #         r -= alpha*v
    #         s[:] = r[:]

    #         if np.linalg.norm(s) < atol:
    #             x += alpha*phat
    #             return postprocess(x), 0

    #         shat = psolve(s)
    #         t = matvec(shat)
    #         omega = dotprod(t, s) / dotprod(t, t)
    #         x += alpha*phat
    #         x += omega*shat
    #         r -= omega*t
    #         rho_prev = rho

    #         if callback:
    #             callback(x)

    #     else:  # for loop exhausted
    #         # Return incomplete progress
    #         return postprocess(x), maxiter

                