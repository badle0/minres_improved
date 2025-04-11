import numpy as np
from scipy.linalg import get_lapack_funcs
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import tensorflow as tf
from scipy.sparse.linalg._isolve.utils import make_system



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
    
    def bicgstab(self, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None,
             callback=None):
        """Use BIConjugate Gradient STABilized iteration to solve ``Ax = b``.

        Parameters
        ----------
        A : {sparse array, ndarray, LinearOperator}
            The real or complex N-by-N matrix of the linear system.
            Alternatively, `A` can be a linear operator which can
            produce ``Ax`` and ``A^T x`` using, e.g.,
            ``scipy.sparse.linalg.LinearOperator``.
        b : ndarray
            Right hand side of the linear system. Has shape (N,) or (N,1).
        x0 : ndarray
            Starting guess for the solution.
        rtol, atol : float, optional
            Parameters for the convergence test. For convergence,
            ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
            The default is ``atol=0.`` and ``rtol=1e-5``.
        maxiter : integer
            Maximum number of iterations.  Iteration will stop after maxiter
            steps even if the specified tolerance has not been achieved.
        M : {sparse array, ndarray, LinearOperator}
            Preconditioner for `A`. It should approximate the
            inverse of `A` (see Notes). Effective preconditioning dramatically improves the
            rate of convergence, which implies that fewer iterations are needed
            to reach a given error tolerance.
        callback : function
            User-supplied function to call after each iteration.  It is called
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

        Notes
        -----
        The preconditioner `M` should be a matrix such that ``M @ A`` has a smaller
        condition number than `A`, see [1]_ .

        References
        ----------
        .. [1] "Preconditioner", Wikipedia, 
            https://en.wikipedia.org/wiki/Preconditioner
        .. [2] "Biconjugate gradient stabilized method", 
            Wikipedia, https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.sparse import csc_array
        >>> from scipy.sparse.linalg import bicgstab
        >>> R = np.array([[4, 2, 0, 1],
        ...               [3, 0, 0, 2],
        ...               [0, 1, 1, 1],
        ...               [0, 2, 1, 0]])
        >>> A = csc_array(R)
        >>> b = np.array([-1, -0.5, -1, 2])
        >>> x, exit_code = bicgstab(A, b, atol=1e-5)
        >>> print(exit_code)  # 0 indicates successful convergence
        0
        >>> np.allclose(A.dot(x), b)
        True
        """
        # Prepare the system for solving
        A, M, x, b, postprocess = make_system(self.A_sparse, M, x0, b) 
        bnrm2 = np.linalg.norm(b)

        # Calculate the absolute tolerance
        atol = max(float(atol), float(rtol) * float(bnrm2))
        
        if bnrm2 == 0:
            return postprocess(b), 0 

        n = len(b)

        dotprod = np.vdot if np.iscomplexobj(x) else np.dot

        if maxiter is None:
            maxiter = n*10

        matvec = A.matvec
        psolve = M.matvec

        # Tolerance values for breakdown checks
        rhotol = np.finfo(x.dtype.char).eps**2
        omegatol = rhotol

        # Dummy values to initialize variables
        rho_prev, omega, alpha, p, v = None, None, None, None, None

        # Initial residual
        r = b - matvec(x) if x.any() else b.copy() # NON-LOOP: WIKI STEP 1
        rtilde = r.copy() # NON-LOOP WIKI STEP 2

        for iteration in range(maxiter):
            # Check for convergence
            if np.linalg.norm(r) < atol:
                print(f"Converged in {iteration + 1} iterations")
                return postprocess(x), 0

            # Compute rho
            rho = dotprod(rtilde, r) # NON-LOOP WIKI STEP 3 | LOOP STEP 11 
            if np.abs(rho) < rhotol:  # Check for rho breakdown
                return postprocess(x), -10

            if iteration > 0:
                if np.abs(omega) < omegatol:  # Check for omega breakdown
                    return postprocess(x), -11

                # Update beta and p
                beta = (rho / rho_prev) * (alpha / omega) # WIKI STEP 12
                # WIKI STEP 13: NEXT 3 LINES
                p -= omega*v
                p *= beta
                p += r # WIKI STEP 13
            else:  # First iteration
                s = np.empty_like(r)
                p = r.copy() # NON-LOOP WIKI STEP 4

            # Apply preconditioner
            phat = psolve(p)
            v = matvec(phat) # WIKI STEP 1
            rv = dotprod(rtilde, v)
            if rv == 0:
                return postprocess(x), -11
            alpha = rho / rv # WIKI STEP 3
            r -= alpha*v # WIKI STEP 4
            s[:] = r[:]  # WIKI STEP 4

            # MAIN CONVERGENCE CRITERIA: Check for convergence
            if np.linalg.norm(s) < atol: 
                # if s (step p-in p residual) is already small, then step and return x
                x += alpha*phat
                print(f"BiCGSTAB Converged in {iteration + 1} iterations")
                return postprocess(x), 0

            # continue if residual with step is not small enough
            # Apply preconditioner to s
            shat = psolve(s)
            t = matvec(shat) # WIKI STEP 6
            omega = dotprod(t, s) / dotprod(t, t) # WIKI STEP 7
            # update x by taking the steps
            x += alpha*phat # WIKI STEP 3 + 8
            x += omega*shat # WIKI STEP 8
            
            # r is overall residual after step in p and correction step in s
            r -= omega*t # WIKI STEP 9 (at this point, s = r)
            rho_prev = rho

            if callback:
                callback(x)

        else:  # for loop exhausted
            # Return incomplete progress
            print(f"Did not converge within the maximum number of iterations: {maxiter}")
            return postprocess(x), maxiter


    def deep_bicgstab(self, b, x0=None, model_predict=None, rtol=1e-5, atol=0., maxiter=None, M=None,
                    callback=None, fluid=False, verbose=True):
        """
        Use Deep Biconjugate Gradient Stabilized (Deep-BiCGSTAB) method with machine learning to predict the search direction.

        Parameters:
        -----------
        b : ndarray
            Right-hand side of the linear system.
        x0 : ndarray, optional
            Starting guess for the solution.
        model_predict : function
            A machine learning model that predicts the search direction given the residual.
        rtol, atol : float, optional
            Parameters for the convergence test. For convergence,
            ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        maxiter : integer, optional
            Maximum number of iterations.
        M : {sparse array, ndarray, LinearOperator}, optional
            Preconditioner for `A`.
        callback : function, optional
            User-supplied function to call after each iteration.
        fluid : bool, optional
            If True, the model will predict using the raw residual. If False, it uses the normalized residual.
        verbose : bool, optional
            If True, prints progress information during iterations.

        Returns:
        --------
        x : ndarray
            The converged solution.
        info : integer
            Convergence information.
        """
        # Prepare the system for solving
        A, M, x, b, postprocess = make_system(self.A_sparse, M, x0, b) 
        bnrm2 = np.linalg.norm(b)

        # Calculate the absolute tolerance
        atol = max(float(atol), float(rtol) * float(bnrm2))
        
        if bnrm2 == 0:
            return postprocess(b), 0 

        n = len(b)

        dotprod = np.vdot if np.iscomplexobj(x) else np.dot

        if maxiter is None:
            maxiter = n*10

        matvec = A.matvec
        psolve = M.matvec

        # Tolerance values for breakdown checks
        rhotol = np.finfo(x.dtype.char).eps**2
        omegatol = rhotol

        # Dummy values to initialize variables
        rho_prev, omega, alpha, p, v = None, None, None, None, None

        # Initial residual
        r = b - matvec(x) if x.any() else b.copy() # NON-LOOP: WIKI STEP 1
        rtilde = r.copy() # NON-LOOP WIKI STEP 2

        for iteration in range(maxiter):
            # Check for convergence
            if np.linalg.norm(r) < atol:
                print(f"Converged in {iteration + 1} iterations")
                return postprocess(x), 0

            # Compute rho
            rho = dotprod(rtilde, r) # NON-LOOP WIKI STEP 3 | LOOP STEP 11 
            if np.abs(rho) < rhotol:  # Check for rho breakdown
                return postprocess(x), -10

            if iteration > 0:
                if np.abs(omega) < omegatol:  # Check for omega breakdown
                    return postprocess(x), -11

                # Update beta and p
                beta = (rho / rho_prev) * (alpha / omega) # WIKI STEP 12
                p = model_predict(r) # Use the model to predict the search direction
                # WIKI STEP 13: NEXT 3 LINES
                # p -= omega*v
                # p *= beta
                # p += r # WIKI STEP 13
            else:  # First iteration
                s = np.empty_like(r)
                p = r.copy() # NON-LOOP WIKI STEP 4

            # Apply preconditioner
            phat = psolve(p)
            v = matvec(phat) # WIKI STEP 1
            rv = dotprod(rtilde, v)
            if rv == 0:
                return postprocess(x), -11
            alpha = rho / rv # WIKI STEP 3
            r -= alpha*v # WIKI STEP 4
            s[:] = r[:]  # WIKI STEP 4

            # MAIN CONVERGENCE CRITERIA: Check for convergence
            if np.linalg.norm(s) < atol: 
                # if s (step p-in p residual) is already small, then step and return x
                x += alpha*phat
                print(f"Converged in {iteration + 1} iterations")
                return postprocess(x), 0

            # continue if residual with step is not small enough
            # Apply preconditioner to s
            shat = psolve(s)
            t = matvec(shat) # WIKI STEP 6
            omega = dotprod(t, s) / dotprod(t, t) # WIKI STEP 7
            # update x by taking the steps
            x += alpha*phat # WIKI STEP 3 + 8
            x += omega*shat # WIKI STEP 8
            
            # r is overall residual after step in p and correction step in s
            r -= omega*t # WIKI STEP 9 (at this point, s = r)
            rho_prev = rho

            if callback:
                callback(x)

        else:  # for loop exhausted
            # Return incomplete progress
            print(f"Deep BiCGSTAB did not converge within the maximum number of iterations: {maxiter}")
            return postprocess(x), maxiter