import numpy as np
from scipy.linalg import get_lapack_funcs
from scipy.sparse.linalg._isolve.utils import make_system
import warnings

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

    def multiply_A(self, x):
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

    # Based off of SciPy's GMRES implementation (v1.15.1)
    def gmres_old(self, b, x0=None, *, rtol=1e-5, atol=0., restart=None, maxiter=None, M=None,
          callback=None, callback_type=None):
        # Set default callback type if not provided
        if callback_type is None:
            callback_type = 'legacy'

        # Validate callback type
        if callback_type not in ('x', 'pr_norm', 'legacy'):
            raise ValueError(f"Unknown callback_type: {callback_type!r}")

        # Disable callback if none is provided
        if callback is None:
            callback_type = None

        # Initialize variables
        A = self.A_sparse
        x = np.zeros_like(b) if x0 is None else x0.copy()
        
        # Define preconditioner solve function
        psolve = lambda v: M @ v if M is not None else v
        n = len(b)
        bnrm2 = np.linalg.norm(b)

        # If the norm of b is zero, return the initial guess
        if bnrm2 == 0:
            return x, 0

        # Machine precision and dot product function
        eps = np.finfo(x.dtype.char).eps
        dotprod = np.vdot if np.iscomplexobj(x) else np.dot

        # Set default maximum iterations and restart value
        if maxiter is None:
            maxiter = n * 10

        if restart is None:
            restart = min(20, n)

        # Compute norm of preconditioned b
        Mb_nrm2 = np.linalg.norm(psolve(b))
        ptol_max_factor = 1.
        ptol = Mb_nrm2 * min(ptol_max_factor, atol / bnrm2)
        presid = 0.
        
        # Get LAPACK function for Givens rotations
        lartg = get_lapack_funcs('lartg', dtype=x.dtype)
        v = np.empty([restart+1, n], dtype=x.dtype)
        h = np.zeros([restart, restart+1], dtype=x.dtype)
        givens = np.zeros([restart, 2], dtype=x.dtype)
        
        inner_iter = 0

        # Main GMRES iteration loop
        for iteration in range(maxiter):
            if iteration == 0:
                r = b - (A @ x) if x.any() else b.copy()
                if np.linalg.norm(r) < atol:
                    return x, 0

            # Arnoldi process
            v[0, :] = psolve(r)
            tmp = np.linalg.norm(v[0, :])
            v[0, :] *= (1 / tmp)
            S = np.zeros(restart+1, dtype=x.dtype)
            S[0] = tmp
            breakdown = False
            
            for col in range(restart):
                av = A @ v[col, :]
                w = psolve(av)
                h0 = np.linalg.norm(w)
                for k in range(col+1):
                    tmp = dotprod(v[k, :], w)
                    h[col, k] = tmp
                    w -= tmp * v[k, :]

                h1 = np.linalg.norm(w)
                h[col, col + 1] = h1
                v[col + 1, :] = w[:]

                if h1 <= eps * h0:
                    h[col, col + 1] = 0
                    breakdown = True
                else:
                    v[col + 1, :] *= (1 / h1)

                # Apply Givens rotations
                for k in range(col):
                    c, s = givens[k, 0], givens[k, 1]
                    n0, n1 = h[col, [k, k+1]]
                    h[col, [k, k + 1]] = [c*n0 + s*n1, -s.conj()*n0 + c*n1]

                # Compute new Givens rotation
                c, s, mag = lartg(h[col, col], h[col, col+1])
                givens[col, :] = [c, s]
                h[col, [col, col+1]] = mag, 0
                tmp = -np.conjugate(s) * S[col]
                S[[col, col + 1]] = [c * S[col], tmp]
                presid = np.abs(tmp)
                inner_iter += 1

                # Call callback function if provided
                if callback_type in ('legacy', 'pr_norm'):
                    callback(presid / bnrm2)
                if callback_type == 'legacy' and inner_iter == maxiter:
                    break
                if presid <= ptol or breakdown:
                    break

            if h[col, col] == 0:
                S[col] = 0

            # Solve the upper triangular system
            y = np.zeros([col+1], dtype=x.dtype)
            y[:] = S[:col+1]
            for k in range(col, 0, -1):
                if y[k] != 0:
                    y[k] /= h[k, k]
                    tmp = y[k]
                    y[:k] -= tmp * h[k, :k]
            if y[0] != 0:
                y[0] /= h[0, 0]

            # Update the solution
            x += y @ v[:col+1, :]

            # Compute the residual
            r = b - (A @ x)
            rnorm = np.linalg.norm(r)
            if rnorm < atol or rnorm / bnrm2 < rtol:
                return x, iteration + inner_iter + 1

        return x, iteration + inner_iter + 1
    
    
    def gmres(self, b, x0=None, *, rtol=1e-5, atol=0., restart=None, maxiter=None, M=None,
          callback=None, callback_type=None):
        """
        Use Generalized Minimal RESidual iteration to solve ``Ax = b``.

        Parameters
        ----------
        A : {sparse array, ndarray, LinearOperator}
            The real or complex N-by-N matrix of the linear system.
            Alternatively, `A` can be a linear operator which can
            produce ``Ax`` using, e.g.,
            ``scipy.sparse.linalg.LinearOperator``.
        b : ndarray
            Right hand side of the linear system. Has shape (N,) or (N,1).
        x0 : ndarray
            Starting guess for the solution (a vector of zeros by default).
        atol, rtol : float
            Parameters for the convergence test. For convergence,
            ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
            The default is ``atol=0.`` and ``rtol=1e-5``.
        restart : int, optional
            Number of iterations between restarts. Larger values increase
            iteration cost, but may be necessary for convergence.
            If omitted, ``min(20, n)`` is used.
        maxiter : int, optional
            Maximum number of iterations (restart cycles).  Iteration will stop
            after maxiter steps even if the specified tolerance has not been
            achieved. See `callback_type`.
        M : {sparse array, ndarray, LinearOperator}
            Inverse of the preconditioner of `A`.  `M` should approximate the
            inverse of `A` and be easy to solve for (see Notes).  Effective
            preconditioning dramatically improves the rate of convergence,
            which implies that fewer iterations are needed to reach a given
            error tolerance.  By default, no preconditioner is used.
            In this implementation, left preconditioning is used,
            and the preconditioned residual is minimized. However, the final
            convergence is tested with respect to the ``b - A @ x`` residual.
        callback : function
            User-supplied function to call after each iteration.  It is called
            as ``callback(args)``, where ``args`` are selected by `callback_type`.
        callback_type : {'x', 'pr_norm', 'legacy'}, optional
            Callback function argument requested:
            - ``x``: current iterate (ndarray), called on every restart
            - ``pr_norm``: relative (preconditioned) residual norm (float),
                called on every inner iteration
            - ``legacy`` (default): same as ``pr_norm``, but also changes the
                meaning of `maxiter` to count inner iterations instead of restart
                cycles.

            This keyword has no effect if `callback` is not set.

        Returns
        -------
        x : ndarray
            The converged solution.
        info : int
            Provides convergence information:
                0  : successful exit
                >0 : convergence to tolerance not achieved, number of iterations

        See Also
        --------
        LinearOperator

        Notes
        -----
        A preconditioner, P, is chosen such that P is close to A but easy to solve
        for. The preconditioner parameter required by this routine is
        ``M = P^-1``. The inverse should preferably not be calculated
        explicitly.  Rather, use the following template to produce M::

        # Construct a linear operator that computes P^-1 @ x.
        import scipy.sparse.linalg as spla
        M_x = lambda x: spla.spsolve(P, x)
        M = spla.LinearOperator((n, n), M_x)

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.sparse import csc_array
        >>> from scipy.sparse.linalg import gmres
        >>> A = csc_array([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
        >>> b = np.array([2, 4, -1], dtype=float)
        >>> x, exitCode = gmres(A, b, atol=1e-5)
        >>> print(exitCode)            # 0 indicates successful convergence
        0
        >>> np.allclose(A.dot(x), b)
        True
        """
        if callback is not None and callback_type is None:
            # Warn about 'callback_type' semantic changes.
            # Probably should be removed only in far future, Scipy 2.0 or so.
            msg = ("scipy.sparse.linalg.gmres called without specifying "
                "`callback_type`. The default value will be changed in"
                " a future release. For compatibility, specify a value "
                "for `callback_type` explicitly, e.g., "
                "``gmres(..., callback_type='pr_norm')``, or to retain the "
                "old behavior ``gmres(..., callback_type='legacy')``"
                )
            warnings.warn(msg, category=DeprecationWarning, stacklevel=3)

        if callback_type is None:
            callback_type = 'legacy'

        if callback_type not in ('x', 'pr_norm', 'legacy'):
            raise ValueError(f"Unknown callback_type: {callback_type!r}")

        if callback is None:
            callback_type = None

        # Prepare the system for solving
        A, M, x, b, postprocess = make_system(self.A_sparse, M, x0, b)
        matvec = A.matvec
        psolve = M.matvec
        n = len(b)
        bnrm2 = np.linalg.norm(b)

        # Calculate the absolute tolerance
        atol = max(float(atol), float(rtol) * float(bnrm2))

        if bnrm2 == 0:
            return postprocess(b), 0

        eps = np.finfo(x.dtype.char).eps

        dotprod = np.vdot if np.iscomplexobj(x) else np.dot

        if maxiter is None:
            maxiter = n*10

        if restart is None:
            restart = 20
        restart = min(restart, n)

        Mb_nrm2 = np.linalg.norm(psolve(b))

        # Tolerance control for the inner iteration
        ptol_max_factor = 1.
        ptol = Mb_nrm2 * min(ptol_max_factor, atol / bnrm2)
        presid = 0.

        lartg = get_lapack_funcs('lartg', dtype=x.dtype)

        # Allocate internal variables
        v = np.empty([restart+1, n], dtype=x.dtype)
        h = np.zeros([restart, restart+1], dtype=x.dtype)
        givens = np.zeros([restart, 2], dtype=x.dtype)

        # Legacy iteration count
        inner_iter = 0

        for iteration in range(maxiter):
            if iteration == 0:
                r = b - matvec(x) if x.any() else b.copy()
                if np.linalg.norm(r) < atol:  # Check for convergence
                    return postprocess(x), 0

            # Apply preconditioner to the residual
            v[0, :] = psolve(r)
            tmp = np.linalg.norm(v[0, :])
            v[0, :] *= (1 / tmp)
            # RHS of the Hessenberg problem
            S = np.zeros(restart+1, dtype=x.dtype)
            S[0] = tmp

            breakdown = False
            for col in range(restart):
                av = matvec(v[col, :])
                w = psolve(av)

                # Modified Gram-Schmidt orthogonalization
                h0 = np.linalg.norm(w)
                for k in range(col+1):
                    tmp = dotprod(v[k, :], w)
                    h[col, k] = tmp
                    w -= tmp*v[k, :]

                h1 = np.linalg.norm(w)
                h[col, col + 1] = h1
                v[col + 1, :] = w[:]

                # Check for exact solution
                if h1 <= eps*h0:
                    h[col, col + 1] = 0
                    breakdown = True
                else:
                    v[col + 1, :] *= (1 / h1)

                # Apply past Givens rotations to current h column
                for k in range(col):
                    c, s = givens[k, 0], givens[k, 1]
                    n0, n1 = h[col, [k, k+1]]
                    h[col, [k, k + 1]] = [c*n0 + s*n1, -s.conj()*n0 + c*n1]

                # Get and apply current rotation to h and S
                c, s, mag = lartg(h[col, col], h[col, col+1])
                givens[col, :] = [c, s]
                h[col, [col, col+1]] = mag, 0

                # Update the RHS of the Hessenberg problem
                tmp = -np.conjugate(s)*S[col]
                S[[col, col + 1]] = [c*S[col], tmp]
                presid = np.abs(tmp)
                inner_iter += 1

                if callback_type in ('legacy', 'pr_norm'):
                    callback(presid / bnrm2)
                # Legacy behavior
                if callback_type == 'legacy' and inner_iter == maxiter:
                    break
                if presid <= ptol or breakdown:
                    break
            # arnoldi process to generate hessenberg matrix. 
            # Solve the upper triangular system
            if h[col, col] == 0:
                S[col] = 0

            y = np.zeros([col+1], dtype=x.dtype)
            y[:] = S[:col+1]
            for k in range(col, 0, -1):
                if y[k] != 0:
                    y[k] /= h[k, k]
                    tmp = y[k]
                    y[:k] -= tmp*h[k, :k]
            if y[0] != 0:
                y[0] /= h[0, 0]

            # Update the solution
            # v is the matrix of Orthogonalized Krylov Vectors
            # Each column v_i represents the i-th Krylov vector
            
            #result of upper triangular system y is multiplied by the orthogonalized Krylov vectors
            x += y @ v[:col+1, :]

            # Compute the new residual
            r = b - matvec(x)
            rnorm = np.linalg.norm(r)

            # Legacy exit
            if callback_type == 'legacy' and inner_iter == maxiter:
                return postprocess(x), 0 if rnorm <= atol else maxiter

            if callback_type == 'x':
                callback(x)

            if rnorm <= atol:
                break
            elif breakdown:
                # Reached breakdown (= exact solution), but the external
                # tolerance check failed. Bail out with failure.
                break
            elif presid <= ptol:
                # Inner loop passed but outer didn't
                ptol_max_factor = max(eps, 0.25 * ptol_max_factor)
            else:
                ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)

            ptol = presid * min(ptol_max_factor, atol / rnorm)

        info = 0 if (rnorm <= atol) else maxiter
        if info == 0:
            print(f"Converged in {iteration + 1} iterations")
        else:
            print(f"Did not converge within the maximum number of iterations: {maxiter}")
        return postprocess(x), info