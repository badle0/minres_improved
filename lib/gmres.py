import numpy as np
from scipy.linalg import get_lapack_funcs

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

    def gmres(self, b, x0=None, *, rtol=1e-5, atol=0., restart=None, maxiter=None, M=None,
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