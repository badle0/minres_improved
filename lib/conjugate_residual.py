import numpy as np
from scipy.linalg import get_lapack_funcs      # <- kept for parity
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import tensorflow as tf
from scipy.sparse.linalg._isolve.utils import make_system


class ConjugateResidualSparse:
    """
    Symmetric (possibly indefinite) solver based on Conjugate‑Residual (CR).

    Construction identical to your GeneralizedConjugateResidual class; only
    the `cr` routine at the end is new.
    """

    def __init__(self, A_sparse):
        if A_sparse.shape[0] != A_sparse.shape[1]:
            raise ValueError("A must be square.")
        self.n = A_sparse.shape[0]
        self.machine_tol = 1.0e-17
        self.A_sparse = A_sparse.copy().astype(np.float32)

    # ---------------- basic utilities (unchanged) ---------------- #
    def multiply_A_sparse(self, x):
        return self.A_sparse.dot(x)

    def norm(self, x):
        return np.linalg.norm(x)

    def dot(self, x, y):
        return np.dot(x, y)
    
    def lanczos_iteration_with_normalization_correction(self, b, max_it=10, tol=1.0e-10):
        Q = np.zeros([max_it, len(b)])
        if max_it==1:
            Q[0]=b.copy()
            Q[0]=Q[0]/self.norm(Q[0])
            return Q, [np.dot(Q[0],self.multiply_A_sparse(Q[0]))], []
        if max_it<=0:
            print("CR.lanczos_iteration: max_it can never be less than 0")
        if max_it > self.n:
            max_it = self.n
            print("max_it is reduced to the dimension ",self.n)
        diagonal = np.zeros(max_it)
        sub_diagonal = np.zeros(max_it)
        #norm_b = np.linalg.norm(b)
        norm_b = self.norm(b)
        Q[0] = b.copy()/norm_b
        #Q[1] = np.matmul(self.A, Q[0])
        Q[1] = self.multiply_A_sparse(Q[0])
        diagonal[0] = np.dot(Q[1],Q[0])
        Q[1] = Q[1] - diagonal[0]*Q[0]
        #sub_diagonal[0] = np.linalg.norm(Q[1])
        sub_diagonal[0] = self.norm(Q[1])
        Q[1] = Q[1]/sub_diagonal[0]
        if sub_diagonal[0]<tol:
            Q = np.resize(Q,[1,self.n])
            diagonal = np.resize(diagonal, [1])
            sub_diagonal = np.resize(sub_diagonal, [0])
            return Q, diagonal, sub_diagonal
        
        invariant_subspace = False
        it = 1
        while ((it<max_it-1) and (not invariant_subspace)):
            print(f"Iteration {it}, Norm of Q[{it}] = {self.norm(Q[it])}")
            Q[it+1] = self.multiply_A_sparse(Q[it])
            diagonal[it] = np.dot(Q[it],Q[it+1])            
            v = Q[it+1] - diagonal[it]*Q[it]-sub_diagonal[it-1]*Q[it-1]
            for j in range(it-1):
                v = v - Q[j]*self.dot(v, Q[j])
            Q[it+1] = v.copy()
            sub_diagonal[it] = self.norm(Q[it+1])
            Q[it+1] = Q[it+1]/sub_diagonal[it]
            if sub_diagonal[it] < tol:
                invariant_subspace = True
            it = it+1
            
        Q = np.resize(Q, [it+1,self.n])
        diagonal = np.resize(diagonal, [it+1])
        sub_diagonal = np.resize(sub_diagonal, [it])
        if not invariant_subspace:
            diagonal[it] = np.dot(Q[it], self.multiply_A_sparse(Q[it]))
        
        return Q, diagonal, sub_diagonal

    # ---------------------------------------------------------------- #
    #                      Conjugate‑Residual solver                    #
    # ---------------------------------------------------------------- #
    def cr(self, b, x0=None, *, rtol=1e-5, atol=0.0, maxiter=None, M=None,
           callback=None):
        """
        Solve Ax = b with the Conjugate‑Residual (CR) method.

        Parameters
        ----------
        b, x0, rtol, atol, maxiter, M, callback :  *same meaning*
            as in scipy`s iterative solvers.

        Returns
        -------
        x   : ndarray
        info: dict  with keys  'niter'  and  'res_norm'
        """
        A, M, x, b, postprocess = make_system(self.A_sparse, M, x0, b)
        n = len(b)
        bnrm2 = np.linalg.norm(b)
        atol = max(float(atol), float(rtol) * float(bnrm2))

        if bnrm2 == 0.0:
            return postprocess(b), {'niter': 0, 'res_norm': 0.0}

        if maxiter is None:
            maxiter = 10 * n

        matvec = A.matvec
        psolve = M.matvec
        dot    = np.vdot if np.iscomplexobj(b) else np.dot

        # --- initialise ---
        r   = b - matvec(x)
        p   = r.copy()
        Ap  = matvec(p)
        Ar  = matvec(r)
        res = np.linalg.norm(r)

        for k in range(maxiter):
            if res <= atol:
                return postprocess(x), {'niter': k, 'res_norm': res}

            denom = dot(Ap, Ap)
            if denom == 0.0:
                # A‑singular direction => breakdown
                return postprocess(x), {'niter': k, 'res_norm': res, 'flag': -11}

            alpha = dot(r, Ap) / denom

            # updates
            x   += alpha * p
            r_new = r - alpha * Ap
            Ar_new = matvec(r_new)

            res = np.linalg.norm(r_new)
            if callback is not None:
                callback(x)

            if res <= atol:
                return postprocess(x), {'niter': k+1, 'res_norm': res}

            beta = dot(r_new, Ar_new) / dot(r, Ar)

            p   = r_new + beta * p            # new direction
            Ap  = Ar_new + beta * Ap          # A p_{k+1}

            # shift for next iteration
            r, Ar = r_new, Ar_new

        # not converged
        return postprocess(x), {'niter': maxiter, 'res_norm': res}
