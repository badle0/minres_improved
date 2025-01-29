#  Checklist: Navigate better coding arrangements that integrates more helper functions to improve code conciseness

import numpy as np
from numpy import inner, zeros, inf, finfo
from math import sqrt
import scipy.sparse as sparse
from scipy.linalg import get_lapack_funcs
import scipy.sparse.linalg as spla
import tensorflow as tf


class MINRESSparse:
    def __init__(self, A_sparse):
        """
        Initializes the MINRESSparse class with a sparse matrix A.

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

    def lanczos_iteration(self, b, max_it=10, tol=1.0e-10):
        """
        Performs the Lanczos iteration to find the eigenvalues and eigenvectors.

        Args:
            b (np.array): The input vector.
            max_it (int): The maximum number of iterations.
            tol (float): The tolerance for convergence.

        Returns:
            tuple: The eigenvectors, diagonal, and sub-diagonal of the tridiagonal matrix.
        """
        Q = np.zeros([max_it, len(b)])
        if max_it == 1:
            Q[0] = b.copy()
            Q[0] = Q[0] / self.norm(Q[0])
            return Q, [np.dot(Q[0], self.multiply_A_sparse(Q[0]))], []
        if max_it <= 0:
            print("CG.lanczos_iteration: max_it can never be less than 0")
        if max_it > self.n:
            max_it = self.n
            print("max_it is reduced to the dimension ", self.n)
        diagonal = np.zeros(max_it)
        sub_diagonal = np.zeros(max_it)
        # norm_b = np.linalg.norm(b)
        norm_b = self.norm(b)
        Q[0] = b.copy() / norm_b
        # Q[1] = np.matmul(self.A, Q[0])
        Q[1] = self.multiply_A_sparse(Q[0])
        diagonal[0] = np.dot(Q[1], Q[0])
        Q[1] = Q[1] - diagonal[0] * Q[0]
        # sub_diagonal[0] = np.linalg.norm(Q[1])
        sub_diagonal[0] = self.norm(Q[1])
        Q[1] = Q[1] / sub_diagonal[0]
        if sub_diagonal[0] < tol:
            Q = np.resize(Q, [1, self.n])
            diagonal = np.resize(diagonal, [1])
            sub_diagonal = np.resize(sub_diagonal, [0])
            return Q, diagonal, sub_diagonal

        invariant_subspace = False
        it = 1
        while ((it < max_it - 1) and (not invariant_subspace)):
            Q[it + 1] = self.multiply_A_sparse(Q[it])
            diagonal[it] = np.dot(Q[it], Q[it + 1])
            Q[it + 1] = Q[it + 1] - diagonal[it] * Q[it] - sub_diagonal[it - 1] * Q[it - 1]
            sub_diagonal[it] = self.norm(Q[it + 1])
            Q[it + 1] = Q[it + 1] / sub_diagonal[it]
            if sub_diagonal[it] < tol:
                invariant_subspace = True
            it = it + 1

        Q = np.resize(Q, [it + 1, self.n])
        diagonal = np.resize(diagonal, [it + 1])
        sub_diagonal = np.resize(sub_diagonal, [it])
        if not invariant_subspace:
            diagonal[it] = np.dot(Q[it], self.multiply_A_sparse(Q[it]))

        return Q, diagonal, sub_diagonal

    def lanczos_iteration_with_normalization_correction(self, b, max_it=10, tol=1.0e-10):
        """
        Performs the Lanczos iteration with normalization correction.

        Args:
            b (np.array): The input vector.
            max_it (int): The maximum number of iterations.
            tol (float): The tolerance for convergence.

        Returns:
            tuple: The eigenvectors, diagonal, and sub-diagonal of the tridiagonal matrix.
        """
        Q = np.zeros([max_it, len(b)])
        if max_it == 1:
            Q[0] = b.copy()
            Q[0] = Q[0] / self.norm(Q[0])
            return Q, [np.dot(Q[0], self.multiply_A_sparse(Q[0]))], []
        if max_it <= 0:
            print("MR.lanczos_iteration: max_it can never be less than 0")
        if max_it > self.n:
            max_it = self.n
            print("max_it is reduced to the dimension ", self.n)
        diagonal = np.zeros(max_it)
        sub_diagonal = np.zeros(max_it)
        # norm_b = np.linalg.norm(b)
        norm_b = self.norm(b)
        Q[0] = b.copy() / norm_b
        # Q[1] = np.matmul(self.A, Q[0])
        Q[1] = self.multiply_A_sparse(Q[0])
        diagonal[0] = np.dot(Q[1], Q[0])
        Q[1] = Q[1] - diagonal[0] * Q[0]
        # sub_diagonal[0] = np.linalg.norm(Q[1])
        sub_diagonal[0] = self.norm(Q[1])
        Q[1] = Q[1] / sub_diagonal[0]
        if sub_diagonal[0] < tol:
            Q = np.resize(Q, [1, self.n])
            diagonal = np.resize(diagonal, [1])
            sub_diagonal = np.resize(sub_diagonal, [0])
            return Q, diagonal, sub_diagonal

        invariant_subspace = False
        it = 1
        while ((it < max_it - 1) and (not invariant_subspace)):
            print(f"Iteration {it}, Norm of Q[{it}] = {self.norm(Q[it])}")
            Q[it + 1] = self.multiply_A_sparse(Q[it])
            diagonal[it] = np.dot(Q[it], Q[it + 1])
            v = Q[it + 1] - diagonal[it] * Q[it] - sub_diagonal[it - 1] * Q[it - 1]
            for j in range(it - 1):
                v = v - Q[j] * self.dot(v, Q[j])
            Q[it + 1] = v.copy()
            sub_diagonal[it] = self.norm(Q[it + 1])
            Q[it + 1] = Q[it + 1] / sub_diagonal[it]
            if sub_diagonal[it] < tol:
                invariant_subspace = True
            it = it + 1

        Q = np.resize(Q, [it + 1, self.n])
        diagonal = np.resize(diagonal, [it + 1])
        sub_diagonal = np.resize(sub_diagonal, [it])
        if not invariant_subspace:
            diagonal[it] = np.dot(Q[it], self.multiply_A_sparse(Q[it]))

        return Q, diagonal, sub_diagonal

    def mult_diag_precond(self, x):
        """
        Applies diagonal preconditioning to vector x.

        Args:
            x (np.array): The input vector.

        Returns:
            np.array: The preconditioned vector.
        """
        y = np.zeros(x.shape)
        for i in range(self.n):
            if self.A_sparse[i, i] > self.machine_tol:
                y[i] = x[i] / self.A_sparse[i, i]
        return y

    def mult_precond_method1(self, x, Q, lambda_):
        """
        Applies a preconditioning method using Ritz vectors and values.

        Args:
            x (np.array): The input vector.
            Q (np.array): The Ritz vectors.
            lambda_ (np.array): The Ritz values.

        Returns:
            np.array: The preconditioned vector.
        """
        y = np.copy(x)
        for i in range(Q.shape[0]):
            qTx = np.dot(Q[i], x)
            y = y + qTx * (1 / lambda_[i] - 1.0) * Q[i]
        return y

    # minres algorithm transformed from minres source file in scipy
    def minres(self, b, x0, max_iter=1000, tol=1e-10, verbose=False):
        """
        Solves the linear system Ax = b using the MINRES algorithm.

        Args:
            b (np.array): The right-hand side vector.
            x0 (np.array): The initial guess for the solution.
            tol (float): The tolerance for convergence.
            max_iter (int): The maximum number of iterations.
            verbose (bool): If True, print the progress.

        Returns:
            tuple: The solution vector x and the array of residuals.
        """
        n = len(b)

        msg = [ ' beta1 = 0.  The exact solution is x0              ',   # 0
                ' A solution to Ax = b was found, given rtol        ',   # 1
                ' A least-squares solution was found, given rtol    ',   # 2
                ' Reasonable accuracy achieved, given eps           ',   # 3
                ' The iteration limit was reached                   ']   # 4

        istop = 0
        itn = 0
        Anorm = 0
        Acond = 0
        rnorm = 0
        ynorm = 0

        x = x0.copy()
        if x0 is None:
            r1 = b.copy()
        else:
            r1 = b - self.multiply_A(x0)
        y = r1

        beta1 = self.dot(r1, y)
        if beta1 == 0:
            return x0, beta1
        
        beta1 = np.sqrt(beta1)

        oldb = 0
        beta = beta1
        dbar = 0
        epsln = 0
        qrnorm = beta1
        phibar = beta1
        rhs1 = beta1
        rhs2 = 0
        tnorm2 = 0
        gmax = 0
        gmin = np.finfo(float).max
        cs = -1
        sn = 0
        w = np.zeros(n)
        w2 = np.zeros(n)
        r2 = r1

        res_arr = [beta1]

        for itn in range(max_iter):
            s = 1.0/beta
            v = s*y
            y = self.multiply_A(v)
            if itn >= 2:
                y = y - (beta / oldb) * r1
            alpha = self.dot(v, y)
            y = y - (alpha / beta) * r1
            r1 = r2
            r2 = y
            y = r2
            oldb = beta
            beta = self.dot(r2, y)
            if beta < 0:
                raise ValueError('non-symmetric matrix')
            beta = sqrt(beta)
            tnorm2 += alpha ** 2 + oldb ** 2 + beta ** 2
            oldeps = epsln
            delta = cs * dbar + sn * alpha
            gbar = sn * dbar - cs * alpha
            epsln = sn * beta
            dbar = - cs * beta
            root = np.sqrt(gbar ** 2 + dbar ** 2)
            Arnorm = phibar * root
            gamma = np.sqrt(gbar ** 2 + beta ** 2)
            gamma = max(gamma, np.finfo(float).eps)
            cs = gbar / gamma
            sn = beta / gamma
            phi = cs * phibar
            phibar = sn * phibar
            denom = 1.0 / gamma
            w1 = w2
            w2 = w
            w = (v - oldeps * w1 - delta * w2) * denom
            x = x + phi * w
            gmax = max(gmax, gamma)
            gmin = min(gmin, gamma)
            z = rhs1 / gamma
            rhs1 = rhs2 - delta * z
            rhs2 = - epsln * z
            Anorm = np.sqrt(tnorm2)
            ynorm = self.norm(x)
            epsa = Anorm * np.finfo(float).eps
            epsx = Anorm * ynorm * np.finfo(float).eps
            epsr = Anorm * ynorm * tol
            diag = gbar
            if diag == 0:
                diag = epsa
            qrnorm = phibar
            rnorm = qrnorm
            if ynorm == 0 or Anorm == 0:
                test1 = inf
            else:
                test1 = rnorm / (Anorm*ynorm)
            if Anorm == 0:
                test2 = inf
            else:
                test2 = root / Anorm
            Acond = gmax / gmin

            if istop == 0:
                t1 = 1 + test1      # These tests work if rtol < eps
                t2 = 1 + test2
                if t2 <= 1:
                    istop = 2
                if t1 <= 1:
                    istop = 1
                if itn >= max_iter:
                    istop = 4
                if epsx >= beta1:
                    istop = 3
                # if rnorm <= epsx   : istop = 2
                # if rnorm <= epsr   : istop = 1
                if test2 <= tol:
                    istop = 2
                if test1 <= tol:
                    istop = 1
            
            if verbose:
                print(f"Iteration {itn+1}, test1: {test1}")
                res_arr.append(rnorm)

            if istop != 0:
                break

        print("Traditional MINRES stopped" + f' istop   =  {istop:3g}               itn   ={itn+1:5g}')
        print("Traditional MINRES stopped" + f' Anorm   =  {Anorm:12.4e}      Acond =  {Acond:12.4e}')
        print("Traditional MINRES stopped" + f' rnorm   =  {rnorm:12.4e}      ynorm =  {ynorm:12.4e}')
        print("Traditional MINRES stopped" + f' Arnorm  =  {Arnorm:12.4e}')
        print("Traditional MINRES stopped" + msg[istop])

        return x, res_arr

    def deepminres(self, b, x0, model_predict, max_iter=100, tol=1e-10, verbose=False):
        """
        Solves the linear system Ax = b using the MINRES algorithm with a deep learning model to predict search direction.

        Args:
            b (np.array): The right-hand side vector.
            x0 (np.array): The initial guess for the solution.
            model_predict (function): A function that predicts the search direction.
            tol (float): The tolerance for convergence.
            max_iter (int): The maximum number of iterations.
            verbose (bool): If True, print the progress.

        Returns:
            tuple: The solution vector x and the array of residuals.
        """
        n = len(b)
        msg = [ ' beta1 = 0.  The exact solution is x0              ',   # 0
                ' A solution to Ax = b was found, given rtol        ',   # 1
                ' A least-squares solution was found, given rtol    ',   # 2
                ' Reasonable accuracy achieved, given eps           ',   # 3
                ' The iteration limit was reached                   ']   # 4
        istop = 0
        itn = 0
        Anorm = 0
        Acond = 0
        rnorm = 0
        ynorm = 0
        x = x0.copy()
        if x0 is None:
            r1 = b.copy()
        else:
            r1 = b - self.multiply_A(x0)
        y = r1
        beta1 = self.dot(r1, y)
        if beta1 == 0:
            return x0, beta1
        oldb = 0
        beta = beta1
        dbar = 0
        epsln = 0
        qrnorm = beta1
        phibar = beta1
        rhs1 = beta1
        rhs2 = 0
        tnorm2 = 0
        gmax = 0
        gmin = np.finfo(float).max
        cs = -1
        sn = 0
        w = np.zeros(n)
        w2 = np.zeros(n)
        r2 = r1
        res_arr = [beta1]

        for itn in range(max_iter):
            # Use the model to predict the search direction
            v = model_predict(y)
            y = self.multiply_A(v)
            if itn >= 2:
                y = y - (beta / oldb) * r1
            alpha = self.dot(v, y)
            y = y - (alpha / beta) * r1
            r1 = r2
            r2 = y
            y = r2
            oldb = beta
            beta = self.dot(r2, y)
            if beta < 0:
                raise ValueError('non-symmetric matrix')
            beta = sqrt(beta)
            tnorm2 += alpha ** 2 + oldb ** 2 + beta ** 2
            oldeps = epsln
            delta = cs * dbar + sn * alpha
            gbar = sn * dbar - cs * alpha
            epsln = sn * beta
            dbar = - cs * beta
            root = np.sqrt(gbar ** 2 + dbar ** 2)
            Arnorm = phibar * root
            gamma = np.sqrt(gbar ** 2 + beta ** 2)
            gamma = max(gamma, np.finfo(float).eps)
            cs = gbar / gamma
            sn = beta / gamma
            phi = cs * phibar
            phibar = sn * phibar
            denom = 1.0 / gamma
            w1 = w2
            w2 = w
            w = (v - oldeps * w1 - delta * w2) * denom
            x = x + phi * w
            gmax = max(gmax, gamma)
            gmin = min(gmin, gamma)
            z = rhs1 / gamma
            rhs1 = rhs2 - delta * z
            rhs2 = - epsln * z
            Anorm = np.sqrt(tnorm2)
            ynorm = self.norm(x)
            epsa = Anorm * np.finfo(float).eps
            epsx = Anorm * ynorm * np.finfo(float).eps
            epsr = Anorm * ynorm * tol
            diag = gbar
            if diag == 0:
                diag = epsa
            qrnorm = phibar
            rnorm = qrnorm
            if ynorm == 0 or Anorm == 0:
                test1 = inf
            else:
                test1 = rnorm / (Anorm*ynorm)
            if Anorm == 0:
                test2 = inf
            else:
                test2 = root / Anorm
            Acond = gmax / gmin

            if istop == 0:
                t1 = 1 + test1      # These tests work if rtol < eps
                t2 = 1 + test2
                if t2 <= 1:
                    istop = 2
                if t1 <= 1:
                    istop = 1
                if itn >= max_iter:
                    istop = 4
                if epsx >= beta1:
                    istop = 3
                # if rnorm <= epsx   : istop = 2
                # if rnorm <= epsr   : istop = 1
                if test2 <= tol:
                    istop = 2
                if test1 <= tol:
                    istop = 1
            
            if verbose:
                print(f"Iteration {itn+1}, test1: {test1}")
                res_arr.append(rnorm)

            if istop != 0:
                break

        print("Deep MINRES stopped" + f' istop   =  {istop:3g}               itn   ={itn+1:5g}')
        print("Deep MINRES stopped" + f' Anorm   =  {Anorm:12.4e}      Acond =  {Acond:12.4e}')
        print("Deep MINRES stopped" + f' rnorm   =  {rnorm:12.4e}      ynorm =  {ynorm:12.4e}')
        print("Deep MINRES stopped" + f' Arnorm  =  {Arnorm:12.4e}')
        print("Deep MINRES stopped" + msg[istop])

        return x, res_arr

    def create_diagonal_preconditioner(self):
        """
        Creates a diagonal (Jacobi) preconditioner for the matrix self.A_sparse.
        Wherever the original matrix's diagonal is zero, this code sets the 
        preconditioner diagonal to 1.
        """

        diag_elements = self.A_sparse.diagonal()  # shape (n,)
        zero_mask = (diag_elements == 0.0)
        nonzero_mask = ~zero_mask
        M_inv_diag = np.empty_like(diag_elements)  # shape (n,)
        M_inv_diag[nonzero_mask] = 1.0 / diag_elements[nonzero_mask]
        M_inv_diag[zero_mask] = 1.0

        # 6. Build the sparse diagonal matrix
        M_inv = sparse.diags(M_inv_diag)

        # 7. Wrap and return (depends on your code structure)
        return MINRESSparse(M_inv)


    # an old algorithm that directly passes in preconditioner matrix as parameter, now uses pcmr_normal that passes in
    # preconditioner multiplying function
    def pc_minres(self, b, x0, M_inv, max_iter=100, tol=1e-10, verbose=False):
        """
        Solves the linear system Ax = b using the Preconditioned MINRES algorithm.

        Args:
            b (np.array): The right-hand side vector.
            x0 (np.array): The initial guess for the solution.
            M_inv (np.array): A preconditioning matrix, in deepminres example is a diagonal matrix
            tol (float): The tolerance for convergence.
            max_iter (int): The maximum number of iterations.
            verbose (bool): If True, print the progress.

        Returns:
            tuple: The solution vector x and the array of residuals.
        """
        n = len(b)
        msg = [ ' beta1 = 0.  The exact solution is x0              ',   # 0
                ' A solution to Ax = b was found, given rtol        ',   # 1
                ' A least-squares solution was found, given rtol    ',   # 2
                ' Reasonable accuracy achieved, given eps           ',   # 3
                ' The iteration limit was reached                   ']   # 4

        istop = 0
        itn = 0
        Anorm = 0
        Acond = 0
        rnorm = 0
        ynorm = 0
        x = x0.copy()
        if x0 is None:
            r1 = b.copy()
        else:
            r1 = b - self.multiply_A(x0)
        y = M_inv.multiply(r1)
        beta1 = self.dot(r1, y)
        if beta1 == 0:
            return x0, beta1
        beta1 = np.sqrt(beta1)
        oldb = 0
        beta = beta1
        dbar = 0
        epsln = 0
        qrnorm = beta1
        phibar = beta1
        rhs1 = beta1
        rhs2 = 0
        tnorm2 = 0
        gmax = 0
        gmin = np.finfo(float).max
        cs = -1
        sn = 0
        w = np.zeros(n)
        w2 = np.zeros(n)
        r2 = r1
        res_arr = [beta1]

        for itn in range(max_iter):
            s = 1.0/beta
            v = s*y
            y = self.multiply_A(v)
            if itn >= 2:
                y = y - (beta / oldb) * r1
            alpha = self.dot(v, y)
            y = y - (alpha / beta) * r1
            r1 = r2
            r2 = y
            y = M_inv.multiply_A(r2)
            oldb = beta
            beta = self.dot(r2, y)
            if beta < 0:
                raise ValueError('non-symmetric matrix')
            beta = np.sqrt(beta)
            tnorm2 += alpha ** 2 + oldb ** 2 + beta ** 2
            oldeps = epsln
            delta = cs * dbar + sn * alpha
            gbar = sn * dbar - cs * alpha
            epsln = sn * beta
            dbar = - cs * beta
            root = np.sqrt(gbar ** 2 + dbar ** 2)
            Arnorm = phibar * root
            gamma = np.sqrt(gbar ** 2 + beta ** 2)
            gamma = max(gamma, np.finfo(float).eps)
            cs = gbar / gamma
            sn = beta / gamma
            phi = cs * phibar
            phibar = sn * phibar
            denom = 1.0 / gamma
            w1 = w2
            w2 = w
            w = (v - oldeps * w1 - delta * w2) * denom
            x = x + phi * w
            gmax = max(gmax, gamma)
            gmin = min(gmin, gamma)
            z = rhs1 / gamma
            rhs1 = rhs2 - delta * z
            rhs2 = - epsln * z
            Anorm = np.sqrt(tnorm2)
            ynorm = self.norm(x)
            epsa = Anorm * np.finfo(float).eps
            epsx = Anorm * ynorm * np.finfo(float).eps
            epsr = Anorm * ynorm * tol
            diag = gbar
            if diag == 0:
                diag = epsa
            qrnorm = phibar
            rnorm = qrnorm
            if ynorm == 0 or Anorm == 0:
                test1 = inf
            else:
                test1 = rnorm / (Anorm*ynorm)
            if Anorm == 0:
                test2 = inf
            else:
                test2 = root / Anorm
            Acond = gmax / gmin

            if istop == 0:
                t1 = 1 + test1      # These tests work if rtol < eps
                t2 = 1 + test2
                if t2 <= 1:
                    istop = 2
                if t1 <= 1:
                    istop = 1
                if itn >= max_iter:
                    istop = 4
                if epsx >= beta1:
                    istop = 3
                # if rnorm <= epsx   : istop = 2
                # if rnorm <= epsr   : istop = 1
                if test2 <= tol:
                    istop = 2
                if test1 <= tol:
                    istop = 1
            
            if verbose:
                print(f"Iteration {itn+1}, test1: {test1}")
                res_arr.append(rnorm)

            if istop != 0:
                break

        print("MINRES stopped" + f' istop   =  {istop:3g}               itn   ={itn+1:5g}')
        print("MINRES stopped" + f' Anorm   =  {Anorm:12.4e}      Acond =  {Acond:12.4e}')
        print("MINRES stopped" + f' rnorm   =  {rnorm:12.4e}      ynorm =  {ynorm:12.4e}')
        print("MINRES stopped" + f' Arnorm  =  {Arnorm:12.4e}')
        print("MINRES stopped" + msg[istop])

        return x, res_arr


    def pmr_normal(self, b, x0, precond, max_iter=100, tol=1e-10, verbose=False):
        """
        Solves the linear system Ax = b using the Preconditioned MINRES algorithm.

        Args:
            b (np.array): The right-hand side vector.
            x0 (np.array): The initial guess for the solution.
            precond (function): A function for multiplying the preconditioner
            tol (float): The tolerance for convergence.
            max_iter (int): The maximum number of iterations.
            verbose (bool): If True, print the progress.

        Returns:
            tuple: The solution vector x and the array of residuals.
        """
        n = len(b)
        msg = [ ' beta1 = 0.  The exact solution is x0              ',   # 0
                ' A solution to Ax = b was found, given rtol        ',   # 1
                ' A least-squares solution was found, given rtol    ',   # 2
                ' Reasonable accuracy achieved, given eps           ',   # 3
                ' The iteration limit was reached                   ']   # 4

        istop = 0
        itn = 0
        Anorm = 0
        Acond = 0
        rnorm = 0
        ynorm = 0
        x = x0.copy()
        if x0 is None:
            r1 = b.copy()
        else:
            r1 = b - self.multiply_A(x0)
        y = precond(r1)
        beta1 = self.dot(r1, y)
        if beta1 < 0:
            raise ValueError('indefinite preconditioner')
        elif beta1 == 0:
            return x0, beta1
        beta1 = np.sqrt(beta1)
        oldb = 0
        beta = beta1
        dbar = 0
        epsln = 0
        qrnorm = beta1
        phibar = beta1
        rhs1 = beta1
        rhs2 = 0
        tnorm2 = 0
        gmax = 0
        gmin = np.finfo(float).max
        cs = -1
        sn = 0
        w = np.zeros(n)
        w2 = np.zeros(n)
        r2 = r1
        res_arr = [beta1]

        for itn in range(max_iter):
            s = 1.0/beta
            v = s*y
            y = self.multiply_A(v)
            if itn >= 2:
                y = y - (beta / oldb) * r1
            alpha = self.dot(v, y)
            y = y - (alpha / beta) * r1
            r1 = r2
            r2 = y
            y = precond(r2)
            oldb = beta
            beta = self.dot(r2, y)
            if beta < 0:
                raise ValueError('non-symmetric matrix')
            beta = np.sqrt(beta)
            tnorm2 += alpha ** 2 + oldb ** 2 + beta ** 2
            oldeps = epsln
            delta = cs * dbar + sn * alpha
            gbar = sn * dbar - cs * alpha
            epsln = sn * beta
            dbar = - cs * beta
            root = np.sqrt(gbar ** 2 + dbar ** 2)
            Arnorm = phibar * root
            gamma = np.sqrt(gbar ** 2 + beta ** 2)
            gamma = max(gamma, np.finfo(float).eps)
            cs = gbar / gamma
            sn = beta / gamma
            phi = cs * phibar
            phibar = sn * phibar
            denom = 1.0 / gamma
            w1 = w2
            w2 = w
            w = (v - oldeps * w1 - delta * w2) * denom
            x = x + phi * w
            gmax = max(gmax, gamma)
            gmin = min(gmin, gamma)
            z = rhs1 / gamma
            rhs1 = rhs2 - delta * z
            rhs2 = - epsln * z
            Anorm = np.sqrt(tnorm2)
            ynorm = self.norm(x)
            epsa = Anorm * np.finfo(float).eps
            epsx = Anorm * ynorm * np.finfo(float).eps
            epsr = Anorm * ynorm * tol
            diag = gbar
            if diag == 0:
                diag = epsa
            qrnorm = phibar
            rnorm = qrnorm
            if ynorm == 0 or Anorm == 0:
                test1 = inf
            else:
                test1 = rnorm / (Anorm*ynorm)
            if Anorm == 0:
                test2 = inf
            else:
                test2 = root / Anorm
            Acond = gmax / gmin

            if istop == 0:
                t1 = 1 + test1  
                t2 = 1 + test2
                if t2 <= 1:
                    istop = 2
                if t1 <= 1:
                    istop = 1
                if itn >= max_iter:
                    istop = 4
                if epsx >= beta1:
                    istop = 3
                # if rnorm <= epsx   : istop = 2
                # if rnorm <= epsr   : istop = 1
                if test2 <= tol:
                    istop = 2
                if test1 <= tol:
                    istop = 1
            
            if verbose:
                print(f"Iteration {itn+1}, test1: {test1}")
                res_arr.append(rnorm)

            if istop != 0:
                break

        print("Preconditioned MINRES stopped" + f' istop   =  {istop:3g}               itn   ={itn+1:5g}')
        print("Preconditioned MINRES stopped" + f' Anorm   =  {Anorm:12.4e}      Acond =  {Acond:12.4e}')
        print("Preconditioned MINRES stopped" + f' rnorm   =  {rnorm:12.4e}      ynorm =  {ynorm:12.4e}')
        print("Preconditioned MINRES stopped" + f' Arnorm  =  {Arnorm:12.4e}')
        print("Preconditioned MINRES stopped" + msg[istop])

        return x, res_arr

    def create_ritz_vectors(self, b, num_vectors, sorting=True):
        """
        Creates Ritz vectors using the Lanczos iteration.

        Args:
            b (np.array): The input vector.
            num_vectors (int): The number of Ritz vectors to create.
            sorting (bool): If True, sort the Ritz vectors by eigenvalues.

        Returns:
            np.array: The Ritz vectors.
        """
        W, diagonal, sub_diagonal = self.lanczos_iteration(b, num_vectors, 1.0e-12)
        # W, diagonal, sub_diagonal = self.lanczos_iteration(b, num_vectors, 1.0e-12)
        if (num_vectors != len(diagonal)):
            print("Careful. Lanczos Iteration converged too early, num_vectors = " + str(num_vectors) + " > " + str(
                len(diagonal)))
            num_vectors = len(diagonal)
        tri_diag = np.zeros([num_vectors, num_vectors])
        for i in range(1, num_vectors - 1):
            tri_diag[i, i] = diagonal[i]
            tri_diag[i, i + 1] = sub_diagonal[i]
            tri_diag[i, i - 1] = sub_diagonal[i - 1]
        tri_diag[0, 0] = diagonal[0]
        tri_diag[0, 1] = sub_diagonal[0]
        tri_diag[num_vectors - 1, num_vectors - 1] = diagonal[num_vectors - 1]
        tri_diag[num_vectors - 1, num_vectors - 2] = sub_diagonal[num_vectors - 2]
        eigvals, Q0 = np.linalg.eigh(tri_diag)
        eigvals = np.real(eigvals)
        Q0 = np.real(Q0)
        Q1 = np.matmul(W.transpose(), Q0).transpose()
        if sorting:
            Q = np.zeros([num_vectors, self.n])
            sorted_eig_vals = sorted(range(num_vectors), key=lambda k: -eigvals[k])
            for i in range(num_vectors):
                Q[i] = Q1[sorted_eig_vals[i]].copy()
            return Q
        else:
            return Q1

    # precond is num_vectors x self.n np array
    def create_ritz_values(self, Q, relative=True):
        """
        Creates Ritz values using the Ritz vectors.

        Args:
            Q (np.array): The Ritz vectors.
            relative (bool): If True, compute relative Ritz values.

        Returns:
            np.array: The Ritz values.
        """
        lambda_ = np.zeros(Q.shape[0])
        for i in range(Q.shape[0]):
            dotx = np.dot(Q[i], Q[i])
            if dotx < self.machine_tol:
                print("Error! Zero vector in matrix Q.")
                return
            lambda_[i] = np.dot(Q[i], self.multiply_A_sparse(Q[i]))
            if relative:
                lambda_[i] = lambda_[i] / dotx
        return lambda_