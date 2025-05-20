#ifndef MINRESSPARSE_HPP
#define MINRESSPARSE_HPP

#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cassert>
#include <memory>
#include <vector>

#include "lstsquares_example/LstsquaresVector.hpp"
#include "lstsquares_example/SOperator.hpp"

//------------------------------------------------------------------------------
// Standard MINRESSparse function
//------------------------------------------------------------------------------
template<typename Operator, typename Vector>
int MINRESSparse(const Operator &A, Vector &x, const Vector &b,
                 int max_iter, double tol, bool verbose)
{
    double eps = std::numeric_limits<double>::epsilon();

    // Termination messages.
    std::vector<std::string> msg(5);
    msg[0] = " beta1 = 0.  The exact solution is x0";
    msg[1] = " A solution to Ax = b was found, given rtol";
    msg[2] = " A least-squares solution was found, given rtol";
    msg[3] = " Reasonable accuracy achieved, given eps";
    msg[4] = " The iteration limit was reached";

    int istop = 0;
    int itn = 0;
    double Anorm = 0.0, Acond = 0.0, Arnorm = 0.0;
    double rnorm = 0.0, ynorm = 0.0;

    // Step 1: Compute initial residual r1 = b - A*x, and set y = r1.
    std::unique_ptr<Vector> r1(x.Clone());
    A.Apply(x, *r1);          // r1 = A*x
    subtract(b, *r1, *r1);    // r1 = b - A*x
    std::unique_ptr<Vector> y(x.Clone());
    *y = *r1;                 // y = r1

    // Compute beta1 = inner(r1,y)
    double beta1 = InnerProduct(*r1, *y);
    if(beta1 == 0.0)
    {
        if(verbose)
            std::cout << msg[0] << std::endl;
        return 0; // x is already the exact solution.
    }
    beta1 = std::sqrt(beta1);

    // Initialize iteration variables.
    double oldb = 0.0, beta = beta1, dbar = 0.0, epsln = 0.0, oldeps = 0.0;
    double qrnorm = beta1, phibar = beta1, rhs1 = beta1, rhs2 = 0.0, tnorm2 = 0.0;
    double gmax = 0.0, gmin = std::numeric_limits<double>::max();
    double cs = -1.0, sn = 0.0;
    double alpha = 0.0, gamma = 0.0, delta = 0.0, gbar = 0.0, z = 0.0;

    // Allocate helper vectors.
    std::unique_ptr<Vector> w(x.Clone());
    std::unique_ptr<Vector> w1(x.Clone());
    std::unique_ptr<Vector> w2(x.Clone());
    std::unique_ptr<Vector> r2(x.Clone());
    *r2 = *r1;
    std::unique_ptr<Vector> v(x.Clone());

    // Initialize helper vectors to zero.
    *w  = 0.0;
    *w1 = 0.0;
    *w2 = 0.0;

    if(verbose)
    {
        std::cout << std::setw(6) << "It"
                  << std::setw(14) << "test1"
                  << std::setw(14) << "test2"
                  << std::setw(14) << "Anorm"
                  << std::setw(14) << "Acond"
                  << std::setw(14) << "gbar/|A|" << std::endl;
    }

    // Main iteration loop.
    for(itn = 1; itn <= max_iter; ++itn)
    {
        double s = 1.0 / beta;
        *v = *y;
        v->Scale(s);
        A.Apply(*v, *y);
        if(itn >= 2)
            add(*y, -beta/oldb, *r1, *y);
        alpha = InnerProduct(*v, *y);
        add(*y, -alpha/beta, *r2, *y);
        *r1 = *r2;
        *r2 = *y;
        *y = *r2;
        oldb = beta;
        beta = InnerProduct(*r2, *y);
        if(beta < 0)
        {
            std::cerr << "Error: non-symmetric matrix encountered." << std::endl;
            istop = -1;
            break;
        }
        beta = std::sqrt(beta);
        tnorm2 += alpha*alpha + oldb*oldb + beta*beta;
        oldeps = epsln;
        delta = cs * dbar + sn * alpha;
        gbar = sn * dbar - cs * alpha;
        epsln = sn * beta;
        dbar = -cs * beta;
        double root = std::sqrt(gbar*gbar + dbar*dbar);
        Arnorm = phibar * root;
        gamma = std::sqrt(gbar*gbar + beta*beta);
        gamma = std::max(gamma, eps);
        cs = gbar / gamma;
        sn = beta / gamma;
        double phi = cs * phibar;
        phibar = sn * phibar;
        double denom = 1.0 / gamma;
        *w1 = *w2;
        *w2 = *w;
        add(-oldeps, *w1, -delta, *w2, *w);
        add(denom, *v, *w, *w);
        add(x, phi, *w, x);
        gmax = std::max(gmax, gamma);
        gmin = std::min(gmin, gamma);
        z = rhs1 / gamma;
        rhs1 = rhs2 - delta * z;
        rhs2 = - epsln * z;
        Anorm = std::sqrt(tnorm2);
        double ynorm2 = InnerProduct(x, x);
        ynorm = std::sqrt(ynorm2);
        double epsa = Anorm * eps;
        double epsx = Anorm * eps;
        double epsr = Anorm * ynorm * tol;
        double diag = gbar;
        if(diag == 0)
            diag = epsa;
        qrnorm = phibar;
        rnorm = qrnorm;
        double test1, test2;
        if(ynorm == 0 || Anorm == 0)
            test1 = std::numeric_limits<double>::infinity();
        else
            test1 = rnorm / (Anorm * ynorm);
        if(Anorm == 0)
            test2 = std::numeric_limits<double>::infinity();
        else
            test2 = root / Anorm;
        Acond = gmax / gmin;

        double t1 = 1.0 + test1;
        double t2 = 1.0 + test2;
        if(t2 <= 1.0)
            istop = 2;
        if(t1 <= 1.0)
            istop = 1;
        if(itn >= max_iter)
            istop = 4;
        if(epsx >= beta1)
            istop = 3;
        if(test2 <= tol)
            istop = 2;
        if(test1 <= tol)
            istop = 1;
        if(verbose)
            std::cout << "Iteration " << itn << ", test1: " << test1 << std::endl;
        if(istop != 0)
            break;
    }

    std::cout << "MINRESSparse stopped with istop = " << istop << " after " << itn << " iterations.\n";
    std::cout << "Anorm = " << Anorm << ", Acond = " << Acond << "\n";
    std::cout << "rnorm = " << rnorm << ", ynorm = " << ynorm << "\n";
    std::cout << "Arnorm = " << Arnorm << "\n";
    std::cout << msg[istop] << std::endl;

    return istop;
}

//------------------------------------------------------------------------------
// DeepMINRESSparse function
//------------------------------------------------------------------------------
// This version uses a pretrained model to predict the search direction.
// The model_predict callable must have the following signature:
//     void model_predict(const Vector & input, Vector & output)
// where the input (typically the current residual or other relevant quantity)
// is reshaped and fed into the ONNX model, and the output is the predicted search direction.
template<typename Operator, typename Vector, typename ModelPredict>
int DeepMINRESSparse(const Operator &A, Vector &x, const Vector &b,
                      ModelPredict model_predict,
                      int max_iter, double tol, bool verbose)
{
    double eps = std::numeric_limits<double>::epsilon();

    std::vector<std::string> msg(5);
    msg[0] = " beta1 = 0.  The exact solution is x0";
    msg[1] = " A solution to Ax = b was found, given rtol";
    msg[2] = " A least-squares solution was found, given rtol";
    msg[3] = " Reasonable accuracy achieved, given eps";
    msg[4] = " The iteration limit was reached";

    int istop = 0;
    int itn = 0;
    double Anorm = 0.0, Acond = 0.0, Arnorm = 0.0;
    double rnorm = 0.0, ynorm = 0.0;

    // Step 1: Compute the initial residual r1 = b - A*x, and set y = r1.
    std::unique_ptr<Vector> r1(x.Clone());
    A.Apply(x, *r1);
    subtract(b, *r1, *r1);
    std::unique_ptr<Vector> y(x.Clone());
    *y = *r1;

    double beta1 = InnerProduct(*r1, *y);
    if(beta1 == 0.0)
    {
        if(verbose)
            std::cout << msg[0] << std::endl;
        return 0;
    }
    beta1 = std::sqrt(beta1);

    double oldb = 0.0, beta = beta1, dbar = 0.0, epsln = 0.0, oldeps = 0.0;
    double qrnorm = beta1, phibar = beta1, rhs1 = beta1, rhs2 = 0.0, tnorm2 = 0.0;
    double gmax = 0.0, gmin = std::numeric_limits<double>::max();
    double cs = -1.0, sn = 0.0;
    double alpha = 0.0, gamma = 0.0, delta = 0.0, gbar = 0.0, z = 0.0;

    std::unique_ptr<Vector> w(x.Clone());
    std::unique_ptr<Vector> w1(x.Clone());
    std::unique_ptr<Vector> w2(x.Clone());
    std::unique_ptr<Vector> r2(x.Clone());
    *r2 = *r1;
    std::unique_ptr<Vector> v(x.Clone());

    *w  = 0.0;
    *w1 = 0.0;
    *w2 = 0.0;

    if(verbose)
    {
        std::cout << std::setw(6) << "It"
                  << std::setw(14) << "test1"
                  << std::setw(14) << "test2"
                  << std::setw(14) << "Anorm"
                  << std::setw(14) << "Acond"
                  << std::setw(14) << "gbar/|A|" << std::endl;
    }

    // Main iteration loop for deep MINRES.
    for(itn = 1; itn <= max_iter; ++itn)
    {
        // Instead of scaling y by 1/beta to get v,
        // use the pretrained model to predict the search direction.
        // model_predict is called as: model_predict(current_y, predicted_v)
        model_predict(*y, *v);

        // Now compute y = A*v.
        A.Apply(*v, *y);
        if(itn >= 2)
            add(*y, -beta/oldb, *r1, *y);
        alpha = InnerProduct(*v, *y);
        add(*y, -alpha/beta, *r2, *y);
        *r1 = *r2;
        *r2 = *y;
        *y = *r2;
        oldb = beta;
        beta = InnerProduct(*r2, *y);
        if(beta < 0)
        {
            std::cerr << "Error: non-symmetric matrix encountered." << std::endl;
            istop = -1;
            break;
        }
        beta = std::sqrt(beta);
        tnorm2 += alpha*alpha + oldb*oldb + beta*beta;
        oldeps = epsln;
        delta = cs * dbar + sn * alpha;
        gbar = sn * dbar - cs * alpha;
        epsln = sn * beta;
        dbar = -cs * beta;
        double root = std::sqrt(gbar*gbar + dbar*dbar);
        Arnorm = phibar * root;
        gamma = std::sqrt(gbar*gbar + beta*beta);
        gamma = std::max(gamma, eps);
        cs = gbar / gamma;
        sn = beta / gamma;
        double phi = cs * phibar;
        phibar = sn * phibar;
        double denom = 1.0 / gamma;
        *w1 = *w2;
        *w2 = *w;
        add(-oldeps, *w1, -delta, *w2, *w);
        add(denom, *v, *w, *w);
        add(x, phi, *w, x);
        gmax = std::max(gmax, gamma);
        gmin = std::min(gmin, gamma);
        z = rhs1 / gamma;
        rhs1 = rhs2 - delta * z;
        rhs2 = - epsln * z;
        Anorm = std::sqrt(tnorm2);
        double ynorm2 = InnerProduct(x, x);
        ynorm = std::sqrt(ynorm2);
        double epsa = Anorm * eps;
        double epsx = Anorm * eps;
        double epsr = Anorm * ynorm * tol;
        double diag = gbar;
        if(diag == 0)
            diag = epsa;
        qrnorm = phibar;
        rnorm = qrnorm;
        double test1, test2;
        if(ynorm == 0 || Anorm == 0)
            test1 = std::numeric_limits<double>::infinity();
        else
            test1 = rnorm / (Anorm * ynorm);
        if(Anorm == 0)
            test2 = std::numeric_limits<double>::infinity();
        else
            test2 = root / Anorm;
        Acond = gmax / gmin;

        double t1 = 1.0 + test1;
        double t2 = 1.0 + test2;
        if(t2 <= 1.0)
            istop = 2;
        if(t1 <= 1.0)
            istop = 1;
        if(itn >= max_iter)
            istop = 4;
        if(epsx >= beta1)
            istop = 3;
        if(test2 <= tol)
            istop = 2;
        if(test1 <= tol)
            istop = 1;
        if(verbose)
            std::cout << "Iteration " << itn << ", test1: " << test1 << std::endl;
        if(istop != 0)
            break;
    }

    std::cout << "DeepMINRESSparse stopped with istop = " << istop << " after " << itn << " iterations.\n";
    std::cout << "Anorm = " << Anorm << ", Acond = " << Acond << "\n";
    std::cout << "rnorm = " << rnorm << ", ynorm = " << ynorm << "\n";
    std::cout << "Arnorm = " << Arnorm << "\n";
    std::cout << msg[istop] << std::endl;

    return istop;
}

#endif // MINRESSPARSE_HPP
