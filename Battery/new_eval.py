import jax.numpy as jnp
import numpy as np
import scipy
import FoKL

from FoKL.getKernels import sp500

import pybamm as pybamm

def evaluate_pybamm(betas, mtx, inputs):
    """

    """
    phis = sp500()
    mmtx, nn = jnp.shape(mtx)
    mmtx += 1
    mbets = 1
    n = jnp.shape(inputs)[0]  # Size of normalized inputs
    mputs = 1
    X_sol = []

    phis = jnp.array(phis)

    mtx = jnp.array(mtx)
    phind = []
    X_st = []
    A = [1, 2, 3]
    for i in range(n):
        phind_temp = inputs[i]*499
        sett = (phind_temp == 0)
        phind_temp = phind_temp + pybamm.Scalar(sett)
        r = 1 / 499  # interval of when basis function changes (i.e., when next cubic function defines spline)
        xmin = (phind_temp - 1) * r
        X = (inputs[i] - xmin) / r
        phind.append(phind_temp - 1)
        X_sc = [X**a for a in A]
        X_st.append(X_sc)

    lspace = []
    for i in range(mputs):
        lspace.append(np.linspace(0,499,499))
    lspace = np.array(lspace)

    phind[0].children[0].size = 1

    for k in range(mmtx):
        phi = 1.
        for j in range(n):

            num = mtx[k][j]
            if num > 0:
                nid = int(num - 1)

                coeff = []

                for jj in range(4):
                    phispace = phis[nid][jj].reshape(1,-1)
                    phi_interp = pybamm.Interpolant(lspace[0], phispace[0], phind[j])
                    coeff.append(phi_interp)

            # multiplies phi(x0)*phi(x1)*etc.
                phi *= coeff[0] + coeff[1] * X_st[j][0] + coeff[2] * X_st[j][1] + coeff[3] * X_st[j][2]
        X_sol.append(phi)

    beta_matrix = np.eye(16)
    X_sol_ones = pybamm.Inner(betas,beta_matrix[0,:])
    mean = X_sol_ones
    for i in range(1,len(X_sol) - 1):

        here = pybamm.Inner(betas, beta_matrix[i,:])
        there = pybamm.Multiplication(X_sol[i], here)
        mean = pybamm.Addition(mean, there)

    return mean


def evaluate_pybamm_new(betas, mtx, inputs):
    """

    """
    phis = sp500()
    m, mmtx = jnp.shape(betas)
    mbets = 1
    n = jnp.shape(inputs)[0]  # Size of normalized inputs
    mputs = 1
    X_sol = []

    phis = jnp.array(phis)

    mtx = jnp.array(mtx)
    phind = []
    X_st = []
    A = [1, 2, 3]
    for i in range(n):
        phind_temp = inputs[i] * 499
        sett = (phind_temp == 0)
        phind_temp = phind_temp + pybamm.Scalar(sett)
        r = 1 / 499  # interval of when basis function changes (i.e., when next cubic function defines spline)
        xmin = (phind_temp - 1) * r
        X = (inputs[i] - xmin) / r
        phind.append(phind_temp - 1)
        X_sc = [X ** a for a in A]
        X_st.append(X_sc)

    lspace = []
    for i in range(mputs):
        lspace.append(np.linspace(0, 499, 499))
    lspace = np.array(lspace)

    phind[0].children[0].size = 1

    # m, mbets = np.shape(betas)  # Size of betas
    # n = np.shape(normputs)[0]  # Size of normalized inputs
    # mputs = int(np.size(normputs) / n)

    # for i in range(n): # number of inputs
    #     for j in range(1, mbets): # number of draws
    #         phi = 1
    #         for k in range(mputs): # number of instances of inputs
    #             num = mtx[j - 1, k]
    #             if num > 0:
    #                 nid = int(num - 1)
    #                 if self.kernel == self.kernels[0]:  # == 'Cubic Splines':
    #                     coeffs = [phis[nid][order][phind[i, k]] for order in range(4)]  # coefficients for cubic
    #                 elif self.kernel == self.kernels[1]:  # == 'Bernoulli Polynomials':
    #                     coeffs = phis[nid]  # coefficients for bernoulli
    #                 phi *= self.evaluate_basis(coeffs, xsm[i, k])  # multiplies phi(x0)*phi(x1)*etc.
    #
    #         X[i, j] = phi
    #
    # X[:, 0] = np.ones((n,))

    for k in range(mmtx):
        phi = 1.

        for j in range(n):

            num = mtx[k][j]
            if num > 0:
                nid = int(num - 1)

                coeff = []
                phispace_int = np.array(phis[nid][0].reshape(1, -1))
                lspace_int = np.array(lspace[0].reshape(1, -1))
                for jj in range(1, 4):
                    phispace = np.array(phis[nid][jj].reshape(1, -1))
                    lspace = np.array(lspace[0].reshape(1,-1))
                    phispace_int = np.vstack([phispace,phispace_int])
                    lspace_int = np.vstack([lspace,lspace_int])

                    # coeff.append(phi_interp)
                phi_interp = pybamm.Interpolant(lspace_int, phispace_int, phind[j])
                # multiplies phi(x0)*phi(x1)*etc.
                phi *= coeff[0] + coeff[1] * X_st[j][0] + coeff[2] * X_st[j][1] + coeff[3] * X_st[j][2]
        X_sol.append(phi)
    beta_matrix = np.eye(betas.shape[0])
    X_sol_ones = pybamm.MatrixMultiplication(betas,np.transpose(beta_matrix[0,:]))
    mean = X_sol_ones
    for i in range(len(X_sol) - 1):
        X_sol_betas = X_sol[i] * pybamm.MatrixMultiplication(betas, np.transpose(beta_matrix[i,:]))
        mean += X_sol_betas

    return mean


def evaluate_pybamm_clone(betas, mtx, inputs):
    """

    """
    phis = sp500()
    mmtx, nn = jnp.shape(mtx)
    mmtx += 1
    mbets = 1
    n = jnp.shape(inputs)[0]  # Size of normalized inputs
    mputs = 1
    X_sol = []

    phis = jnp.array(phis)

    mtx = jnp.array(mtx)
    phind = []
    X_st = []
    A = [1, 2, 3]
    for i in range(n):
        phind_temp = inputs[i]*499
        sett = (phind_temp == 0)
        phind_temp = phind_temp + sett
        r = 1 / 499  # interval of when basis function changes (i.e., when next cubic function defines spline)
        xmin = (phind_temp - 1) * r
        X = (inputs[i] - xmin) / r
        phind.append(phind_temp - 1)
        X_sc = [X**a for a in A]
        X_st.append(X_sc)

    lspace = []
    for i in range(mputs):
        lspace.append(np.linspace(0,499,499))
    lspace = np.array(lspace)
    #
    # phind[0].children[0].size = 1

    # m, mbets = np.shape(betas)  # Size of betas
    # n = np.shape(normputs)[0]  # Size of normalized inputs
    # mputs = int(np.size(normputs) / n)

    # for i in range(n): # number of inputs
    #     for j in range(1, mbets): # number of draws
    #         phi = 1
    #         for k in range(mputs): # number of instances of inputs
    #             num = mtx[j - 1, k]
    #             if num > 0:
    #                 nid = int(num - 1)
    #                 if self.kernel == self.kernels[0]:  # == 'Cubic Splines':
    #                     coeffs = [phis[nid][order][phind[i, k]] for order in range(4)]  # coefficients for cubic
    #                 elif self.kernel == self.kernels[1]:  # == 'Bernoulli Polynomials':
    #                     coeffs = phis[nid]  # coefficients for bernoulli
    #                 phi *= self.evaluate_basis(coeffs, xsm[i, k])  # multiplies phi(x0)*phi(x1)*etc.
    #
    #         X[i, j] = phi
    #
    # X[:, 0] = np.ones((n,))

    for k in range(mmtx):
        phi = 1.
        for j in range(n):

            num = mtx[k][j]
            if num > 0:
                nid = int(num - 1)

                coeff = []

                for jj in range(4):
                    phispace = phis[nid][jj].reshape(1,-1)
                    phi_interp = scipy.interpolate.interp1d(lspace[0], phispace[0])(phind[j])
                    coeff.append(phi_interp)

            # multiplies phi(x0)*phi(x1)*etc.
                phi *= coeff[0] + coeff[1] * X_st[j][0] + coeff[2] * X_st[j][1] + coeff[3] * X_st[j][2]
        X_sol.append(phi)

    beta_matrix = np.eye(16)
    X_sol_ones = 1 * np.matmul(betas,np.transpose(beta_matrix[0, :]))[0]
    mean = X_sol_ones
    for i in range(len(X_sol) - 1):
        X_sol_betas = X_sol[i]*np.matmul(betas,beta_matrix[i,:])[0]
        mean += X_sol_betas

    return mean