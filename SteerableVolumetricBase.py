import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt


# Calculation of GPSWF's #
# Calculates the radial eigenfunctions and eigenvalues.
def matrix_differential_operator(c, N, trunc):
    """
    Creates the truncated matrix form of the differential operator.
    :param c: the band-limit parameter.
    :param N: the prolate order N.
    :param trunc: where to truncate the matrix.
    :return: first, an array of shape [trunc + 1,] of the main diagonal elements.
             second, an array of shape [trunc,] of the off diagonal elements.
    """
    off_d = np.array(range(1, trunc + 1), dtype=float)
    off_d = -(c ** 2 * off_d * (off_d + N + 0.5)) / ((2 * off_d + N + 0.5) * np.sqrt((2 * off_d + N + 0.5) ** 2 - 1))
    d = np.array(range(trunc + 1), dtype=float)
    d = -(c ** 2 * ((2 * d + N + 1.5) * (N + 0.5) + 2 * (d + 1) * d)/((2 * d + N + 0.5) * (2 * d + N + 2.5))
          + (2 * d + N + 1) * (2 * d + N + 2))
    return d, off_d


def x_times_dfunc(N, coeffs):
    """
    Calculation of the coefficients of xf'(x) in the normalized Zernike polynomials basis from the expansion of f(x).
    :param N: the N parameter of the Zernike polynomials.
    :param coeffs: an array with shape [k, m] of coefficients.
                   each row i corresponds to the coefficients of \hat{R}_{N,i}.
                   each column j corresponds to the expansion coefficients of a single function f_j.
    :return: an array with shape [k, m] of coefficients.
             each row i corresponds to the coefficients of \hat{R}_{N,i}.
             each column j corresponds to the expansion coefficients of a single function xf'_j(x).
    """
    res = np.zeros_like(coeffs)
    indices = np.array(range(coeffs.shape[0] - 1, 0, -1))
    indices = np.sqrt(2 * indices + N + 1.5).reshape(-1, 1)
    indices = np.cumsum(coeffs[-1:0:-1] * indices, axis=0)[::-1]
    res[coeffs.shape[0] - 1] = (2 * (coeffs.shape[0] - 1) + N) * coeffs[coeffs.shape[0] - 1]
    for i in range(coeffs.shape[0] - 2, -1, -1):
        res[i] = (2 * i + N) * coeffs[i] + 2 * np.sqrt(2 * i + N + 1.5) * indices[i]
    return res


def calculate_gamma_0(c, N, coeffs):
    """
    Computation of the eigenvalue \gamma_{N,0} of the integral operator with prolate order N.
    :param c: the band-limit parameter
    :param N: the prolate order N
    :param coeffs: an array of shape [k,].
                   the coefficients of the eigenfunction expanded in normalized Zernike basis.
    :return: the eigenvalue \gamma_{N,0}
    """
    size = np.size(coeffs)
    k = np.array(range(size))
    cfs1 = (-1) ** k * np.sqrt(4 * k + 2 * N + 3)
    cfs2 = sp.special.gamma(N + 1.5) * np.concatenate((np.array([1]), np.cumprod((N + 0.5 + k[1:]) / k[1:])))
    trunc1 = np.where(np.abs(cfs2) > np.finfo(np.float64).max * 1e-10)[0]
    trunc1 = np.min(trunc1) if trunc1.size > 0 else size - 1
    trunc2 = np.where(np.abs(coeffs) < np.finfo(np.float64).tiny * 1e10)[0]
    trunc2 = np.min(trunc2) if trunc2.size > 0 else size - 1
    trunc = np.min((trunc1, trunc2))
    return ((c / 2) ** (N + 1)  * coeffs[:trunc][0]
            / (np.sqrt(N + 1.5) * np.sum(coeffs[:trunc] * cfs1[:trunc] * cfs2[:trunc])))


def calculate_gamma_ratios(N, coeffs):
    """
    Computation of the ratios of eigenvalues of the integral operator with prolate order N.
    :param N: the prolate order N.
    :param coeffs: an array with shape [k, n] of coefficients.
                   each row i corresponds to the coefficients of \hat{R}_{N,i}.
                   each column j corresponds to the expansion coefficients of the eigenfunction Phi_{N,j}.
    :return: an array with shape [n,] of the ratios \gamma_{N,i} / \gamma_{N,0}.
    """
    xdfunc = x_times_dfunc(N, coeffs)
    numerator = np.sum(np.roll(xdfunc, 1, axis=1) * coeffs, axis=0)
    denominator = np.sum(np.roll(coeffs, 1, axis=1) * xdfunc, axis=0)
    ratios = numerator / denominator
    ratios[0] = 1
    return np.cumprod(ratios)


def radial_coefficients(c, N, min_ratio, matdim):
    """
    Computation of the eigenvalues and the coefficients of the eigenfunction of the integral operator with prolate
        parameter N.
    :param c: the band-limit parameter.
    :param N: the prolate order N.
    :param min_ratio: threshold for minimal eigenvalue kept.
    :param matdim: dimension of the matrix used in the calculation.
    :return: first, an array with shape [k, n] of coefficients.
                    each row i corresponds to the coefficients of \hat{R}_{N,i}.
                    each column j corresponds to the expansion coefficients of the eigenfunction \Phi_{N,j}.
             second, an array with shape [n,] of the eigenvalues for GPSWF \alpha_{N,i}.
    """
    if not c > 0:
        raise ValueError('The parameter c must be greater than 0')
    if not (N >= 0 and isinstance(N, int)):
        raise ValueError('The order N must be a non-negative integer')
    if not min_ratio < 1:
        raise ValueError('The minimal ratio between eigenvalues must be less than 1')
    if not min_ratio > 0:
        raise ValueError('The minimal ratio between eigenvalues must be greater than 0')
    if matdim <= 10:
        raise ValueError('The matrix for computing the coefficients cannot be too small')
    mat = matrix_differential_operator(c, N, matdim - 1)
    vecs = sp.linalg.eigh_tridiagonal(mat[0], mat[1])[1][:, ::-1]
    vecs = vecs * ((vecs[0, :] >= 0) * 2. - 1).reshape(1, -1) * (-1) ** np.array(range(matdim)).reshape(1, -1)
    gammas = calculate_gamma_0(c, N, vecs[:, 0]) * calculate_gamma_ratios(N, vecs)
    prolate_to_keep = np.where(np.abs(gammas) * np.sqrt(c) <= min_ratio)[0]
    prolate_to_keep = np.min(prolate_to_keep) if np.size(prolate_to_keep) > 0 else matdim
    if not prolate_to_keep:
        return np.array([]), np.array([])
    if prolate_to_keep + 20 >= matdim:
        raise ValueError('Small margin in matrix size - number of prolates')
    coeffs_to_keep = np.where(np.max(np.abs(vecs[:, :prolate_to_keep]), axis=1) >= np.finfo(float).eps/100)[0]
    coeffs_to_keep = np.max(coeffs_to_keep) if np.size(coeffs_to_keep) > 0 else matdim
    if coeffs_to_keep + 20 >= matdim:
        raise ValueError('Small margin in matrix size - number of coefficients')
    alphas = gammas[:prolate_to_keep] / c * 1j ** N * (2 * np.pi) ** 1.5
    return vecs[:coeffs_to_keep, :prolate_to_keep], alphas


def funcs_from_zernike_base(N, coeffs, rvals):
    """
    Evaluation of functions expanded in the normalized Zernike polynomials basis at given x values.
    :param N: the N parameter of the Zernike polynomials.
    :param coeffs: an array with shape [k, m] of coefficients.
                   each row i corresponds to the coefficients of \hat{R}_{N,i}.
                   each column j corresponds to the expansion coefficients of a single function f_j.
    :param rvals: an array with shape [x,] of x values.
    :return: an array with shape [x, m].
             each row corresponds to the evaluation at x_i.
             each column corresponds to the evaluation of a single function at all x values.
    """
    yvals = 1 - 2 * rvals**2
    k = np.array(range(coeffs.shape[0]), dtype=float).reshape(-1, 1)
    coeffs = (-1)**k * np.sqrt(4 * k + 2 * N + 3) * coeffs
    prev1 = np.ones_like(yvals)
    res = np.outer(prev1, coeffs[0])
    if coeffs.shape[0] == 1:
        return rvals.reshape(-1, 1) ** N * res
    curr = 0.5 * (N + 0.5 + (2.5 + N) * yvals)
    res += np.outer(curr, coeffs[1])
    if coeffs.shape[0] == 2:
        return rvals.reshape(-1, 1) ** N * res
    for i in range(2, coeffs.shape[0]):
        temp = ((2*i+N-0.5)*((N+0.5)**2+(2*i+N+0.5)*(2*i+N-1.5)*yvals)*curr-2*(i+N-0.5)*(i-1)*(2*i+N+0.5)*prev1)\
               / (2*i*(i+N+0.5)*(2*i+N-1.5))
        prev1 = curr
        curr = temp
        res += np.outer(curr, coeffs[i])
    return rvals.reshape(-1, 1) ** N * res


def radial_functions(N, coeffs, rvals, mode='eval', l=None):
    """
    Evaluates \Phi_{N,n} at given x values.
    :param N: the prolate order N.
    :param coeffs: an array with shape [k, n] of the coefficients of the eigenfunctions.
    :param rvals: an array with shape [x,] of x values. shape dependent on the parameter 'mode'.
    :param mode: 'eval' - calculation for the construction of the GPSWF's coefficients.
                          assumes data is arranged as in 'shortened_data_points' and returns it arranged as in
                           'construction_data_points'.
                 'const' - calculation for the evaluation in GPSWF basis.
                           assumes data is arranged as in 'evaluation_data_points'.
    :param l: the sampling frequency L. should only be given for the case of construction of coefficients.
    :return: an array with shape [num_points, m].
             each row corresponds to the evaluation at x_i.
             each column corresponds to the evaluation of the radial eigenfunction \Phi_{N,j} at all x values.
    """
    if np.size(coeffs.shape) == 1:
        coeffs = coeffs.reshape(-1, 1)
    temp = funcs_from_zernike_base(N, coeffs, rvals)
    if mode =='eval':
        return temp
    index = -2 * l - 1
    res = np.zeros((temp.shape[0] * 4 + 3 * index, coeffs.shape[1]))
    res[:index:4] = temp[:index]
    res[1:index:4] = temp[:index]
    res[2:index:4] = temp[:index]
    res[3:index:4] = temp[:index]
    res[index:] = temp[index:]
    return res


# Calculates the spherical harmonics.
def legendre_poly(max_N, rvals):
    """
    Evaluation of the normalized associated Legendre polynomials \hat{P}^m_N at given x values
        for all orders 0<=N<=max_N and 0<=m<=max_N.
    :param max_N: the maximal prolate order N.
    :param rvals: an array with shape [x,] of x values.
    :return: an array with shape [x, max_N + 1, max_m + 1] of the evaluations.
             vector res[:, N, m] corresponds to the evaluation of \hat{P}^m_N at the x values.
    """
    res = np.zeros((rvals.size, max_N + 1, max_N + 1))
    for m in range(max_N + 1):
        res[:, m, m] = ((-1) ** m * np.sqrt((2 * m + 1) / (4 * np.pi * sp.special.factorial(2 * m)))
                        * sp.special.factorial2(2 * m + 1) / (2 * m + 1) * (1 - rvals ** 2) ** (m / 2))
        if m == max_N:
            break
        res[:, m + 1, m] = rvals * np.sqrt(2 * m + 3) * res[:, m, m]
        for l in range(m + 2, max_N + 1):
            res[:, l, m] = (np.sqrt((4 * l ** 2 - 1) / (l ** 2 - m ** 2))
                            * (rvals * res[:, l - 1, m] - np.sqrt(((l - 1) ** 2 - m ** 2)
                            / (4 * (l - 1) ** 2 - 1)) * res[:, l - 2, m]))
    return res


def spherical_harmonics(max_N, thetas ,phis):
    """
    Evaluation of the spherical harmonics S_{N,m} at points (theta, phi).
    :param max_N: the maximal prolate order N.
    :param thetas: an array with shape [x,] of theta values.
    :param phis: an array with shape [x,] of phi values.
    :return: an array with shape [x, max_N + 1, max_N + 1] of the evaluations S_{N,m}(theta, phi).
             vector res[:, N, m] corresponds to the spherical harmonics S_{N,m}(thetas, phis)
    """
    return (legendre_poly(max_N, np.cos(thetas))
            * np.exp(1.j * np.outer(phis, np.array(range(max_N + 1))).reshape((phis.size, 1, max_N + 1))))


# Calculates the GPSWF's.
def prolate_fixed_order(N, l, radial, eigs, sph_harm, mode='const'):
    """
    Evaluation of the GPSWF's \psi_{N,m,n} of orders N, 0<=n<=radial.shape[1]-1, 0<=m<=N.
    :param N: the prolate order N.
    :param l: the sampling frequency L.
    :param radial: an array with shape [x, n] of the evaluations of \Phi_{N,i}.
                   the dimension 0 corresponds to the evaluations of \Phi_{N,i} at the x_k for all 0<=i<=n-1.
                   the dimension 1 corresponds to the evaluations of \Phi_{N,i} at all x values and constant i.
    :param eigs: an array with shape [n,] of the eigenvalues alpha_{N,n}.
    :param sph_harm: an array with shape [k, N+1] of evaluations of the S_{N,j}.
                     the dimension 0 corresponds to the evaluations of S_{N,j} at (theta_k,phi_k) for all 0<=j<=N.
                     the dimension 1 corresponds to the evaluations of S_{N,m} at all (theta,phi) values.
                     assumes that negative m orders are not included.
    :param mode: 'const' - calculation for the construction of the GPSWF's coefficients.
                           assumes data is arranged as in 'shortened_data_points'.
                 'eval' - calculation for the evaluation in GPSWF basis.
                          assumes data is arranged as in 'evaluation_data_points'.
    :return: first, an array with shape [x, n*(2*N+1)].
                    the dimension 0 corresponds to the evaluations of \psi_{N,i,j} in (r_k,theta_k,phi_k) for -N<=i<=N
                        and 0<=j<=n-1.
                    the dimension 1 corresponds to the evaluation of \psi{N,m,n} at all (rvals, thetas, phis)
                        the indices are ordered like [m,n]=([-N,0],...,[N,0],[-N,1],...,[N,1],...,[-N,n-1],...,[N,n-1]).
             second, an array with shape [n(2N+1),] of the corresponding eigenvalues \alpha_{N,n}.
                     each \alpha_{N,n} repeats 2N+1 times.
    """
    if mode == 'const':
        index = - 2 * l - 1
        angular = np.zeros((radial.shape[0], 2 * N + 1), dtype=complex)
        angular[:index:4, N] = sph_harm[:index, 0]
        angular[1:index:4, N] = 1j ** N * angular[:index:4, N]
        angular[2:index:4, N] = (-1) ** N * angular[:index:4, N]
        angular[3:index:4, N] = (-1j) ** N * angular[:index:4, N]
        angular[index:, N] = sph_harm[index:, 0]
        for der in range(1, N + 1):
            angular[:index:4, N + der] = sph_harm[:index, der]
            angular[1:index:4, N + der] = 1j ** der * angular[:index:4, N + der]
            angular[2:index:4, N + der] = (-1) ** der * angular[:index:4, N + der]
            angular[3:index:4, N + der] = (-1j) ** der * angular[:index:4, N + der]

            angular[:index:4, N - der] = (-1) ** der * np.conjugate(angular[:index:4, N + der])
            angular[1:index:4, N - der] = (-1) ** der * np.conjugate(angular[1:index:4, N + der])
            angular[2:index:4, N - der] = (-1) ** der * np.conjugate(angular[2:index:4, N + der])
            angular[3:index:4, N - der] = (-1) ** der * np.conjugate(angular[3:index:4, N + der])

            angular[index:, N + der] = sph_harm[index:, der]
            angular[index:, N - der] = (-1) ** der * np.conjugate(angular[index:, N + der])
    else:
        angular = np.zeros((radial.shape[0], 2 * N + 1), dtype=complex)
        angular[:, N] = sph_harm[:, 0]
        for der in range(N + 1):
            angular[:, N + der] = sph_harm[:, der]
            angular[:, N - der] = (-1) ** der * np.conjugate(angular[:, N + der])
    numprols = radial.shape[1] * (2 * N + 1)
    res = np.zeros((radial.shape[0], numprols), dtype=complex)
    alphavec = np.zeros(numprols, dtype=complex)
    for prol in range(radial.shape[1]):
        res[:, prol * (2 * N + 1): (prol + 1) * (2 * N + 1)] = angular * radial[:, prol].reshape(-1, 1)
        alphavec[prol * (2 * N + 1): (prol + 1) * (2 * N + 1)] = eigs[prol]
    return res, alphavec


# Calculation of the GPSWF basis coefficients #
def coefficients_fixed_order(c, N, l, radial, eigs, sph_harm, fvals):
    """
    Calculation of the GPSWF's basis coefficients \hat{a}_{N,m,n} of a sampled function for a fixed order N.
    :param c: the band-limit parameter.
    :param N: the prolate order N.
    :param l: the sampling frequency L.
    :param radial: an array with shape [x, n] of the evaluations of \Phi_{N,i} for 0<=i<=n-1.
    :param eigs: an array with shape [n,] of the eigenvalues \alpha_{N,i} for 0<=i<=n-1.
    :param sph_harm: an array with shape [x, N+1] of the evaluations of S_{N,j} for 0<=j<=N.
    :param fvals: an array with shape [x,] of the sampled function.
    :return: an array with shape [1, n(2N+1)] of the GPSWF's basis coefficients.
    """
    prolates, eigenvals = prolate_fixed_order(N, l, radial, eigs, sph_harm, 'const')
    coeffs = np.sum((fvals * (c / (2 * np.pi * l))**3).reshape(-1, 1)
                    * np.conjugate(prolates * eigenvals.reshape(1, -1)), axis=0)
    return coeffs.reshape(1, -1)


def construct_series(T, c, l, polar, fvals, min_ratio, matdim):
    """
    Calculation of the all the GPSWF's basis coefficients \hat{a}_{N,m,n} of a sampled function.
    :param T: the truncation parameter for the indices.
    :param c: the band-limit parameter.
    :param l: the sampling frequency L.
    :param polar: an array with shape [x, 3] of the spherical coordinates of the sampling grid.
                  assumes data is arranged as in 'shortened_data_points'.
    :param fvals: an array with shape [x,] of the sampled function.
                  assumes data is arranged as in 'construction_data_points'.
    :param min_ratio: threshold for minimal eigenvalue kept.
    :param matdim: dimension of the matrix used in the calculation.
    :return: a list of arrays. each array is of shape [1, n_N(2N+1)] and corresponds to the coefficients of order N.
    """
    coeffs, radials, alphas, truncs = [], [], [], []
    factor = (c / (2 * np.pi)) ** 3
    for N in range(551):
        radial, eigs = radial_coefficients(c, N, min_ratio, matdim)
        if not np.size(eigs):
            break
        truncation = factor * np.abs(eigs) ** 2
        truncation = np.where(np.sqrt(truncation / (1 - truncation)) > T)[0]
        if np.size(truncation):
            truncs.append(np.max(truncation))
        else:
            break
        radials.append(radial[:, :truncs[-1] + 1])
        alphas.append(eigs[:truncs[-1] + 1])
    max_N = len(truncs) - 1
    if not max_N:
        raise ValueError('Truncation parameter T is too large')
    sph_harm = spherical_harmonics(max_N, polar[:, 1], polar[:, 2])
    for N in range(max_N + 1):
        radial = radial_functions(N, radials[N], polar[:, 0], mode='const', l=l)
        coeffs.append(coefficients_fixed_order(c, N, l, radial, alphas[N], sph_harm[:, N, :N+1], fvals))
    return coeffs


# Evaluation given GPSWF basis coefficients #
def evaluate_series(c, l, coeffs, polar, min_ratio, matdim):
    """
    Evaluation of the series given the coefficients in the GPSWF's base.
    :param c: the band-limit parameter.
    :param l: the sampling frequency L.
    :param polar: an array with shape [x, 3] of the spherical coordinates of the sampling grid.
    :param coeffs: a list of arrays.
                   each array is of shape [1, n_N(2N+1)] and corresponds to the coefficients of order N \hat{a}_{N,m,n}.
    :param min_ratio: threshold for minimal eigenvalue kept.
    :param matdim: dimension of the matrix used in the calculation.
    :return: an array with shape [x,] of the evaluation of the series at points (rvals, thetas, phis)
    """
    res = np.zeros(polar.shape[0], dtype=complex)
    truncations = np.array([np.size(lst) for lst in coeffs]) / np.array(range(1, 2 * len(coeffs), 2))
    sph_harm = spherical_harmonics(len(coeffs), polar[:, 1], polar[:, 2])
    for N in range(len(coeffs)):
        radial, eigs = radial_coefficients(c, N, min_ratio, matdim)
        radial = radial_functions(N, radial[:, :int(truncations[N])], polar[:, 0])
        prolate, eigs = prolate_fixed_order(N, l, radial, eigs[:int(truncations[N])], sph_harm[:, N, :N+1], mode='eval')
        res += np.sum(prolate * (eigs.reshape(1,-1) * coeffs[N]), axis=1)
    return res


# Approximation error and error bounds #
def approximation_error(l, fvals, approx):
    """
    Evaluation of the approximation error.
    :param l: the sampling frequency L.
    :param fvals: an array with shape [x,] of the sampled function in the unit ball.
    :param approx: an array with shape [x,] of the approximation in the unit ball.
    :return: the value of the calculated approximation error.
    """
    return np.sqrt(1 / (l ** 3) * np.sum(np.abs(fvals - approx) ** 2))


def delta_c(c, sigma):
    """
    Calculation of the \delta_c parameter in the error bound.
    :param c: the band-limit parameter.
    :param sigma: the variance \sigma.
    :return: the \delta_c parameter for the \sigma value.
    """
    return np.sqrt((2 * np.pi) ** -3 *((np.pi / sigma) ** 1.5 * sp.special.erfc(c * np.sqrt(sigma))
                                       + 2 * c * np.pi / sigma * np.exp(-sigma * c ** 2)))


def eps(mu, sigma):
    """
    Calculation of the \epsilon parameter in the error bound.
    :param mu: an array with shape [3,] of the Gaussian means.
    :param sigma: the variance \sigma.
    :return: the \epsilon parameter for the \sigma value.
    """
    norm = np.linalg.norm(mu)
    return np.sqrt((np.pi * sigma) ** 1.5 * sp.special.erfc((1 - norm)/np.sqrt(sigma))
                   + 2 * np.pi * sigma * (1 - norm) * np.exp(-(1 - norm) ** 2 / sigma))


def error_bound(T, c, l, delta, epsilon, fvals):
    """
    Evaluation of the approximation error bound.
    :param T: the truncation parameter T.
    :param c: the band-limit parameter.
    :param l: the sampling frequency L.
    :param delta: the \delta_c parameter
    :param epsilon: the \epsilon parameter
    :param fvals: an array with shape [x,] of the sampled function in points outside the unit ball.
    :return: the error bound.
    """
    return (epsilon + delta) * T + np.sqrt(4/3 * np.pi * (c / l) ** 3 * np.sum(np.abs(fvals) ** 2)) + 4 * delta


# Data points creations #
def evaluation_data_points(l):
    """
    Creates an array of data points in the cartesian grid.
    :param l: the sampling frequency L.
    :return: an array with shape [(2l + 1)^3, 3] of the data points.
    """
    return (np.indices((2 * l + 1, 2 * l + 1, 2 * l + 1))/ l - 1).reshape((2 * l + 1) ** 3, 3)


def construction_data_points(l):
    """
    Creates an array of data points in the unit ball, in the cartesian grid.
    :param l: the sampling frequency L.
    :return: an array of the data points in the unit ball.
             ordered as: each point in the first quadrant (0,1] * [0,1] * [-1,1] is replicated and rotated to account
                         for the three other quadrants before moving to the next point. the last 2l+1 points are those
                         representing the line [0,0,z].
    """
    res = []
    for x in range(1, l + 1):
        for y in range(l + 1):
            for z in range(-l, l + 1):
                if x ** 2 + y ** 2 + z ** 2 <= l ** 2:
                    res += [[x, y, z], [-y, x, z], [-x, -y, z], [y, -x, z]]
    for z in range(-l, l + 1):
        res.append([0, 0, z])
    return np.array(res) / l


def shortened_data_points(l):
    """
    Creates an array of data points in the first quadrant of unit ball, in the cartesian grid.
    :param l: the sampling frequency L.
    :return: an array of the data points in the unit ball that are in the quadrant (0,1] * [0,1] * [-1,1].
             the last 2l+1 points are those representing the line [0,0,z].
    """
    res = []
    for x in range(1, l + 1):
        for y in range(l + 1):
            for z in range(-l, l + 1):
                if x ** 2 + y ** 2 + z ** 2 <= l ** 2:
                    res += [[x, y, z]]
    for z in range(-l, l + 1):
        res.append([0, 0, z])
    return np.array(res) / l


def cartesian_to_spherical(points):
    """
    Convertion of points in the cartesian form to the spherical form
    :param points: an array with shape [x, 3] of the points in the cartesian form.
    :return: an array with shape [x, 3] of the points in the spherical form.
             the convension used here is r in [0,inf), theta in [0,pi], phi in (-pi, pi].
    """
    res = np.zeros_like(points)
    xy = points[:, 0] ** 2 + points[:, 1] ** 2
    res[:, 0] = np.sqrt(xy + points[:, 2] ** 2)
    res[:, 1] = np.pi/2 - np.arctan2(points[:, 2], xy)
    res[:, 2] = np.arctan2(points[:, 1], points[:, 0])
    return res


# Sanity checks #
def sanity_check_1(order):
    """
    Checks if the calculated zernike polynomials of order N are orthonormal
    :param order: the N parameter of the normalized Zernike polynomials
    :return: Nothing.
             prints an array of shape [20,20]. it is the Gram matrix of the first 20 polynomials.
    """
    xx = 0.0000005 + np.linspace(0, 0.999999, 100000)
    funcs = funcs_from_zernike_base(order, np.identity(20), xx)
    res = np.zeros((20, 20))
    for first in range(20):
        for second in range(first, 20):
            res[first, second] = np.sum(xx**2 * funcs[:, first] * funcs[:, second])/1000000
            res[second, first] = res[first, second]
    print(res)


def sanity_check_2(l):
    """
    Checks if the two modes of calculation gives the same result in "radial_functions"
    :param l: the sampling frequency L.
    :return: nothing.
             prints the norm of the difference between the two methods for prolate order 20.
    """
    coeffs, eigs = radial_coefficients(np.pi * l, 20, 1e-16, 300)
    vals1 = cartesian_to_spherical(shortened_data_points(l))
    vals2 = cartesian_to_spherical(construction_data_points(l))
    res1 = radial_functions(20, coeffs, vals1[:, 0], mode='const', l=l)
    res2 = radial_functions(20, coeffs, vals2[:, 0])
    print(np.linalg.norm(res1 - res2))


def sanity_check_3(l):
    """
    Checks if the eigenvalues calculated in python are the same as MATLAB's.
    :param l: the sampling frequency L.
    :return: nothing.
             prints the relative error of the eigenvalues \gamma_{20,n}.
    """
    matlab = np.array([3.162277660168357e-02, -3.162277660168352e-02, 3.162277660168353e-02, -3.162277660168313e-02,
                       3.162277660158250e-02, -3.162277658487909e-02, 3.162277465151673e-02, -3.162261797571675e-02,
                       3.161390450923598e-02, -3.130729356207943e-02, 2.657618809031583e-02, -1.151994324662320e-02,
                       2.486330785171229e-03, -3.959610714230365e-04, 5.300748077565321e-05, -6.185107304087326e-06,
                       6.411013534288275e-07, -5.979220292932927e-08, 5.065606644685822e-09, -3.927533281899329e-10,
                       2.803710748243992e-11, -1.852073075026927e-12, 1.136985389806555e-13, -6.510765196431446e-15,
                       3.488983911762243e-16, -1.754705808145429e-17])
    print((radial_coefficients(np.pi * l, 20, 1e-16, 300)[1]-matlab).real / matlab)


def sanity_check_4():
    """
    Checks if the calculated \Phi_{N,n} and \beta_{N,n} are eigenfunctions and eigenvalues.
    :return: nothing.
             prints the norm of error between the left hand side and the right hand side of the integral equation for
             \Phi_{20,7} and \beta_{20,7}.
    """
    x_values = 0.00005 + np.linspace(0, 0.9999, 10000).reshape(-1, 1)
    r_values = 0.005 + np.linspace(0, 0.99, 100)
    coeffs, eigs = radial_coefficients(np.pi * 20, 20, 1e-16, 300)
    eigs = eigs.real / (2 * np.pi) ** 1.5
    lhs = eigs[7] * radial_functions(20, coeffs[:, 7].reshape(-1, 1), r_values)
    int = np.pi * 20 * x_values * r_values
    int = sp.special.jn(20.5, int) / np.sqrt(int)
    rhs = np.sum(int * radial_functions(20, coeffs[:, 7].reshape(-1, 1), x_values)
                 * x_values.reshape(-1, 1) ** 2, axis=0).reshape(-1 ,1) / 10000
    print('||LHS - RHS|| = {}'.format(np.linalg.norm((lhs - rhs)/100)))


def sanity_check_5(max_N):
    """
    Checks if the evaluated polynomials are normalized.
    :param max_N: the maximal prolate order N.
    :return: Nothing.
             prints an array with shape [2max_N+1,] of the squared norm of \hat{P}^m_{max_N}.
    """
    xx = np.linspace(-0.9995, 0.9995, 2000)
    yy = legendre_poly(max_N, xx)[:, 20]
    print(np.sum(yy ** 2, axis=0) * 2e-3 * np.pi)


def sanity_check_6(order):
    """
    Checks if the calculated spherical harmonics of order N are orthonormal.
    :param order: the N parameter of the spherical harmonics.
    :return: Nothing.
             prints an array of shape [N+1,N+1]. it is the Gram matrix of S_{N,m} where 0<=m<=N.
    """
    polar = []
    for theta in np.linspace(0, np.pi, 1000):
        for phi in np.linspace(-np.pi, np.pi, 1000):
            polar += [[theta, phi]]
    polar = np.array(polar)
    sph_harm = spherical_harmonics(order, polar[:, 0], polar[:, 1])[:, order, :]
    res = np.zeros((order + 1, order + 1), dtype=complex)
    for first in range(order + 1):
        for second in range(first, order + 1):
            res[first, second] = np.sum(np.conjugate(sph_harm[:, first]) * sph_harm[:, second] * np.sin(polar[:, 0]))
            res[second, first] = res[first, second]
    res *= (2 * np.pi ** 2) / 1000000
    print(res)


def sanity_check_7(l):
    """
    Checks if the two modes of calculation gives the same result in "prolate_fixed_order"
    :param l: the sampling frequency L.
    :return: nothing.
             prints the norm of the difference between the two methods for prolate order 20.
    """
    coeffs, eigs = radial_coefficients(np.pi * l, 20, 1e-16, 300)
    vals1 = cartesian_to_spherical(shortened_data_points(l))
    vals2 = cartesian_to_spherical(construction_data_points(l))
    radial = radial_functions(20, coeffs, vals1[:, 0], mode='const', l=l)
    sph_harm_1 = spherical_harmonics(20, vals1[:, 1], vals1[:, 2])[:,20,:]
    sph_harm_2 = spherical_harmonics(20, vals2[:, 1], vals2[:, 2])[:,20,:]
    res1 = prolate_fixed_order(20, l, radial, eigs, sph_harm_1, mode='const')[0]
    res2 = prolate_fixed_order(20, l, radial, eigs, sph_harm_2, mode='eval')[0]
    print(np.linalg.norm(res1 - res2))


# Results #
def function(vals, mu, sigma):
    return (2 * np.pi * sigma) ** (-1.5) * np.exp(-np.sum((vals - mu) ** 2, axis=1) / (2 * sigma))


means = np.array([0.1, 0.1, 0.1])
sig = 0.01
t_param = 100
times = np.zeros((5, 2))
errors = np.zeros(5)
bound = np.zeros(5)
for rep in range(5):
    L = 16 + 4 * rep
    grid = construction_data_points(L)
    f_values = function(grid, means, sig)
    band_limit = L * np.pi
    start1 = time.time()
    grid = cartesian_to_spherical(shortened_data_points(L))
    coefficients = construct_series(t_param, band_limit, L, grid, f_values, 1e-16, 300)
    end1 = time.time()
    times[rep, 0] = end1 - start1
    start2 = time.time()
    grid = cartesian_to_spherical(construction_data_points(L))
    approximation = evaluate_series(band_limit, L, coefficients, grid, 1e-16, 300)
    end2 = time.time()
    times[rep, 1] = end2 - start2
    errors[rep] = approximation_error(L, f_values, approximation)
    delt = delta_c(band_limit, sig)
    ep = eps(means, sig)
    bound[rep] = error_bound(t_param, band_limit, L, delt, ep, f_values)
    print(rep)

plt.figure(1)
plt.plot([16, 20, 24, 28, 32], times[:, 0])
plt.plot([16, 20, 24, 28, 32], times[:, 1])
plt.legend(['Construction', 'Evaluation'])
plt.xlabel('L')
plt.ylabel('time [sec]')
plt.grid()
plt.show()

plt.figure(2)
plt.plot([16, 20, 24, 28, 32], errors)
plt.legend(['Error'])
plt.xlabel('L')
plt.yscale('log')
plt.grid()
plt.show()

plt.figure(3)
plt.plot([16, 20, 24, 28, 32], bound)
plt.legend(['Bound'])
plt.xlabel('L')
plt.grid()
plt.show()


