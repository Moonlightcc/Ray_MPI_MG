# Adapted from Yue's matlab version

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import kron, eye
from scipy.sparse import diags
import time

def lin_interp_mat(xo,xi):
    xs = np.sort(xo)
    is_sorted = np.argsort(xo)
    m = len(xo)
    n = len(xi)
    J = lil_matrix((m, n))

    j0 = 0
    j1 = 1
    x0 = xi[j0]
    x1 = xi[j1]

    for i in range(m):
        while xs[i] > x1:
            j0 = j1
            j1 += 1
            if j1 >= n: 
                break
            x0 = xi[j0]
            x1 = xi[j1]

        if j1 < n:
            J[i, j0] = (xs[i] - x1) / (x0 - x1)
            J[i, j1] = 1 - J[i, j0]

    J = J[is_sorted, :]

    return J

def vars_func(r, x):
    p = np.prod(1 - x[:, None] / r[None, :], axis=0)
    w = x / (1 - p**2)
    f = np.sqrt(np.dot(w, p))
    g_tmp = np.sum(1.0 / (x[:, None] - r[None, :]), axis=0)
    g = x * (1.0/(2.0*w) + g_tmp)
    num = p[None, :] * (x[:, None]**2 + r[None, :] * (x[:, None] - r[None, :]))
    den = x[:, None] - r[None, :]
    ngp = np.sum(num / den, axis=0)
    return w, f, g, ngp

def opt_roots(k):

    # Initial setup
    eps = np.finfo(float).eps
    r = 0.5 * np.cos(np.arange(1, k+1) / (k+5) * np.pi)
    x = 0.5 - 0.5 * np.cos(np.arange(0.5, k, 1) / (k+5) * np.pi)
    dr = np.zeros_like(r)
    drsize = 1.0

    # Iterative refinement
    while drsize > 128 * eps:
        dx = x.copy()
        dxsize = 1.0
        while dxsize > 128 * eps:
            dxsize = np.linalg.norm(dx, np.inf)
            w, f, g, ngp = vars_func(r, x)
            dx = g / ngp
            x = x + dx

        x1 = np.array([x[0]])
        w1, f1, _, _ = vars_func(r, x1)

        f0 = np.sqrt(0.5 / np.sum(1.0 / r))

        J = f0**3 / (r**2) + w1 * abs(f1) / (r * ( (x1 - r)**2 ))

        drsize = np.linalg.norm(dr, np.inf)
        dr = -J * (f0 - abs(f1))
        r = r + dr

    return r



def opt_coef(k):
    x = np.cos(np.arange(1, k + 1) * np.pi / (k + 0.5))
    w = (1 - x) / (k + 0.5)

    W = np.zeros((k, k))
    W[:, 0] = 1  
    if k >= 2:
        W[:, 1] = 2 * x + 1  

    for i in range(2, k):  
        W[:, i] = (2 * x) * W[:, i - 1] - W[:, i - 2]

    r = opt_roots(k)
    lambda_vals = (1 - x) / 2
    p = np.prod(1 - lambda_vals / r)
    alpha = W.T @ (w * p)  

    beta = 1 - np.cumsum(2 * (np.arange(0, k) + 1) * alpha)

    return beta


def v_cycle(A,Nx,Ny,Nz,f,u):
    if Nx == 1:
        A = A.tocsr()
        return spsolve(A, f)

    f_0 = A @ u

    # pre-smoothing config
    # r = f - f_0
    # lmax = 1.9
    # di = 1 / A.diagonal()
    # d = (4 / 3) / lmax * di * r
    # beta = 1  # 4th-kind Chebyshev Smoothing without optimization
    r = f - f_0
    omega = 2/3
    d_inv = (1 / A.diagonal()).reshape(-1,1)

    xc = np.linspace(1 / (Nx + 2), Nx / 2 / (Nx + 1), Nx // 2)
    x = np.linspace(1 / (Nx + 2), Nx / (Nx + 1), Nx)
    Jx = lin_interp_mat(xc, x)

    yc = np.linspace(1 / (Ny + 2), Ny / 2 / (Ny + 1), Ny // 2)
    y = np.linspace(1 / (Ny + 2), Ny / (Ny + 1), Ny)
    Jy = lin_interp_mat(yc, y)

    zc = np.linspace(1 / (Nz + 2), Nz / 2 / (Nz + 1), Nz // 2)
    z = np.linspace(1 / (Nz + 2), Nz / (Nz + 1), Nz)
    Jz = lin_interp_mat(zc, z)

    J = kron(Jz, kron(Jy, Jx))
    nsmooth = 5;
    
    # Pre-smoothing
    # for k in range(1, nsmooth + 1):
    #     beta = opt_coef(k)
    #     u = u + beta*d
    #     r = r - A @ d
    #     d = ((2 * k - 1) / (2 * k + 3)) * d + ((8 * k + 4) / (2 * k + 3) / lmax) * di * r
    # u = u + beta * d
    for nu in range(nsmooth):
        u = u + omega * (d_inv * r)
        r = f - A @ u

    # Restriction 
    rhs = J @ r
    eps = np.zeros_like(rhs)
    eeps = np.zeros_like(eps)
    Ac = J @ A @ J.T
    eps = v_cycle(Ac, Nx // 2, Ny // 2, Nz // 2, rhs, eeps)
    ec = J.T @ eps
    u = u + ec
    r = f - A @ u

    # r = di * (f - A @ u)
    
    # for k in range(1, nsmooth + 1): 
    #     beta = opt_coef(k)
    #     beta = beta[0, 0]
    #     u = u + beta * d
    #     r = r - A @ d
    #     d = ((2 * k - 1) / (2 * k + 3)) * d + ((8 * k + 4) / (2 * k + 3) / lmax) * di * r
    # u = u + beta * d
    for nu in range(nsmooth):
        u = u + omega * (d_inv * r)
        r = f - A @ u

    return u

def tridiag(n):
    """Creates a sparse tridiagonal matrix for size n."""
    e = np.ones(n)
    h = 1 / (n + 1)
    h2i = 1 / (h * h)

    diagonals = [-e, 2 * e, -e] 
    offsets = [-1, 0, 1]         
    A = h2i * diags(diagonals, offsets, shape=(n, n), format="csr")
    return A

def main():

    Nx, Ny, Nz = 16, 16, 16

    Ax = tridiag(Nx)
    Ix = eye(Nx, format='csr')
    Ay = tridiag(Ny)
    Iy = eye(Ny, format='csr')
    Az = tridiag(Nz)
    Iz = eye(Nz, format='csr')

    m = Nx * Ny * Nz
    f = np.random.rand(m, 1)  
    u = np.zeros_like(f)

    A_true = kron(kron(Az, Iy), Ix) + kron(kron(Iz, Ay), Ix) + kron(kron(Iz, Iy), Ax)
    residue = f
    ret = np.zeros_like(f)
    start_time = time.time()

    while np.linalg.norm(residue) / np.linalg.norm(f) > 1e-8:
        u = v_cycle(A_true, Nx, Ny, Nz, f, u)
        residue = f - A_true @ u
        print(np.linalg.norm(residue))

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsA_trueed_time} seconds")

    final_norm = np.linalg.norm(A_true @ u - f)
    print(f"Final norm: {final_norm}")

if __name__ == "__main__":
    main()
