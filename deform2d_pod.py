import simkit
import igl
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial
from time import perf_counter

from deform2d import energy, gradient, hessian

def energy_z(z, B, x_mean, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g):
    E = energy(x_mean + B @ z, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g)
    return E

def gradient_z(z, B, x_mean, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g):
    grad_x = gradient(x_mean + B @ z, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g)
    return B.T @ grad_x

def hessian_z(z, B, x_mean, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g):
    H_x = hessian(x_mean + B @ z, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g)
    return B.T @ H_x @ B

def pod_subspace(x, num_cols=10):
    x_mean = np.mean(x, axis=1, keepdims=True)
    D = x - x_mean
    U, _, _ = np.linalg.svd(D, full_matrices=False)
    return U[:, :num_cols], x_mean

def solve(z0, B, x_mean, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g, iters=30):
    ener = partial(energy_z, B=B, x_mean=x_mean, deform_jacobian=deform_jacobian, mu=mu, lam=lam, vol=vol, Q_pin=Q_pin, b_pin=b_pin, g=g)
    grad = partial(gradient_z, B=B, x_mean=x_mean, deform_jacobian=deform_jacobian, mu=mu, lam=lam, vol=vol, Q_pin=Q_pin, b_pin=b_pin, g=g)
    hess = partial(hessian_z, B=B, x_mean=x_mean, deform_jacobian=deform_jacobian, mu=mu, lam=lam, vol=vol, Q_pin=Q_pin, b_pin=b_pin, g=g)

    return simkit.solvers.newton.newton_solver(z0, ener, grad, hess, max_iter=iters, do_line_search=True)

if __name__ == "__main__":
    snapshots_dir = Path("./deformation_snapshots")
    npy_files = [str(p) for p in sorted(snapshots_dir.glob("*.npy"))][0:6]
    X = np.column_stack([np.load(v).reshape(-1, 1) for v in npy_files])

    V, F = igl.read_triangle_mesh("data/2d/teddy_bear/teddy_bear.obj")
    V = V[:, :2]

    pinned_indices = np.array([235, 248, 263, 262, 278, 288, 287, 299, 298, 297, 311, 406, 410, 310, 309, 411, 407, 308])
    target_pos = V[pinned_indices]
    Q_pin, b_pin = simkit.dirichlet_penalty(pinned_indices, target_pos, len(V), 10.0)

    vol = igl.doublearea(V, F) / 2
    mu, lam = 1.0, 1.0
    G = np.tile(np.array([0.0, -0.1]), len(V)).reshape(-1, 1)
    deform_jacobian = simkit.deformation_jacobian(V, F)

    B, x_mean = pod_subspace(X)
    z0 = np.zeros((B.shape[1], 1))

    start = perf_counter()
    z = solve(z0, B, x_mean, deform_jacobian, mu, lam, vol, Q_pin, b_pin, G)
    print(f"Solve time: {perf_counter() - start:.6f} seconds")
    
    x = x_mean + B @ z
    v = x.reshape(-1, 2)

    plt.figure(figsize=(8, 8))
    plt.triplot(V[:, 0], V[:, 1], F, color="gray", linewidth=0.5, label="initial")
    plt.triplot(v[:, 0], v[:, 1], F, color="blue", linewidth=1.0, label="deformed")
    plt.show()

