import simkit
import igl
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from functools import partial
from time import perf_counter


def energy(x, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g):
    elastic = simkit.energies.neo_hookean.neo_hookean_energy_x(x.reshape(-1, 2), deform_jacobian, mu, lam, vol)
    pinned = (0.5 * x.T @ Q_pin @ x + b_pin.T @ x).item()
    gravity = -(x.T @ g).item()
    return elastic + pinned + gravity

def gradient(x, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g):
    elastic = simkit.energies.neo_hookean.neo_hookean_gradient_x(x.reshape(-1, 2), deform_jacobian, mu, lam, vol)
    pinned = Q_pin @ x + b_pin
    gravity = -g
    return elastic + pinned + gravity

def hessian(x, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g):
    elastic = simkit.energies.neo_hookean.neo_hookean_hessian_x(x.reshape(-1, 2), deform_jacobian, mu, lam, vol)
    pinned = Q_pin
    gravity_hess = sp.sparse.csc_matrix((x.shape[0], x.shape[0]))
    return elastic + pinned + gravity_hess

def solve(x0, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g, iters=30):
    e = partial(energy, deform_jacobian=deform_jacobian, mu=mu, lam=lam, vol=vol, Q_pin=Q_pin, b_pin=b_pin, g=g)
    grad = partial(gradient, deform_jacobian=deform_jacobian, mu=mu, lam=lam, vol=vol, Q_pin=Q_pin, b_pin=b_pin, g=g)
    hess = partial(hessian, deform_jacobian=deform_jacobian, mu=mu, lam=lam, vol=vol, Q_pin=Q_pin, b_pin=b_pin, g=g)
    return simkit.solvers.newton.newton_solver(x0, e, grad, hess, max_iter=iters, do_line_search=True)


if __name__ == "__main__":
    V, F = igl.read_triangle_mesh("data/2d/teddy_bear/teddy_bear.obj")
    V = V[:, :2]

    pinned_indices = np.array([2702, 77, 78, 79, 89, 90, 100, 101, 109, 986, 128, 146, 153, 156, 160, 159, 163, 167, 166])
    target_pos = V[pinned_indices]
    Q_pin, b_pin = simkit.dirichlet_penalty(pinned_indices, target_pos, len(V), 10.0)

    vol = igl.doublearea(V, F) / 2
    mu, lam = 1.0, 1.0
    G = np.tile(np.array([0.0, -0.1]), len(V)).reshape(-1, 1)
    deform_jacobian = simkit.deformation_jacobian(V, F)

    start = perf_counter()
    x = solve(V.copy().reshape(-1, 1), deform_jacobian, mu, lam, vol, Q_pin, b_pin, G)
    print(f"Solve time: {perf_counter() - start:.6f} seconds")
    v = x.reshape(-1, 2)

    plt.figure(figsize=(8, 8))
    plt.triplot(V[:, 0], V[:, 1], F, color="gray", linewidth=0.5, label="initial")
    plt.triplot(v[:, 0], v[:, 1], F, color="blue", linewidth=1.0, label="deformed")
    plt.show()