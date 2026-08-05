import igl
import simkit
import numpy as np
import scipy as sp
import polyscope as ps
import polyscope.imgui as psim
from deform2d import energy, gradient, hessian, solve

def two_spring_hessian_xx(x, a, k):
    x = np.asarray(x, dtype=float).reshape(3, -1)
    a = np.asarray(a, dtype=float).reshape(2)

    d = x.shape[1]
    identity = np.eye(d)

    def Ki(xi, xj, ai):
        d = xj - xi
        L = np.linalg.norm(d)

        u = d / L
        uuT = np.outer(u, u)

        return k * (uuT + ((L - ai) / L) * (identity - uuT))

    K1 = Ki(x[0], x[1], a[0])
    K2 = Ki(x[1], x[2], a[1])

    H = np.zeros((3 * d, 3 * d))

    H[0:d, 0:d] += K1
    H[0:d, d:2*d] -= K1
    H[d:2*d, 0:d] -= K1
    H[d:2*d, d:2*d] += K1

    H[d:2*d, d:2*d] += K2
    H[d:2*d, 2*d:3*d] -= K2
    H[2*d:3*d, d:2*d] -= K2
    H[2*d:3*d, 2*d:3*d] += K2

    return H

def two_spring_hessian_xa(x, a, k):
    x = np.asarray(x, dtype=float).reshape(3, -1)
    d = x.shape[1]

    d1 = x[1] - x[0]
    d2 = x[2] - x[1]

    L1 = np.linalg.norm(d1)
    L2 = np.linalg.norm(d2)

    u1 = d1 / L1
    u2 = d2 / L2

    H_xa = np.zeros((3 * d, 2))

    # Column for a1:
    H_xa[0:d, 0] = k * u1
    H_xa[d:2*d, 0] = -k * u1

    # Column for a2:
    H_xa[d:2*d, 1] = k * u2
    H_xa[2*d:3*d, 1] = -k * u2

    return H_xa

def two_spring_third_xxa(x, a, k):
    x = np.asarray(x, dtype=float).reshape(3, -1)
    d = x.shape[1]
    identity = np.eye(d)

    d1 = x[1] - x[0]
    d2 = x[2] - x[1]

    L1 = np.linalg.norm(d1)
    L2 = np.linalg.norm(d2)

    u1 = d1 / L1
    u2 = d2 / L2

    K1 = -(k / L1) * (identity - np.outer(u1, u1))
    K2 = -(k / L2) * (identity - np.outer(u2, u2))

    J_xxa = np.zeros((3*d, 3*d, 2)) # 3 bc there's three verts making up the springs

    J_xxa[0:d, 0:d, 0] += K1
    J_xxa[0:d, d:2*d, 0] -= K1
    J_xxa[d:2*d, 0:d, 0] -= K1
    J_xxa[d:2*d, d:2*d, 0] += K1

    J_xxa[d:2*d, d:2*d, 1] += K2
    J_xxa[d:2*d, 2*d:3*d, 1] -= K2
    J_xxa[2*d:3*d, d:2*d, 1] -= K2
    J_xxa[2*d:3*d, 2*d:3*d, 1] += K2

    return J_xxa

def two_spring_third_xxx(x, a, k, dx_da_i, dx_da_j):
    x = np.asarray(x, dtype=float).reshape(3, -1)
    a = np.asarray(a, dtype=float).reshape(2)
    dx_da_i = np.asarray(dx_da_i, dtype=float).reshape(3, -1)
    dx_da_j = np.asarray(dx_da_j, dtype=float).reshape(3, -1)

    d = x.shape[1]

    d1 = x[1] - x[0]
    d2 = x[2] - x[1]

    L1 = np.linalg.norm(d1)
    L2 = np.linalg.norm(d2)

    u1 = d1 / L1
    u2 = d2 / L2

    dx1_i = dx_da_i[1] - dx_da_i[0]
    dx2_i = dx_da_i[2] - dx_da_i[1]
    dx1_j = dx_da_j[1] - dx_da_j[0]
    dx2_j = dx_da_j[2] - dx_da_j[1]

    t1 = (k * a[0] / L1**2) * (np.dot(u1, dx1_j) * dx1_i + np.dot(u1, dx1_i) * dx1_j + np.dot(dx1_i, dx1_j) * u1 - 3 * np.dot(u1, dx1_i) * np.dot(u1, dx1_j) * u1)
    t2 = (k * a[1] / L2**2) * (np.dot(u2, dx2_j) * dx2_i + np.dot(u2, dx2_i) * dx2_j + np.dot(dx2_i, dx2_j) * u2 - 3 * np.dot(u2, dx2_i) * np.dot(u2, dx2_j) * u2)

    J_xxx = np.zeros(3 * d)

    J_xxx[0:d] -= t1
    J_xxx[d:2*d] += t1

    J_xxx[d:2*d] -= t2
    J_xxx[2*d:3*d] += t2

    return J_xxx

def actuated_static_linear_response(a, a0, x0, H_xx, H_xa):
    delta_a = a - a0
    dx_da = -sp.sparse.linalg.spsolve(H_xx, H_xa)

    x_flat = x0.reshape(-1) + dx_da @ delta_a
    x = x_flat.reshape(x0.shape)

    return x, dx_da  

def actuated_static_quadratic_response(a, a0, x0, H_xx, H_xa, J_xxx, J_xxa):
    x, dx_da = actuated_static_linear_response(a, a0, x0, H_xx, H_xa)
    delta_a = a - a0

    n = x0.size
    m = len(a)
    d2x_da2 = np.zeros((n, m, m))
    x_flat = x.reshape(-1)

    for i in range(m):
        for j in range(m):
            rhs = J_xxx[:, i, j] + J_xxa[:, :, j] @ dx_da[:, i] + J_xxa[:, :, i] @ dx_da[:, j]
            d2x_da2[:, i, j] = -sp.sparse.linalg.spsolve(H_xx, rhs)
            x_flat += 0.5 * d2x_da2[:, i, j] * delta_a[i] * delta_a[j]

    x = x_flat.reshape(x0.shape)

    return x, dx_da, d2x_da2


class SpringActuatorApp:

    def __init__(self, mesh):
        self.v, self.f = igl.read_triangle_mesh(mesh)
        self.v = self.v[:, :2]
        self.v_rest = self.v.copy()

        # pinning
        pinned_indices = np.array([11, 17, 4, 22, 7])
        self.Q_pin, self.b_pin = simkit.dirichlet_penalty(pinned_indices, self.v[pinned_indices], len(self.v), 1000.0)
        self.H_xx = self.Q_pin

        # elasticity
        self.mu, self.lam = 1.0, 1.0
        self.J = simkit.deformation_jacobian(self.v, self.f)
        self.vol = igl.doublearea(self.v, self.f) / 2

        self.H_xx += simkit.energies.neo_hookean.neo_hookean_hessian_x(self.v, self.J, self.mu, self.lam, self.vol)

        # spring
        self.spring_ids = np.array([3, 4, 5])
        spring_x = self.v[self.spring_ids]

        self.k = 1000.0

        self.a_rest = np.array([np.linalg.norm(spring_x[1] - spring_x[0]), 
                            np.linalg.norm(spring_x[2] - spring_x[1])])

        self.a = self.a_rest.copy()
        self.use_quadratic = True  # Start with the quadratic response.

        n, dim = self.v.shape
        spring_dofs = np.concatenate([np.arange(dim * i, dim * i + dim) for i in self.spring_ids])

        H_xx_small = two_spring_hessian_xx(spring_x, self.a_rest, self.k)
        H_xa_small = two_spring_hessian_xa(spring_x, self.a_rest, self.k)
        J_xxa_small = two_spring_third_xxa(spring_x, self.a_rest, self.k)  # Compute the local mixed third derivatives.

        self.H_xx = self.H_xx.tolil()
        self.H_xx[np.ix_(spring_dofs, spring_dofs)] += H_xx_small
        self.H_xx = self.H_xx.tocsc()

        self.H_xa = np.zeros((n*dim, 2))
        self.H_xa[spring_dofs, :] = H_xa_small

        self.J_xxa = np.zeros((n*dim, n*dim, 2))  # Allocate the global mixed third derivatives.
        self.J_xxa[np.ix_(spring_dofs, spring_dofs, np.arange(2))] = J_xxa_small  # Add the local mixed third derivatives.

        _, dx_da = actuated_static_linear_response(self.a_rest, self.a_rest, self.v_rest, self.H_xx, self.H_xa)  # Compute the first-order response.
        dx_da_small = dx_da[spring_dofs, :]  # Restrict the first-order response to the spring vertices.

        self.J_xxx = np.zeros((n*dim, 2, 2))  # Allocate the global contracted spatial third derivatives.
        for i in range(2):  # Loop over the first actuation index.
            for j in range(2):  # Loop over the second actuation index.
                J_xxx_small = two_spring_third_xxx(spring_x, self.a_rest, self.k, dx_da_small[:, i], dx_da_small[:, j])  # Compute one local contraction.
                self.J_xxx[spring_dofs, i, j] = J_xxx_small  # Add the local contraction.

    def update(self):
        if self.use_quadratic:
            x, _, _ = actuated_static_quadratic_response(self.a, self.a_rest, self.v_rest, self.H_xx, self.H_xa, self.J_xxx, self.J_xxa)
        else:
            x, _ = actuated_static_linear_response(self.a, self.a_rest, self.v_rest, self.H_xx, self.H_xa)

        self.mesh.update_vertex_positions(x)
        self.springs.update_node_positions(x[self.spring_ids])

    def callback(self):
        if psim.Button('Use linear response' if self.use_quadratic else 'Use quadratic response'):
            self.use_quadratic = not self.use_quadratic
            self.update()

        changed_a0, a0 = psim.SliderFloat("spring 1", float(self.a[0]), v_min=0.0, v_max=2.0*self.a_rest[0])
        changed_a1, a1 = psim.SliderFloat("spring 2", float(self.a[1]), v_min=0.0, v_max=2.0*self.a_rest[1])

        if changed_a0 or changed_a1:
            self.a[0] = a0
            self.a[1] = a1
            self.update()

    def show(self):
        ps.init()
        ps.set_navigation_style("planar")

        self.mesh = ps.register_surface_mesh("mesh", self.v, self.f, color=(0.25, 0.55, 0.95), edge_width=1.0)

        spring_edges = np.array([[0, 1],
                                 [1, 2]])

        self.springs = ps.register_curve_network("springs", self.v[self.spring_ids], spring_edges, color=(1.0, 0.5, 0.0), radius=0.003, material="flat")

        ps.set_user_callback(self.callback)
        ps.show()


if __name__ == "__main__":
    mesh = "data/2d/T/T.obj"
    SpringActuatorApp(mesh).show()
