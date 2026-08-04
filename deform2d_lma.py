import argparse
from functools import partial

import igl
import numpy as np
import polyscope
import polyscope.imgui as psim
import scipy as sp
import simkit

from deform2d import energy, gradient, hessian


def energy_z(z, B, x_rest, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g):
    return energy(x_rest + B @ z, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g)


def gradient_z(z, B, x_rest, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g):
    grad_x = gradient(x_rest + B @ z, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g)
    return B.T @ grad_x


def hessian_z(z, B, x_rest, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g):
    H_x = hessian(x_rest + B @ z, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g)
    return B.T @ H_x @ B


def solve(z0, B, x_rest, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g, iters=30):
    ener = partial(energy_z, B=B, x_rest=x_rest, deform_jacobian=deform_jacobian, mu=mu, lam=lam, vol=vol, Q_pin=Q_pin, b_pin=b_pin, g=g)
    grad = partial(gradient_z, B=B, x_rest=x_rest, deform_jacobian=deform_jacobian, mu=mu, lam=lam, vol=vol, Q_pin=Q_pin, b_pin=b_pin, g=g)
    hess = partial(hessian_z, B=B, x_rest=x_rest, deform_jacobian=deform_jacobian, mu=mu, lam=lam, vol=vol, Q_pin=Q_pin, b_pin=b_pin, g=g)

    return simkit.solvers.newton.newton_solver(z0, ener, grad, hess, max_iter=iters, do_line_search=True)


def lma(H, M, num_modes=10):
    """Linear modal analysis."""
    eigvals, eigvecs = sp.sparse.linalg.eigsh(H, k=num_modes, M=M, sigma=0.0, which="LM")
    idx = np.argsort(eigvals)
    return eigvecs[:, idx]


class DeformationApp:
    def __init__(self, mesh_file):
        self.V, self.F = igl.read_triangle_mesh(mesh_file)
        self.V = self.V[:, :2]
        
        self.mu, self.lam = 1.0, 1.0
        self.J = simkit.deformation_jacobian(self.V, self.F)
        self.vol = igl.doublearea(self.V, self.F) / 2

        self.H = simkit.energies.neo_hookean.neo_hookean_hessian_x(self.V, self.J, self.mu, self.lam, self.vol)
        vertex_mass = igl.massmatrix(self.V, self.F, igl.MASSMATRIX_TYPE_VORONOI)
        self.M = sp.sparse.kron(vertex_mass, sp.sparse.identity(2))

        self.G = np.tile(np.array([0.0, -0.1]), len(self.V)).reshape(-1, 1)

        self.x_rest = self.V.reshape(-1, 1)

        self.pinned_indices = []

        self.deformed_vertices = None
        self.deformed = False

    def update_pinned_vertices(self):
        polyscope.remove_point_cloud("Pinned vertices", error_if_absent=False)
        if self.pinned_indices:
            pinned_indices = np.array(self.pinned_indices, dtype=int)
            polyscope.register_point_cloud("Pinned vertices", self.V[pinned_indices], point_render_mode="quad", color=(1.0, 0.1, 0.1), transparency=0.75)

    def select_vertices(self):
        io = psim.GetIO()
        if not io.MouseClicked[0]:
            return
        pick = polyscope.pick(screen_coords=io.MousePos)
        if not pick.is_hit:
            return
        if pick.structure_name == "Mesh":
            vertex_index = int(pick.structure_data["index"])
            if vertex_index in self.pinned_indices:
                self.pinned_indices.remove(vertex_index)
            else:
                self.pinned_indices.append(vertex_index)
                self.pinned_indices.sort()
        elif pick.structure_name == "Pinned vertices":
            point_index = int(pick.structure_data["index"])
            self.pinned_indices.pop(point_index)
        else:
            return
        self.update_pinned_vertices()
        polyscope.reset_selection()

    def apply_deformation(self):
        if not self.pinned_indices or self.deformed:
            return
        
        pinned_indices = np.array(self.pinned_indices, dtype=int)
        Q_pin, b_pin = simkit.dirichlet_penalty(pinned_indices, self.V[pinned_indices], len(self.V), 10000.0)

        H = self.H + Q_pin
        B = lma(H, self.M)
        
        z0 = np.zeros((B.shape[1], 1))
        z = solve(z0, B, self.x_rest, self.J, self.mu, self.lam, self.vol, Q_pin, b_pin, self.G)
        self.deformed_vertices = (self.x_rest + B @ z).reshape(-1, 2)

        self.mesh.update_vertex_positions(self.deformed_vertices)
        self.deformed = True

    def reset(self):
        self.pinned_indices.clear()
        self.deformed = False
        self.mesh.update_vertex_positions(self.V)
        self.update_pinned_vertices()
        polyscope.reset_selection()

    def callback(self):
        psim.TextUnformatted("Left-click a vertex to pin or unpin it.")
        psim.TextUnformatted(f"Pinned vertices: {len(self.pinned_indices)}")
        if not self.deformed:
            self.select_vertices()
        if psim.Button("Apply deformation"):
            self.apply_deformation()
        if not self.pinned_indices:
            psim.TextUnformatted("Select at least one vertex before applying deformation.")
        elif self.deformed:
            psim.TextUnformatted("Reset to choose new pins.")
        if psim.Button("Reset"):
            self.reset()

    def show(self):
        polyscope.init()
        polyscope.set_navigation_style("planar")
        self.mesh = polyscope.register_surface_mesh("Mesh", self.V, self.F, edge_width=1.0)
        self.mesh.set_selection_mode("vertices_only")
        polyscope.set_user_callback(self.callback)
        polyscope.show()


def main():
    parser = argparse.ArgumentParser(description="Interactively pin and deform a 2D triangle mesh.")
    parser.add_argument("--mesh-file", default="data/2d/teddy_bear/teddy_bear.obj", help="Path to the triangle mesh file.")
    args = parser.parse_args()
    DeformationApp(args.mesh_file).show()


if __name__ == "__main__":
    main()
