import argparse
from functools import partial

import igl
import simkit
import numpy as np
import scipy as sp
import polyscope as ps
import polyscope.imgui as psim

from deform2d_lma import energy_z, gradient_z, hessian_z, lma


# Add circle collision to the modal energy.
def collision_energy_z(z, B, x_rest, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g, collision_stiffness, circle_center, circle_radius):
    x = x_rest + B @ z
    deformation = energy_z(z, B, x_rest, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g)
    collision = simkit.energies.contact_springs_sphere.contact_springs_sphere_energy(x.reshape(-1, 2), collision_stiffness, circle_center, circle_radius)
    return deformation + collision


# Add circle collision to the modal gradient.
def collision_gradient_z(z, B, x_rest, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g, collision_stiffness, circle_center, circle_radius):
    x = x_rest + B @ z
    deformation = gradient_z(z, B, x_rest, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g)
    collision = simkit.energies.contact_springs_sphere.contact_springs_sphere_gradient(x.reshape(-1, 2), collision_stiffness, circle_center, circle_radius)
    return deformation + B.T @ collision


# Add circle collision to the modal Hessian.
def collision_hessian_z(z, B, x_rest, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g, collision_stiffness, circle_center, circle_radius):
    x = x_rest + B @ z
    deformation = hessian_z(z, B, x_rest, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g)
    collision = simkit.energies.contact_springs_sphere.contact_springs_sphere_hessian(x.reshape(-1, 2), collision_stiffness, circle_center, circle_radius)
    return deformation + B.T @ collision @ B


# Take collision-aware Newton steps in the modal subspace.
def solve(z0, B, x_rest, deform_jacobian, mu, lam, vol, Q_pin, b_pin, g, collision_stiffness, circle_center, circle_radius):
    ener = partial(collision_energy_z, B=B, x_rest=x_rest, deform_jacobian=deform_jacobian, mu=mu, lam=lam, vol=vol, Q_pin=Q_pin, b_pin=b_pin, g=g, collision_stiffness=collision_stiffness, circle_center=circle_center, circle_radius=circle_radius)
    grad = partial(collision_gradient_z, B=B, x_rest=x_rest, deform_jacobian=deform_jacobian, mu=mu, lam=lam, vol=vol, Q_pin=Q_pin, b_pin=b_pin, g=g, collision_stiffness=collision_stiffness, circle_center=circle_center, circle_radius=circle_radius)
    hess = partial(collision_hessian_z, B=B, x_rest=x_rest, deform_jacobian=deform_jacobian, mu=mu, lam=lam, vol=vol, Q_pin=Q_pin, b_pin=b_pin, g=g, collision_stiffness=collision_stiffness, circle_center=circle_center, circle_radius=circle_radius)
    return simkit.solvers.newton.newton_solver(z0, ener, grad, hess, max_iter=1, do_line_search=True)

def load_2d_mesh(filepath):
    vertices, faces = igl.read_triangle_mesh(str(filepath))
    return np.asarray(vertices[:, :2], dtype=float), np.asarray(faces, dtype=int)

class CollisionApp:
    def __init__(self, mesh_file, obstacle_file):
        self.vmesh, self.fmesh = load_2d_mesh(mesh_file)
        self.x_rest = self.vmesh.reshape(-1, 1)

        # pinning
        pinned_indices = np.array([171, 174, 178, 181, 182, 185, 188, 191, 193, 961, 195, 682, 197, 199, 201, 203, 208])
        self.Q_pin, self.b_pin = simkit.dirichlet_penalty(pinned_indices, self.vmesh[pinned_indices], len(self.vmesh), 1000.0)
        self.H = self.Q_pin

        # elastic
        self.mu, self.lam = 1.0, 1.0
        self.J = simkit.deformation_jacobian(self.vmesh, self.fmesh)
        self.vol = igl.doublearea(self.vmesh, self.fmesh) / 2

        self.H += simkit.energies.neo_hookean.neo_hookean_hessian_x(self.vmesh, self.J, self.mu, self.lam, self.vol)

        vertex_mass = igl.massmatrix(self.vmesh, self.fmesh, igl.MASSMATRIX_TYPE_VORONOI)
        self.M = sp.sparse.kron(vertex_mass, sp.sparse.identity(2))

        # gravity
        self.G = np.tile(np.array([0.0, -0.05]), len(self.vmesh)).reshape(-1, 1)

        # subspace
        self.B = lma(self.H, self.M)

        self.quadrature_ratio = 0.1  # Adjustable fraction of elastic triangles to evaluate.
        sample_count = max(1, int(self.quadrature_ratio * len(self.fmesh)))  # Convert the ratio to a sample count.
        sampled_indices = np.random.choice(len(self.fmesh), sample_count, replace=False)  # Draw one fixed random quadrature sample.
        sampled_rows = (4 * sampled_indices[:, None] + np.arange(4)).reshape(-1)  # Select the four Jacobian rows for each 2D triangle.
        self.J = self.J[sampled_rows]  # Evaluate elastic derivatives only on sampled triangles.
        self.vol = self.vol[sampled_indices] * len(self.fmesh) / sample_count  # Reweight sampled areas to approximate the full mesh.

        self.vcirc, self.fcirc = load_2d_mesh(obstacle_file)
        self.circ_center = self.vcirc.mean(axis=0)
        self.circ_radius = np.linalg.norm(self.vcirc - self.circ_center, axis=1).max()
        self.circle_translation = np.zeros(2)
        self.translation_range = float(np.ptp(np.vstack((self.vmesh, self.vcirc)), axis=0).max())
        self.collision_stiffness = 10000.0
        self.z = np.zeros((self.B.shape[1], 1))

    # Advance and display the deformation whenever the circle moves
    def update_deformation(self):
        circle_center = self.circ_center + self.circle_translation
        self.z = solve(self.z, self.B, self.x_rest, self.J, self.mu, self.lam, self.vol, self.Q_pin, self.b_pin, self.G, self.collision_stiffness, circle_center, self.circ_radius)
        self.mesh.update_vertex_positions((self.x_rest + self.B @ self.z).reshape(-1, 2))

    # Reset the mesh and circle to their rest positions
    def reset(self):
        self.z[:] = 0.0
        self.circle_translation[:] = 0.0
        self.mesh.update_vertex_positions(self.vmesh)
        self.circle.update_vertex_positions(self.vcirc)

    def callback(self):
        changed_x, x = psim.SliderFloat("X", float(self.circle_translation[0]), v_min=-self.translation_range, v_max=self.translation_range)
        changed_y, y = psim.SliderFloat("Y", float(self.circle_translation[1]), v_min=-self.translation_range,v_max=self.translation_range)
        if changed_x or changed_y:
            self.circle_translation[:] = (x, y)
            self.circle.update_vertex_positions(self.vcirc + self.circle_translation)
            self.update_deformation()
        if psim.Button("Reset"):
            self.reset()

    def show(self):
        ps.init()
        ps.set_navigation_style("planar")

        self.mesh = ps.register_surface_mesh("Mesh", self.vmesh, self.fmesh, color=(0.25, 0.55, 0.95), edge_width=1.0)
        self.circle = ps.register_surface_mesh("Obstacle", self.vcirc, self.fcirc, color=(0.95, 0.3, 0.2), edge_width=0.0)

        ps.set_user_callback(self.callback)
        ps.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh",default="data/2d/teddy_bear/teddy_bear.obj")
    parser.add_argument("--obstacle",default="data/2d/circle/circle.obj")
    args = parser.parse_args()
    CollisionApp(args.mesh, args.obstacle).show()
