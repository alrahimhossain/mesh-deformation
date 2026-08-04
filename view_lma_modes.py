import argparse
import igl
import numpy as np
import polyscope
import polyscope.imgui as psim
import scipy as sp
import simkit
from deform2d_lma import lma

class ModalViewer:
    def __init__(self, mesh_path, num_modes):
        self.V, self.F = igl.read_triangle_mesh(mesh_path)
        self.V = self.V[:, :2]

        J = simkit.deformation_jacobian(self.V, self.F)
        vol = igl.doublearea(self.V, self.F) / 2

        H = simkit.energies.neo_hookean.neo_hookean_hessian_x(self.V, J, 1.0, 1.0, vol)
        vertex_mass = igl.massmatrix(self.V, self.F, igl.MASSMATRIX_TYPE_VORONOI)
        M = sp.sparse.kron(vertex_mass, sp.sparse.identity(2))
        self.modes = lma(H, M, num_modes)
        
        self.num_modes = num_modes
        self.mode_number = 0
        self.phase = 0.0
        self.amplitude = 0.1 * np.linalg.norm(np.ptp(self.V, axis=0))
        rest_edges = np.stack((self.V[self.F[:, 1]] - self.V[self.F[:, 0]], self.V[self.F[:, 2]] - self.V[self.F[:, 0]]), axis=2)
        self.rest_edge_inverse = np.linalg.inv(rest_edges)

    def triangle_strain(self, displacement):
        displacement_edges = np.stack((displacement[self.F[:, 1]] - displacement[self.F[:, 0]], displacement[self.F[:, 2]] - displacement[self.F[:, 0]]), axis=2)
        displacement_gradient = displacement_edges @ self.rest_edge_inverse
        strain = 0.5 * (displacement_gradient + displacement_gradient.transpose(0, 2, 1))
        return np.linalg.norm(strain, axis=(1, 2))

    def callback(self):
        changed, self.mode_number = psim.SliderInt("Mode", self.mode_number, 0, self.num_modes-1)
        if changed:
            self.phase = 0.0
        mode = self.modes[:, self.mode_number].reshape(-1, 2)
        mode /= np.max(np.linalg.norm(mode, axis=1))
        displacement = self.amplitude * np.sin(self.phase) * mode
        face_deformation = self.triangle_strain(displacement)
        self.mesh.update_vertex_positions(self.V + displacement)
        self.mesh.add_scalar_quantity("Deformation", face_deformation, defined_on="faces", enabled=True, vminmax=(0.0, 0.1), cmap="reds")
        self.phase += 0.1

    def show(self):
        polyscope.init()
        polyscope.set_navigation_style("planar")
        self.mesh = polyscope.register_surface_mesh("Mesh", self.V, self.F, edge_width=1.0)
        polyscope.set_user_callback(self.callback)
        polyscope.show()

def main():
    parser = argparse.ArgumentParser(description="View the modal deformations of a 2D mesh.")
    parser.add_argument("--mesh", default="data/2d/teddy_bear/teddy_bear.obj", help="Path to the triangle mesh file.")
    parser.add_argument("--num-modes", type=int, default=20, help="Number of modes to compute.")
    args = parser.parse_args()

    ModalViewer(args.mesh, args.num_modes).show()

if __name__ == "__main__":
    main()
