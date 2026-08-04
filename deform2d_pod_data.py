from pathlib import Path

import igl
import matplotlib.pyplot as plt
import numpy as np
import simkit

from deform2d import solve


def transform(points, angle, translation):
    """Rotate points about their center, then translate them."""
    center = points.mean(axis=0)
    angle = np.deg2rad(angle)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return (points - center) @ rotation.T + center + translation

if __name__ == "__main__":
    V, F = igl.read_triangle_mesh("data/2d/teddy_bear/teddy_bear.obj")
    V = V[:, :2]

    pins = np.array([2702, 77, 78, 79, 89, 90, 100, 101, 109, 986, 128, 146, 153, 156, 160, 159, 163, 167, 166])
    # moving_pins = np.array([235, 248, 263, 262, 278, 288, 287, 299, 298, 297, 311, 406, 410, 310, 309, 411, 407, 308])
    # moving_pins = np.array([382, 381, 386, 385, 380, 409, 404, 379, 378, 377, 374, 373, 371, 370, 367, 364])
    moving_pins = np.array([35, 34, 45, 44, 52, 51, 60, 59, 66, 2585, 80, 92, 91, 102, 110])
    pinned_indices = np.concatenate((pins, moving_pins))

    vol = igl.doublearea(V, F) / 2
    mu, lam = 1.0, 1.0
    gravity = np.tile([0.0, -0.1], len(V)).reshape(-1, 1)
    deform_jacobian = simkit.deformation_jacobian(V, F)

    output_dir = Path("deformation_snapshots")
    output_dir.mkdir(exist_ok=True)

    width, height = np.ptp(V, axis=0)
    x_shifts = np.linspace(-0.50 * width, 0.50 * width, 10)
    y_shifts = 0.35 * height * np.sin(np.linspace(0.0, 2.0 * np.pi, 10))
    angles = np.linspace(0.0, 0.0, 10)

    for snapshot, (angle, dx, dy) in enumerate(zip(angles, x_shifts, y_shifts)):
        moved_pos = transform(V[moving_pins], angle, np.array([dx, dy]))
        target_pos = np.vstack((V[pins], moved_pos))
        Q_pin, b_pin = simkit.dirichlet_penalty(pinned_indices, target_pos, len(V), 1e5)

        x = solve(V.copy().reshape(-1, 1), deform_jacobian, mu, lam, vol, Q_pin, b_pin, gravity)
        deformed_vertices = x.reshape(-1, 2)

        name = f"snapshot_{(snapshot+20):02d}"
        np.save(output_dir / f"{name}_vertices.npy", deformed_vertices)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.triplot(V[:, 0], V[:, 1], F, color="lightgray", linewidth=0.5)
        ax.triplot(deformed_vertices[:, 0], deformed_vertices[:, 1], F, color="blue", linewidth=0.8)
        ax.scatter(target_pos[:, 0], target_pos[:, 1], color="red", s=12)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.savefig(output_dir / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved {name}")