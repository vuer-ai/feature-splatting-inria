import numpy as np
import taichi as ti
from sklearn.neighbors import BallTree

gravity = ti.Vector([0, -9.8, 0])
spring_Y = 3e4
dashpot_damping = 1e4
drag_damping = 1

@ti.data_oriented
class cloth_simulator:
    def __init__(self):
        # Time per substep
        self.dt = 4e-5

        # Substep is the number of steps per 1/60 second
        self.substeps = int(1 / 60 // self.dt)

        self.ball_radius = 0.3
        self.ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
        self.ball_center[0] = [0, 0, 0]
    
    @ti.kernel
    def substep(self, x: ti.template(), v: ti.template(), quad_size: ti.template(), n: ti.template(), connected_springs: ti.template(), original_dist_matrix: ti.template()):
        for i in ti.grouped(x):
            v[i] += gravity * self.dt

        for i in ti.grouped(x):
            force = ti.Vector([0.0, 0.0, 0.0])
            for connected_idx in range(connected_springs.shape[1]):
                j = connected_springs[i, connected_idx]
                if j == -1:
                    continue
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = original_dist_matrix[i, j]
                # Spring force
                force += -spring_Y * d * (current_dist / original_dist - 1)
                # Dashpot damping
                force += -v_ij.dot(d) * d * dashpot_damping * quad_size

            v[i] += force * self.dt

        for i in ti.grouped(x):
            v[i] *= ti.exp(-drag_damping * self.dt)
            offset_to_center = x[i] - self.ball_center[0]
            if offset_to_center.norm() <= self.ball_radius:
                # Velocity projection
                normal = offset_to_center.normalized()
                v[i] -= ti.min(v[i].dot(normal), 0) * normal
            x[i] += self.dt * v[i]

def main():
    arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
    ti.init(arch=arch)

    n = 128
    quad_size = 1.0 / n

    x = ti.Vector.field(3, dtype=float, shape=n * n)
    v = ti.Vector.field(3, dtype=float, shape=n * n)

    num_triangles = (n - 1) * (n - 1) * 2
    indices = ti.field(int, shape=num_triangles * 3)
    vertices = ti.Vector.field(3, dtype=float, shape=n * n)
    colors = ti.Vector.field(3, dtype=float, shape=n * n)

    # ======= Make a uniform grid of particles =======
    @ti.kernel
    def initialize_mass_points():
        random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1

        for idx in ti.ndrange(n * n):
            i = idx // n
            j = idx % n
            x[idx] = [
                i * quad_size - 0.5 + random_offset[0],
                0.6,
                j * quad_size - 0.5 + random_offset[1],
            ]
            v[idx] = [0, 0, 0]
    
    initialize_mass_points()

    @ti.kernel
    def initialize_mesh_indices():
        for i, j in ti.ndrange(n - 1, n - 1):
            quad_id = (i * (n - 1)) + j
            # 1st triangle of the square
            indices[quad_id * 6 + 0] = i * n + j
            indices[quad_id * 6 + 1] = (i + 1) * n + j
            indices[quad_id * 6 + 2] = i * n + (j + 1)
            # 2nd triangle of the square
            indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
            indices[quad_id * 6 + 4] = i * n + (j + 1)
            indices[quad_id * 6 + 5] = (i + 1) * n + j

        for i, j in ti.ndrange(n, n):
            if (i // 4 + j // 4) % 2 == 0:
                colors[i * n + j] = (0.22, 0.72, 0.52)
            else:
                colors[i * n + j] = (1, 0.334, 0.52)

    initialize_mesh_indices()

    # ======= Compute connected springs and original distances =======
    assert x.shape == v.shape
    assert len(x.shape) == 1
    n_vertices = x.shape[0]

    original_dist_matrix = ti.field(float, shape=(n_vertices, n_vertices))

    @ti.kernel
    def initialize_original_dist_matrix(x: ti.template()):
        for i, j in ti.ndrange(n_vertices, n_vertices):
            original_dist = (x[i] - x[j]).norm()
            original_dist_matrix[i, j] = original_dist
    
    initialize_original_dist_matrix(x)

    tree = BallTree(x.to_numpy())

    # TODO: combine radius and KNN
    K = 8
    result = tree.query(x.to_numpy(), k=K + 1)
    max_spring_count = np.unique(result[1], return_counts=True)[1].max() + K
    connected_springs = ti.field(int, shape=(n_vertices, max_spring_count))
    connected_springs.fill(-1)

    knn_arr = ti.field(int, shape=(n_vertices, K))
    knn_arr.from_numpy(result[1][:, 1:].astype(np.int32))

    num_triangles = indices.shape[0] // 3

    @ti.kernel
    def initialize_connected_springs(triangle_indices: ti.template(), num_triangles: ti.template(), connected_springs: ti.template()):
        # From KNN
        for i in ti.ndrange(n_vertices):
            for j in ti.static(range(K)):
                connected_springs[i, j] = knn_arr[i, j]

                # Connected springs are bidirectional
                # other_spring_idx = K
                # while connected_springs[knn_arr[i, j], other_spring_idx] != -1:
                #     other_spring_idx += 1
                # connected_springs[knn_arr[i, j], other_spring_idx] = i
    
    initialize_connected_springs(indices, num_triangles, connected_springs)

    @ti.kernel
    def update_vertices():
        for i, j in ti.ndrange(n, n):
            vertices[i * n + j] = x[i * n + j]

    window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (768, 768), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    current_t = 0.0

    my_simulator = cloth_simulator()

    fps = 60
    current_t = 0.0

    while window.running:
        if current_t > 3:
            break

        for i in range(my_simulator.substeps):
            my_simulator.substep(x, v, quad_size, n, connected_springs, original_dist_matrix)

        current_t += (1. / fps)
        update_vertices()

        camera.position(0.0, 0.0, 3)
        camera.lookat(0.0, 0.0, 0)
        scene.set_camera(camera)

        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.mesh(vertices, indices=indices, per_vertex_color=colors, two_sided=True)

        # Draw a smaller ball to avoid visual penetration
        scene.particles(my_simulator.ball_center, radius=my_simulator.ball_radius * 0.95, color=(0.5, 0.42, 0.8))
        canvas.scene(scene)
        window.show()

if __name__ == "__main__":
    main()
