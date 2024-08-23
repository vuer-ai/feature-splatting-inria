import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from .utils import point_to_plane_distance, vector_angle

class ground_estimator:
    def __init__(self, distance_threshold=0.005, rotation_flip=False):
        self.distance_threshold = distance_threshold
        self.rotation_flip = rotation_flip

    def estimate(self, ground_pts):
        point_cloud = ground_pts.copy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        plane_model, inliers = pcd.segment_plane(distance_threshold=self.distance_threshold,
                                                ransac_n=3,
                                                num_iterations=2000)
        # [a, b, c, d] = plane_model
        # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        origin_plane_distance = point_to_plane_distance((0, 0, 0), plane_model)

        # Calculate rotation angle between plane normal & z-axis
        plane_normal = tuple(plane_model[:3])
        plane_normal = np.array(plane_normal) / np.linalg.norm(plane_normal)

        # Taichi uses y-axis as up-axis (OpenGL convention)
        if self.rotation_flip:
            y_axis = np.array((0, -1, 0))
        else:
            y_axis = np.array((0, 1, 0))  # Taichi uses y-axis as up-axis
        
        rotation_angle = vector_angle(plane_normal, y_axis)

        # Calculate rotation axis
        rotation_axis = np.cross(plane_normal, y_axis)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        # Generate axis-angle representation
        axis_angle = tuple([x * rotation_angle for x in rotation_axis])

        # Rotate point cloud
        rotation_object = R.from_rotvec(axis_angle)
        rotation_matrix = rotation_object.as_matrix()

        return (rotation_matrix, np.array((0, origin_plane_distance, 0)), inliers)
