import numpy as np
import torch
from scipy.spatial.transform import Rotation

class gaussian_editor:
    """
    Deform gaussians in a scene.
    """
    def __init__(self, editing_modifier_dict):
        assert "scene" in editing_modifier_dict
        assert "objects" in editing_modifier_dict
        self.ground_R = editing_modifier_dict["scene"]["ground_R"]
        self.ground_T = editing_modifier_dict["scene"]["ground_T"]
        self.gravity_up_vec = self.ground_R.T @ np.array((0, 1, 0))
        self.object_editing_actions = editing_modifier_dict["objects"]
    
    def modify_gaussian(self, gaussians, step_idx):
        """
        Modify gaussians in place.
        """
        for obj_dict in self.object_editing_actions:
            selected_obj_idx = obj_dict["affected_gaussian_idx"]
            for action_dict in obj_dict["actions"]:
                action_type = action_dict['action']
                if action_type == "remove":
                    if step_idx == 0:
                        gaussians._opacity[selected_obj_idx] = -1000
                elif action_type == "scaling":
                    if step_idx == 0:
                        # Only scale the first time
                        scale = action_dict['scale']
                        gaussians._xyz[selected_obj_idx] = gaussians._xyz[selected_obj_idx] / scale
                        gaussians._scaling = gaussians.inverse_opacity_activation(
                            gaussians.scaling_activation(gaussians._scaling[selected_obj_idx]) / scale
                        )
                elif action_type == "translate":
                    if step_idx == 0:
                        # Only translate the first time
                        translation = action_dict['translation']
                        translation = torch.tensor(translation, dtype=gaussians._xyz.dtype).to(gaussians._xyz.device)
                        gaussians._xyz[selected_obj_idx] = gaussians._xyz[selected_obj_idx] + translation
                elif action_type == "rotate":
                    if step_idx == 0:
                        # Only rotate the first time
                        rot_mat = action_dict['rotation']

                        # Rotate x/y/z
                        selected_pts = gaussians.get_xyz.cpu().numpy()[selected_obj_idx]
                        object_center = np.mean(selected_pts, axis=0)

                        selected_pts = selected_pts - object_center
                        selected_pts = rot_mat @ selected_pts.T
                        selected_pts = selected_pts.T
                        selected_pts = selected_pts + object_center

                        gaussians.get_xyz[selected_obj_idx] = torch.from_numpy(selected_pts).cuda().float()

                        # Rotate covariance
                        r = gaussians._rotation[selected_obj_idx]
                        rot_mat = rot_mat.reshape((1, 3, 3))  # (N, 3, 3)

                        r = self.get_gaussian_rotation(rot_mat, r)

                        gaussians._rotation[selected_obj_idx] = r
                elif action_type == 'physics':
                    # TODO(roger): implement particle rotation
                    assert 'particles_trajectory_tn3' in action_dict, "Run physics simulation first."
                    particles_trajectory_tn3 = action_dict['particles_trajectory_tn3']
                    assert step_idx < particles_trajectory_tn3.shape[0]
                    particles_pos = torch.from_numpy(particles_trajectory_tn3[step_idx]).float().cuda()
                    gaussians._xyz[selected_obj_idx] = particles_pos

                    if step_idx == 0:
                        self.initial_rotation = gaussians._rotation[selected_obj_idx]

                    # Rotate covariance
                    if 'rot_mat_arr_tn3' in action_dict:
                        rot_mat_arr_tn3 = action_dict['rot_mat_arr_tn3']
                        assert step_idx < rot_mat_arr_tn3.shape[0]
                        rot_mat = rot_mat_arr_tn3[step_idx]

                        r = self.initial_rotation.clone()
                        r = self.get_gaussian_rotation(rot_mat, r)
                        gaussians._rotation[selected_obj_idx] = r
                else:
                    raise NotImplementedError(f"Action type {action_type} not implemented.")
        return gaussians

    def get_gaussian_rotation(self, rot_mat, r):
        norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
        q = r / norm[:, None]

        R = torch.zeros((q.size(0), 3, 3), device='cuda')

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - r*z)
        R[:, 0, 2] = 2 * (x*z + r*y)
        R[:, 1, 0] = 2 * (x*y + r*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - r*x)
        R[:, 2, 0] = 2 * (x*z - r*y)
        R[:, 2, 1] = 2 * (y*z + r*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)

        R = rot_mat @ R.detach().cpu().numpy()

        # Convert back to quaternion
        r = Rotation.from_matrix(R).as_quat()
        r[:, [0, 1, 2, 3]] = r[:, [3, 0, 1, 2]]  # x,y,z,w -> r,x,y,z
        r = torch.from_numpy(r).cuda().float()

        r = r * norm[:, None]
        return r
