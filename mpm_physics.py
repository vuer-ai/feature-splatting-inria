import os
import pickle
import numpy as np
import taichi as ti
from tqdm import tqdm, trange
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.system_utils import searchForMaxIteration
from submodules.mpm_engine.mpm_solver import MPMSolver

def main(dataset, sim_res, infilling_method, material_type, rigid_speed, use_rigidity):
    load_iters = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))

    gaussians = GaussianModel(dataset.sh_degree, dataset.distill_feature_dim)

    gaussians.load_ply(os.path.join(dataset.model_path,
                                    "point_cloud",
                                    "iteration_" + str(load_iters),
                                    "point_cloud.ply"))

    modifier_dict_pt = os.path.join(dataset.model_path,
                                        "point_cloud",
                                        "iteration_" + str(load_iters),
                                        "editing_modifier.pkl")

    with open(modifier_dict_pt, "rb") as f:
        editing_modifier_dict = pickle.load(f)

    # Gather necessary variables
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    ground_R = editing_modifier_dict['scene']['ground_R']
    ground_T = editing_modifier_dict['scene']['ground_T']

    for obj_idx in range(len(editing_modifier_dict['objects'])):
        obj_dict = editing_modifier_dict['objects'][obj_idx]
        for action_idx in range(len(obj_dict['actions'])):
            action_dict = obj_dict['actions'][action_idx]
            if action_dict['action'] == 'physics':
                obj_dict['actions'][action_idx] = sim_mpm_physics(obj_dict, action_dict,
                                                                  xyz, sim_res, ground_R, ground_T,
                                                                  infilling_method, material_type, rigid_speed, use_rigidity)

    with open(modifier_dict_pt, "wb") as f:
        pickle.dump(editing_modifier_dict, f)

def sim_mpm_physics(obj_dict, action_dict, xyz, sim_res, ground_R, ground_T,
                    infilling_method, material_type, rigid_speed, use_rigidity):
    # constants
    infilling_voxel_res = 128
    GROUND_Y = 0.05
    support_per_particles = 20
    youngs_modulus_scale = 2.5 # larger Young’s modulus E indicates higher stiffness
    poisson_ratio = 0.24 # a larger poission ratio ν leads to better volume preservation
    max_surface_particles = 10000

    real_gaussian_particle = xyz[obj_dict['affected_gaussian_idx']]
    real_gaussian_particle_size = real_gaussian_particle.shape[0]

    # get idx of rigid particles
    rigid_idx = np.zeros(real_gaussian_particle_size, dtype=bool)
    if use_rigidity and action_dict['static_idx'] is not None:
        rigid_idx = action_dict['static_idx']

    surface_particles = action_dict['infilling_surface_pts']
    surface_particles = np.random.permutation(surface_particles)[:int(max_surface_particles)]
    surface_particles.mean(axis=0)

    all_particles = np.concatenate([real_gaussian_particle, surface_particles], axis=0)
    all_particles.shape

    particles = all_particles @ ground_R.T
    particles += ground_T

    # Normalize everything to a unit world box; x-z coordinates are centered at 0.5
    particle_max = particles.max(axis=0)
    particle_min = particles.min(axis=0)
    particle_min[1] = min(particle_min[1], GROUND_Y)

    longest_side = max(particle_max - particle_min)

    particles[:, 0] /= longest_side
    particles[:, 1] /= longest_side
    particles[:, 2] /= longest_side

    # Align centers of x and z to 0.5
    # Set the bottom of the object to 0
    shift_constant = np.array([
        -particles[:,0].mean() + 0.5,
        -particles[:,1].min(),
        -particles[:,2].mean() + 0.5
    ])

    particles += shift_constant

    particles, rigid_flag = infill_particles(infilling_method, infilling_voxel_res, support_per_particles,
                                             real_gaussian_particle, rigid_idx, surface_particles, particles)

    ti.init(arch=ti.cuda, device_memory_GB=6.0)
    gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41)

    mpm = MPMSolver(res=(sim_res, sim_res, sim_res), size=1, max_num_particles=2 ** 21,
                    E_scale=youngs_modulus_scale, poisson_ratio=poisson_ratio, unbounded=True)

    mpm.add_particles(particles=particles,
                material=material_type,
                color=0xFFFF00, motion_override_flag_arr=rigid_flag)

    mpm.add_surface_collider(point=(0.0, GROUND_Y, 0.0),
                                normal=(0, 1, 0),
                                surface=mpm.surface_sticky)

    mpm.set_gravity((0, -4.5, 0))  # TODO(maybe use custom gravity?)

    particles_trajectory = []

    for frame in trange(500):
        particles_info = mpm.particle_info()
        real_gaussian_pos = particles_info['position'][:real_gaussian_particle_size]
        particles_trajectory.append(real_gaussian_pos.copy())

        # 2D visualization
        np_x = particles_info['position'][:real_gaussian_particle_size] / 1.0
        screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2**0.5) - 0.2
        screen_y = (np_x[:, 1])
        screen_pos = np.stack([screen_x, screen_y], axis=-1)

        gui.circles(screen_pos, radius=2, color=particles_info['color'][:real_gaussian_particle_size])
        gui.show()

        if frame < 100:
            override_velocity = [0, 0, 0]
        else:
            cycle_idx = frame // 30
            if cycle_idx % 2 == 0:
                override_velocity = [rigid_speed, 0, 0]
            else:
                override_velocity = [-rigid_speed, 0, 0]
        mpm.step(4e-3, override_velocity=override_velocity)

    particles_trajectory_tn3 = np.stack(particles_trajectory)

    particles_trajectory_tn3 -= shift_constant
    particles_trajectory_tn3 *= longest_side

    # Reverse rigid transformation
    particles_trajectory_tn3 = particles_trajectory_tn3 - ground_T
    particles_trajectory_tn3 = particles_trajectory_tn3 @ ground_R

    assert particles_trajectory_tn3.shape[1] == real_gaussian_particle_size

    action_dict['particles_trajectory_tn3'] = particles_trajectory_tn3

    return action_dict

def infill_particles(infilling_method, infilling_voxel_res, support_per_particles, real_gaussian_particle, rigid_idx, surface_particles, particles):
    voxel_occupancy_arr = np.zeros((infilling_voxel_res, infilling_voxel_res, infilling_voxel_res), dtype=np.uint8)
    rigid_voxel = np.zeros((infilling_voxel_res, infilling_voxel_res, infilling_voxel_res), dtype=bool)

    # Create a voxelized version of the particles
    for particles_idx in range(real_gaussian_particle.shape[0]):
        start_pos = particles[:real_gaussian_particle.shape[0]].mean(axis=0)
        end_pos = particles[particles_idx]
        for support_idx in range(support_per_particles):
            # interpolate
            pos = (start_pos * (support_per_particles - support_idx) + end_pos * support_idx) / support_per_particles
            voxel_pos = (pos * infilling_voxel_res).astype(int)
            voxel_occupancy_arr[voxel_pos[0], voxel_pos[1], voxel_pos[2]] = 1
            rigid_voxel[voxel_pos[0], voxel_pos[1], voxel_pos[2]] = rigid_idx[particles_idx] | rigid_voxel[voxel_pos[0], voxel_pos[1], voxel_pos[2]]

    if infilling_method == 'simple_interpolation':
        infilled_particles = np.mgrid[0:infilling_voxel_res, 0:infilling_voxel_res, 0:infilling_voxel_res].reshape(3, -1).T
        infilled_particles = infilled_particles[voxel_occupancy_arr.flatten() == 1] / infilling_voxel_res
        infilled_rigid_flag = rigid_voxel.flatten()[voxel_occupancy_arr.flatten() == 1]
    elif infilling_method == 'ray_testing':
        raise NotImplementedError("Not implemented yet")
    else:
        raise ValueError("Invalid infilling method: {}".format(infilling_method))

    particles = np.concatenate([particles, infilled_particles], axis=0)

    rigid_flag = rigid_idx.astype(np.int32)
    rigid_flag = np.concatenate([rigid_flag, np.zeros(surface_particles.shape[0], dtype=np.int32), np.array(infilled_rigid_flag, dtype=np.int32)], axis=0)
    return particles, rigid_flag

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--sim_res", default=64, type=int)
    parser.add_argument("--infilling_method", default="simple_interpolation", choices=["simple_interpolation", "ray_testing"])
    parser.add_argument("--material_type", default="elastic", choices=["elastic", "snow", "sand"])
    parser.add_argument("--rigid_speed", default=0.0, type=float)
    parser.add_argument("--use_rigidity", default=False, action="store_true")
    args = get_combined_args(parser)
    print("Simulating physics for " + args.model_path)

    if args.material_type == "elastic":
        material_type = MPMSolver.material_elastic
    elif args.material_type == "snow":
        material_type = MPMSolver.material_snow
    elif args.material_type == "sand":
        material_type = MPMSolver.material_sand
    elif args.material_type == "water":
        material_type = MPMSolver.material_water
    else:
        raise ValueError("Invalid material type: {}".format(args.material_type))

    main(model.extract(args), args.sim_res, args.infilling_method, material_type, args.rigid_speed, args.use_rigidity)