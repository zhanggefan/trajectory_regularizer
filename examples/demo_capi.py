import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from motreg import capi
from scipy.spatial.transform import Rotation as R
import os.path as osp
import json


def regularize(traj_boxes):
    input = []
    for item in traj_boxes:
        box = capi.ObjBBox()
        box.sequence = item['sequence']
        box.timestamp = item['timestamp']
        box.boxBottomCtrXYZ = item['pose'][:3, 3]
        box.boxRotationXYZW = R.from_matrix(item['pose'][:3, :3]).as_quat()
        box.boxType = capi.BoxType.Normal
        input.append(box)
    model_params = capi.MotionModelParams()
    model_params.verbose = True
    model_params.weightMotion = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    model_params.weightMotionConsistency = [1.0, 1.0]
    model_params.weightObjPose2Label = [0.2, 0.2, 0.2, 0.5, 0.5, 0.2]
    model = capi.MotionModel(input, model_params)
    outputs = model.output([_['sequence'] for _ in traj_boxes])
    for obj, output in zip(traj_boxes, outputs):
        obj['sequence'] = output.sequence
        obj['timestamp'] = output.timestamp
        obj['pose'][:3, :3] = R.from_quat(output.boxRotationXYZW).as_matrix()
        obj['pose'][:3, 3] = output.boxBottomCtrXYZ
        obj['boxType'] = int(output.boxType)
        obj['motion'] = output.motion
        obj['errMotionFwd'] = output.errMotionFwd
        obj['errMotionBwd'] = output.errMotionBwd
        obj['errLabel'] = output.errLabel

    return traj_boxes


def box3d_corners(dims):
    if dims.shape[0] == 0:
        return np.empty([0, 8, 3], dtype=dims.dtype)
    corners_norm = np.stack(np.unravel_index(
        np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0], dtype=dims.dtype)
    corners = dims.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
    return corners


def renderbox(trajectories, color_amp=1.0):
    clr_map = plt.get_cmap('tab10').colors
    corners = box3d_corners(
        np.stack([obj['size'] for obj in trajectories]))
    vels_norm = None
    if 'motion' in trajectories[0]:
        vels_norm = np.stack([obj['motion'][0] for obj in trajectories])

    cores = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
        (8, 4), (8, 5), (8, 6), (8, 7)
    ]
    ret = None
    vel_vectors = None
    for i, corners_i in enumerate(corners):
        label_i = trajectories[i]['gt_labels_3d']
        bpose_i = trajectories[i]['pose']
        corners_i = corners_i.astype(np.float64)
        frontcenter = corners_i[[4, 5, 6, 7]].mean(axis=0, keepdims=True)
        heading = corners_i[4] - corners_i[0]
        frontcenter += 0.3 * heading / np.linalg.norm(heading)
        corners_i = np.concatenate((corners_i, frontcenter), axis=0)
        corners_i = o3d.utility.Vector3dVector(corners_i)
        corners_i = o3d.geometry.PointCloud(points=corners_i)

        if vels_norm is not None:  # with velocity
            vel_norm = vels_norm[i].item()
            if vel_norm > 0:
                vel_vector = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=0.1, cone_radius=0.3,
                    cylinder_height=vel_norm, cone_height=0.5)
                R = vel_vector.get_rotation_matrix_from_xyz(
                    (0, np.pi / 2, 0))
                vel_vector.rotate(R, center=(0, 0, 0))

                vel_vector.rotate(bpose_i[:3, :3], center=(0, 0, 0))
                vel_vector.translate(bpose_i[:3, 3])

                vel_vector.paint_uniform_color(
                    [color_amp * c for c in
                     clr_map[label_i % len(clr_map)]])

                if vel_vectors is None:
                    vel_vectors = vel_vector
                else:
                    vel_vectors += vel_vector

        box = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            corners_i, corners_i, cores)

        error = trajectories[i].get('errLabel', None)
        if error is None or np.abs(error).max() < 0.025:
            color = [color_amp *
                     c for c in clr_map[label_i % len(clr_map)]]
        else:
            color = [1., 1., 1.]
        box.paint_uniform_color(color)

        box.rotate(bpose_i[:3, :3], center=(0, 0, 0))
        box.translate(bpose_i[:3, 3])

        if ret is None:
            ret = box
        else:
            ret += box
    return ret, vel_vectors


def main():
    with open(osp.join(osp.split(__file__)[0], 'demo.json')) as f:
        all_infos = json.load(f)
    all_infos = sorted(all_infos, key=lambda x: x['_id'])

    def vis_iter_out(all_infos):
        for i, info in enumerate(all_infos):
            traj_boxes = info['trajectory']
            pose_0_inv = np.linalg.inv(np.array(info['base_pose']))
            for seq in traj_boxes:
                rel_pose = pose_0_inv @ np.array(seq['ego_pose'])
                b = seq['gt_boxes_3d']
                seq['timestamp'] = (
                                           seq['pts_info']['timestamp'] - traj_boxes[0]['pts_info']['timestamp']) / 1e9
                seq['sequence'] = seq['frame_idx']
                seq['size'] = b[3:6]
                seq['pose'] = rel_pose @ np.array(
                    [[np.cos(b[6]), -np.sin(b[6]), 0, b[0]],
                     [np.sin(b[6]), np.cos(b[6]), 0, b[1]],
                     [0, 0, 1, b[2]],
                     [0, 0, 0, 1]])

            yield 2 * i, traj_boxes
            traj_boxes = regularize(traj_boxes)
            yield 2 * i + 1, traj_boxes

    vis_iter = vis_iter_out(all_infos)

    def key_cbk(vis: o3d.visualization.Visualizer):
        try:
            idx, traj = next(vis_iter)
        except StopIteration:
            return True

        vis.clear_geometries()
        gt_box, gt_vel = renderbox(traj, color_amp=1.0)
        vis.add_geometry(gt_box, idx % 2 == 0)
        if gt_vel is not None:
            vis.add_geometry(gt_vel, idx % 2 == 0)

        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(ord(" "), key_cbk)
    vis.create_window(width=1080, height=720)
    op = vis.get_render_option()
    op.background_color = np.array([0., 0., 0.])
    op.point_size = 0.5
    if key_cbk(vis):
        return
    else:
        vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
