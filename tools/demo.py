import copy
import math
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import matplotlib.pyplot as plt
from motreg import utils, MotionModel
from deploy3d.misc.loading import MONGODBBackend


class RegularTrajectory:
    def __init__(self, data_iter):
        max_length = 200
        self.data_iter = data_iter

        self.show_vel = True
        self.weightMotion = 1.0
        self.weightMotionConsistency = 1.0
        self.weightObjPose2Label = 1.0

        self.selected = []
        self.regular_ret_list = []
        self.regular_vel_vectors_list = []

        self.window = gui.Application.instance.create_window(
            'Open3D Relabel Bus', 1920, 1080)
        em = self.window.theme.font_size
        separation_height = int(2 * em)

        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        self._scene.scene.set_background([1, 0, 0, 0])

        self.material = rendering.MaterialRecord()
        self.material.shader = 'defaultUnlit'
        self.material.point_size = 3

        self._settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em,
                                                       0.25 * em, 0.25 * em))

        tree = gui.TreeView()
        tree.set_on_selection_changed(self._on_tree)
        cb_list = []
        for name in range(max_length):
            cb = gui.Checkbox(str(name))
            cb.set_on_checked(self._on_cb)
            tree.add_item(tree.get_root_item(), cb)
            cb_list.append(cb)
        self.cb_list = cb_list

        switch_to_show_vel = gui.ToggleSwitch("Show Vel")
        switch_to_show_vel.set_on_clicked(self._on_switch_to_show_vel)
        switch_to_show_vel.is_on = True
        self.switch_to_show_vel = switch_to_show_vel

        switch_layout = gui.Horiz()
        switch_to_hide_origin = gui.ToggleSwitch("Hide Original")
        switch_to_hide_origin.set_on_clicked(self._on_switch_to_hide_origin)
        self.switch_to_hide_origin = switch_to_hide_origin

        switch_to_hide_regular = gui.ToggleSwitch("Hide Regular")
        switch_to_hide_regular.set_on_clicked(self._on_switch_to_hide_regular)
        self.switch_to_hide_regular = switch_to_hide_regular

        switch_layout.add_child(switch_to_hide_origin)
        switch_layout.add_fixed(int(em))
        switch_layout.add_child(switch_to_hide_regular)

        vis_layout = gui.Horiz()
        reverse_button = gui.Button("Reverse Selected")
        reverse_button.horizontal_padding_em = 0.0
        reverse_button.vertical_padding_em = 1.0
        reverse_button.set_on_clicked(self._reverse_selected)

        clear_button = gui.Button("Clear Selected")
        clear_button.horizontal_padding_em = 1.0
        clear_button.vertical_padding_em = 1.0
        clear_button.set_on_clicked(self._clear_selected)
        vis_layout.add_child(reverse_button)
        vis_layout.add_fixed(int(em))
        vis_layout.add_child(clear_button)

        motion_layout = gui.Vert()
        weightMotion = gui.Slider(gui.Slider.DOUBLE)
        weightMotion.set_limits(-10, 10)
        weightMotion.double_value = 1.0
        weightMotion.set_on_value_changed(self._on_slider_motion)
        self.weightMotion_s = weightMotion

        tedit = gui.TextEdit()
        tedit.placeholder_text = "Edit me some text here"
        tedit.set_on_text_changed(self._on_text_changed_motion)
        motion_layout.add_child(weightMotion)
        motion_layout.add_fixed(int(0.5 * em))
        motion_layout.add_child(tedit)

        motionconsis_layout = gui.Vert()
        weightMotionConsistency = gui.Slider(gui.Slider.DOUBLE)
        weightMotionConsistency.set_limits(-10, 10)
        weightMotionConsistency.double_value = 1.0
        weightMotionConsistency.set_on_value_changed(
            self._on_slider_motion_consis)
        self.weightMotionConsistency_s = weightMotionConsistency

        tedit = gui.TextEdit()
        tedit.placeholder_text = "Edit me some text here"
        tedit.set_on_text_changed(self._on_text_changed_motion_consis)
        motionconsis_layout.add_child(weightMotionConsistency)
        motionconsis_layout.add_fixed(int(0.5 * em))
        motionconsis_layout.add_child(tedit)

        objpose2label_layout = gui.Vert()
        weightObjPose2Label = gui.Slider(gui.Slider.DOUBLE)
        weightObjPose2Label.set_limits(-10, 10)
        weightObjPose2Label.double_value = 1.0
        weightObjPose2Label.set_on_value_changed(self._on_slider_objpose2label)
        self.weightObjPose2Label_s = weightObjPose2Label

        tedit = gui.TextEdit()
        tedit.placeholder_text = "Edit me some text here"
        tedit.set_on_text_changed(self._on_text_changed_objpose2label)
        objpose2label_layout.add_child(weightObjPose2Label)
        objpose2label_layout.add_fixed(int(0.5 * em))
        objpose2label_layout.add_child(tedit)

        do_button = gui.Button("Regularize")
        do_button.horizontal_padding_em = 1.0
        do_button.vertical_padding_em = 1.0
        do_button.set_on_clicked(self._regularize)

        next_button = gui.Button("Next Seq")
        next_button.horizontal_padding_em = 1.0
        next_button.vertical_padding_em = 1.0
        next_button.set_on_clicked(self._next_button)

        self._settings_panel.add_child(gui.Label('Select Objs'))
        self._settings_panel.add_child(tree)
        self._settings_panel.add_fixed(separation_height)

        self._settings_panel.add_child(switch_to_show_vel)
        self._settings_panel.add_fixed(separation_height)

        self._settings_panel.add_child(switch_layout)
        self._settings_panel.add_fixed(separation_height)

        self._settings_panel.add_child(vis_layout)
        self._settings_panel.add_fixed(separation_height)

        self._settings_panel.add_child(gui.Label('weightMotion'))
        self._settings_panel.add_child(motion_layout)
        self._settings_panel.add_fixed(separation_height)

        self._settings_panel.add_child(gui.Label('weightMotionConsistency'))
        self._settings_panel.add_child(motionconsis_layout)
        self._settings_panel.add_fixed(separation_height)

        self._settings_panel.add_child(gui.Label('weightObjPose2Label'))
        self._settings_panel.add_child(objpose2label_layout)
        self._settings_panel.add_fixed(separation_height)

        self._settings_panel.add_child(do_button)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(next_button)
        self._settings_panel.add_fixed(separation_height)

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._scene)
        self.window.add_child(self._settings_panel)

        self._next_seq()

    def _on_cb(self, show_label):
        if self.ret_list and self.cur_id < len(self.ret_list):
            geo_name = f'original_box_{self.cur_id}'
            box = self.ret_list[self.cur_id]
            if not self._scene.scene.has_geometry(geo_name):
                self._scene.scene.add_geometry(geo_name, box, self.material)
            self._scene.scene.show_geometry(geo_name, show_label)

            if self.regular_ret_list:
                geo_name = f'regularized_box_{self.cur_id}'
                box = self.regular_ret_list[self.cur_id]
                if not self._scene.scene.has_geometry(geo_name):
                    self._scene.scene.add_geometry(
                        geo_name, box, self.material)
                self._scene.scene.show_geometry(geo_name, show_label)

        if show_label:
            self.selected.append(self.cur_id)
        else:
            self.selected.remove(self.cur_id)

    def _on_tree(self, new_item_id):
        self.cur_id = new_item_id - 1

    def _on_switch_to_show_vel(self, is_on):
        if self._scene.scene.has_geometry('all vel'):
            self._scene.scene.show_geometry('all vel', is_on)

        if self._scene.scene.has_geometry('all regularized vel'):
            self._scene.scene.show_geometry('all regularized vel', is_on)

        self.show_vel = is_on

    def _on_switch_to_hide_origin(self, is_on):
        is_show = not is_on
        if self._scene.scene.has_geometry('all box'):
            self._scene.scene.show_geometry('all box', is_show)

        is_show = False
        if (self.show_vel and not is_on):
            is_show = True

        if self._scene.scene.has_geometry('all vel'):
            self._scene.scene.show_geometry('all vel', is_show)

    def _on_switch_to_hide_regular(self, is_on):
        is_show = not is_on
        if self._scene.scene.has_geometry('all regularized box'):
            self._scene.scene.show_geometry('all regularized box', is_show)

        is_show = False
        if (self.show_vel and not is_on):
            is_show = True

        if self._scene.scene.has_geometry('all regularized vel'):
            self._scene.scene.show_geometry(
                'all regularized vel', is_show)

    def _reverse_selected(self):
        reversed_selected = []
        for idx, box in enumerate(self.ret_list):
            geo_name = f'original_box_{idx}'
            if idx in self.selected:
                self.cb_list[idx].checked = False
                if self._scene.scene.has_geometry(geo_name):
                    self._scene.scene.show_geometry(geo_name, False)
            else:
                self.cb_list[idx].checked = True
                if not self._scene.scene.has_geometry(geo_name):
                    self._scene.scene.add_geometry(
                        geo_name, box, self.material)
                self._scene.scene.show_geometry(geo_name, True)
                reversed_selected.append(idx)

            if self.regular_ret_list:
                geo_name = f'regularized_box_{idx}'
                box = self.regular_ret_list[idx]
                if idx in self.selected:
                    if self._scene.scene.has_geometry(geo_name):
                        self._scene.scene.show_geometry(geo_name, False)
                else:
                    if not self._scene.scene.has_geometry(geo_name):
                        self._scene.scene.add_geometry(
                            geo_name, box, self.material)
                    self._scene.scene.show_geometry(geo_name, True)

        self.selected = reversed_selected

    def _clear_selected(self):
        for idx in self.selected:
            geo_name = f'original_box_{idx}'
            self.cb_list[idx].checked = False
            if self._scene.scene.has_geometry(geo_name):
                self._scene.scene.show_geometry(geo_name, False)
        
        for idx in range(len(self.ret_list)):
            geo_name = f'regularized_box_{idx}'
            if self._scene.scene.has_geometry(geo_name):
                self._scene.scene.remove_geometry(geo_name)

        self.selected.clear()

    def _on_slider_motion(self, new_val):
        self.weightMotion = math.pow(10, new_val)
        self._regularize()

    def _on_text_changed_motion(self, new_text):
        self.weightMotion = math.pow(10, eval(new_text))
        self._regularize()

    def _on_slider_motion_consis(self, new_val):
        self.weightMotionConsistency = math.pow(10, new_val)
        self._regularize()

    def _on_text_changed_motion_consis(self, new_text):
        self.weightMotionConsistency = math.pow(10, eval(new_text))
        self._regularize()

    def _on_slider_objpose2label(self, new_val):
        self.weightObjPose2Label = math.pow(10, new_val)
        self._regularize()

    def _on_text_changed_objpose2label(self, new_text):
        self.weightObjPose2Label = math.pow(10, eval(new_text))
        self._regularize()

    def _regularize(self):
        if self._scene.scene.has_geometry('all regularized box'):
            self._scene.scene.remove_geometry('all regularized box')

        if self._scene.scene.has_geometry('all regularized vel'):
            self._scene.scene.remove_geometry('all regularized vel')

        self._clear_selected()
        self.regular_ret_list.clear()
        self.regular_vel_vectors_list.clear()

        traj_boxes = copy.deepcopy(self.traj_boxes)
        input = [utils.ObjBBox(_) for _ in traj_boxes]
        model_params = utils.MotionModelParams()
        model_params.verbose = True
        model_params.weightMotion = np.diag(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * self.weightMotion
        model_params.weightMotionConsistency = np.diag(
            [1.0, 1.0]) * self.weightMotionConsistency
        model_params.weightObjPose2Label = np.diag(
            [0.2, 0.2, 0.2, 0.5, 0.5, 0.2]) * self.weightObjPose2Label
        model = MotionModel(input, model_params)
        outputs = [_.toDict() for _ in model.output(
            [_['sequence'] for _ in traj_boxes])]
        for obj, output in zip(traj_boxes, outputs):
            obj.update(output)

        gt_box, gt_vel, ret_list, vel_vectors_list = RegularTrajectory.renderbox(
            traj_boxes, color_amp=1.0)

        self._scene.scene.add_geometry(
            "all regularized box", gt_box, self.material)

        if gt_vel:
            self._scene.scene.add_geometry(
                "all regularized vel", gt_vel, self.material)
            self._scene.scene.show_geometry(
                'all regularized vel', self.show_vel)

        self.regular_ret_list = ret_list
        self.regular_vel_vectors_list = vel_vectors_list
        assert len(self.regular_ret_list) == len(self.ret_list)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(r.height, self._settings_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(
            r.get_right() - width, r.y, width, height)

    def _next_button(self):
        self.weightMotion = 1.0
        self.weightMotionConsistency = 1.0
        self.weightObjPose2Label = 1.0
        self.weightMotion_s.double_value = 1.0
        self.weightMotionConsistency_s.double_value = 1.0
        self.weightObjPose2Label_s.double_value = 1.0

        self.switch_to_hide_origin.is_on = False
        self.switch_to_hide_regular.is_on = False

        self.show_vel = True
        self.switch_to_show_vel.is_on = True

        self._clear_selected()
        self.regular_ret_list.clear()
        self.regular_vel_vectors_list.clear()
        self._next_seq()

        self.window.post_redraw()

    def _next_seq(self):
        try:
            traj = next(self.data_iter)
        except StopIteration:
            return

        gt_box, gt_vel, ret_list, vel_vectors_list = RegularTrajectory.renderbox(
            traj, color_amp=1.0)
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry("all box", gt_box, self.material)
        if gt_vel:
            self._scene.scene.add_geometry("all vel", gt_vel, self.material)

        bounds = o3d.geometry.AxisAlignedBoundingBox(
            (-20, -20, -2), (20, 20, 10))
        self._scene.setup_camera(60, bounds, bounds.get_center())

        self.traj_boxes = traj
        self.ret_list = ret_list
        self.vel_vectors_list = vel_vectors_list

    @staticmethod
    def box3d_corners(dims):
        if dims.shape[0] == 0:
            return np.empty([0, 8, 3], dtype=dims.dtype)
        corners_norm = np.stack(np.unravel_index(
            np.arange(8), [2] * 3), axis=1)
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        corners_norm = corners_norm - np.array([0.5, 0.5, 0], dtype=dims.dtype)
        corners = dims.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
        return corners

    @staticmethod
    def renderbox(trajectories, color_amp=1.0):
        clr_map = plt.get_cmap('tab10').colors
        corners = RegularTrajectory.box3d_corners(
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
        rets, vel_vectors = None, None
        rets_list, vel_vectors_list = [], []
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

                    vel_vectors_list.append(copy.deepcopy(vel_vector))
                    if vel_vectors is None:
                        vel_vectors = vel_vector
                    else:
                        vel_vectors += vel_vector

            box = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
                corners_i, corners_i, cores)

            error = trajectories[i].get('errLabel', None)
            if error is None or error.max() < 0.025:
                color = [color_amp *
                         c for c in clr_map[label_i % len(clr_map)]]
            else:
                color = [1., 1., 1.]
            box.paint_uniform_color(color)

            box.rotate(bpose_i[:3, :3], center=(0, 0, 0))
            box.translate(bpose_i[:3, 3])

            rets_list.append(copy.deepcopy(box))
            if rets is None:
                rets = box
            else:
                rets += box
        return rets, vel_vectors, rets_list, vel_vectors_list


def main():
    mongo_cfg = dict(
        database='ai-cowa3d-u1-v56-reference',
        host='mongodb://perception:nJLSj65JOy@172.16.100.107,172.16.100.108,172.16.100.109/?tls=false')
    info_client = MONGODBBackend(**mongo_cfg, pbar=True)

    all_infos = info_client.query(
        'trajectories/labels',
        filter={'trajectory.0.gt_names_3d': {'$in': [
            'cyclist',
            'vehicle',
            'big_vehicle',
        ]}, '$expr': {"$gt": [{"$size": '$trajectory'}, 50]}},
        projection=['_id', 'trajectory', 'context',
                    'base_pose', 'pts'],
        limit=100,
        cursor_type='NON_TAILABLE')
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

            yield traj_boxes

    data_iter = vis_iter_out(all_infos)

    gui.Application.instance.initialize()
    w = RegularTrajectory(data_iter)
    gui.Application.instance.run()


if __name__ == '__main__':
    main()
