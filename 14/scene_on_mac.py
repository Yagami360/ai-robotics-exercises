import argparse
import genesis as gs
import numpy as np


def run_sim(scene, camera, robot):
    for i in range(200):
        # ロボットの関節点位置を設定
        if i < 50:
            robot.set_dofs_position(np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]), dofs_idx)
        elif i < 100:
            robot.set_dofs_position(np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]), dofs_idx)
        else:
            robot.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), dofs_idx)

        scene.step()
        camera.set_pose(
            pos    = (3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
        )
        camera.render()

    camera.stop_recording(save_to_filename='video.mp4', fps=60)
    scene.viewer.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=tuple, default=(1280, 960))
    parser.add_argument('--camera_pos', type=tuple, default=(0, -3.5, 2.5))
    parser.add_argument('--camera_lookat', type=tuple, default=(0.0, 0.0, 0.5))
    parser.add_argument('--camera_fov', type=float, default=30)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f'{key}: {value}')

    # -------------------------------------
    # initialize Genesis
    # -------------------------------------
    if args.device == 'cpu':
        gs.init(backend=gs.cpu)
    elif args.device == 'cuda':
        gs.init(backend=gs.gpu)
    else:
        raise ValueError(f'Invalid device: {args.device}')

    # -------------------------------------
    # create scene
    # -------------------------------------
    scene = gs.Scene(
        show_viewer = True,
        viewer_options = gs.options.ViewerOptions(
            res           = args.resolution,
            camera_pos    = args.camera_pos,
            camera_lookat = args.camera_lookat,
            camera_fov    = args.camera_fov,
            max_FPS       = 60,
        ),
        vis_options = gs.options.VisOptions(
            show_world_frame = True,
            world_frame_size = 1.0,
            show_link_frame  = False,
            show_cameras     = False,
            plane_reflection = True,
            ambient_light    = (0.1, 0.1, 0.1),
        ),
        # シミュレーションの時間ステップ
        # 時間ステップを指定すると scene.step() で自動的に物理シミュレーション（重力など）が有効になる
        sim_options = gs.options.SimOptions(
            dt = 0.01,
        ),
        renderer=gs.renderers.Rasterizer(),
    )
    # print("scene: ", scene)
    # print("scene.viewer: ", scene.viewer)

    # -------------------------------------
    # add camera
    # -------------------------------------
    camera = scene.add_camera(
        res    = args.resolution,
        pos    = args.camera_pos,
        lookat = args.camera_lookat,
        fov    = args.camera_fov,
        GUI    = False,
    )

    # -------------------------------------
    # add objects (entities)
    # -------------------------------------
    plane = scene.add_entity(gs.morphs.Plane())

    # Add robot
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file='xml/franka_emika_panda/panda.xml',
            pos   = (0.0, 0.0, 0.0),
            euler = (0, 0, 0),
        ),
    )
    print("robot: ", robot)
    # robot:  ────────────────────── <gs.RigidEntity> ──────────────────────
    #                 'n_qs': <int>: 9
    #               'n_dofs': <int>: 9
    #              'n_links': <int>: 11
    #              'n_geoms': <int>: 23
    #              'n_cells': <numpy.int64>: 1023837
    #              'n_verts': <int>: 1363
    #              'n_faces': <int>: 2634
    #              'n_edges': <int>: 3951
    #             'n_joints': <int>: 11
    #             'n_vgeoms': <int>: 58
    #             'n_vverts': <int>: 397454
    #             'n_vfaces': <int>: 134888
    #                'q_end': <int>: 9
    #              'q_start': <int>: 0
    #             'is_built': <bool>: False
    #              'dof_end': <int>: 9
    #            'dof_start': <int>: 0
    #                  'idx': <int>: 1
    #                  'sim': <gs.Simulator>
    #                  'uid': <gs.UID>('f35162f-dcc1948e7836369278f6365e1')
    #            'base_link': <gs.RigidLink>: <bac9e9c>, name: 'link0', idx: 1
    #           'base_joint': <gs.RigidJoint>: <3581f7c>, name: 'link0_joint', idx: 1, type: <FIXED: 0>
    #        'base_link_idx': <int>: 1
    #             'cell_end': <numpy.int64>: 1029981
    #           'cell_start': <numpy.int64>: 6144
    #           'edge_start': <int>: 18
    #           'face_start': <int>: 12
    #             'geom_end': <int>: 24
    #           'geom_start': <int>: 1
    #            'init_qpos': <numpy.ndarray>: array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
    #             'link_end': <int>: 12
    #           'link_start': <int>: 1
    #           'vert_start': <int>: 8
    #                'geoms': <gs.List>(len=23, [
    #                             <gs.RigidGeom>: <e7b533f>, idx: 1 (from entity <f35162f>, link <bac9e9c>),
    #                             <gs.RigidGeom>: <95a120f>, idx: 2 (from entity <f35162f>, link <e805006>),
    #                             <gs.RigidGeom>: <2f552ac>, idx: 3 (from entity <f35162f>, link <4e50f0c>),
    #                             <gs.RigidGeom>: <6a3b5db>, idx: 4 (from entity <f35162f>, link <83a07a3>),
    #                             <gs.RigidGeom>: <82876f6>, idx: 5 (from entity <f35162f>, link <122f08c>),
    #                             <gs.RigidGeom>: <fb16926>, idx: 6 (from entity <f35162f>, link <d4b0e18>),
    #                             <gs.RigidGeom>: <238879b>, idx: 7 (from entity <f35162f>, link <d4b0e18>),
    #                             <gs.RigidGeom>: <090dc51>, idx: 8 (from entity <f35162f>, link <d4b0e18>),
    #                             <gs.RigidGeom>: <a10641d>, idx: 9 (from entity <f35162f>, link <d48fca7>),
    #                             ...
    #                             <gs.RigidGeom>: <19e384e>, idx: 23 (from entity <f35162f>, link <3602164>),
    #                         ])
    #                'links': <gs.List>(len=11, [
    #                             <gs.RigidLink>: <bac9e9c>, name: 'link0', idx: 1,
    #                             <gs.RigidLink>: <e805006>, name: 'link1', idx: 2,
    #                             <gs.RigidLink>: <4e50f0c>, name: 'link2', idx: 3,
    #                             <gs.RigidLink>: <83a07a3>, name: 'link3', idx: 4,
    #                             <gs.RigidLink>: <122f08c>, name: 'link4', idx: 5,
    #                             <gs.RigidLink>: <d4b0e18>, name: 'link5', idx: 6,
    #                             <gs.RigidLink>: <d48fca7>, name: 'link6', idx: 7,
    #                             <gs.RigidLink>: <3cc4205>, name: 'link7', idx: 8,
    #                             <gs.RigidLink>: <266467d>, name: 'hand', idx: 9,
    #                             <gs.RigidLink>: <2a7e0ac>, name: 'left_finger', idx: 10,
    #                             <gs.RigidLink>: <3602164>, name: 'right_finger', idx: 11,
    #                         ])
    #                'morph': <gs.morphs.MJCF(file='/opt/anaconda3/envs/genenis/lib/python3.10/site-packages/genesis/assets/xml/franka_emika_panda/panda.xml')>
    #                'scene': <gs.Scene>
    #          'vface_start': <int>: 12
    #          'vvert_start': <int>: 8
    #               'joints': <gs.List>(len=11, [
    #                             <gs.RigidJoint>: <3581f7c>, name: 'link0_joint', idx: 1, type: <FIXED: 0>,
    #                             <gs.RigidJoint>: <0c7d374>, name: 'joint1', idx: 2, type: <REVOLUTE: 1>,
    #                             <gs.RigidJoint>: <0ea5f33>, name: 'joint2', idx: 3, type: <REVOLUTE: 1>,
    #                             <gs.RigidJoint>: <63b634e>, name: 'joint3', idx: 4, type: <REVOLUTE: 1>,
    #                             <gs.RigidJoint>: <9705cf4>, name: 'joint4', idx: 5, type: <REVOLUTE: 1>,
    #                             <gs.RigidJoint>: <3c6f2a3>, name: 'joint5', idx: 6, type: <REVOLUTE: 1>,
    #                             <gs.RigidJoint>: <b0fd515>, name: 'joint6', idx: 7, type: <REVOLUTE: 1>,
    #                             <gs.RigidJoint>: <d1f7685>, name: 'joint7', idx: 8, type: <REVOLUTE: 1>,
    #                             <gs.RigidJoint>: <d85b780>, name: 'hand_joint', idx: 9, type: <FIXED: 0>,
    #                             <gs.RigidJoint>: <cd6e319>, name: 'finger_joint1', idx: 10, type: <PRISMATIC: 2>,
    #                             <gs.RigidJoint>: <3f47d8f>, name: 'finger_joint2', idx: 11, type: <PRISMATIC: 2>,
    #                         ])
    #               'solver': <gs.RigidSolver>: <f5900fd>, n_entities: 2
    #               'vgeoms': <gs.List>(len=58, [
    #                             <gs.RigidVisGeom>: <18c1bc5>, idx: 1 (from entity <f35162f>, link <bac9e9c>),
    #                             <gs.RigidVisGeom>: <5956043>, idx: 2 (from entity <f35162f>, link <bac9e9c>),
    #                             <gs.RigidVisGeom>: <1e61205>, idx: 3 (from entity <f35162f>, link <bac9e9c>),
    #                             <gs.RigidVisGeom>: <41a6d89>, idx: 4 (from entity <f35162f>, link <bac9e9c>),
    #                             <gs.RigidVisGeom>: <5c91e62>, idx: 5 (from entity <f35162f>, link <bac9e9c>),
    #                             <gs.RigidVisGeom>: <e5dd2e6>, idx: 6 (from entity <f35162f>, link <bac9e9c>),
    #                             <gs.RigidVisGeom>: <f0cb1b6>, idx: 7 (from entity <f35162f>, link <bac9e9c>),
    #                             <gs.RigidVisGeom>: <7bd1e41>, idx: 8 (from entity <f35162f>, link <bac9e9c>),
    #                             <gs.RigidVisGeom>: <5541656>, idx: 9 (from entity <f35162f>, link <bac9e9c>),
    #                             ...
    #                             <gs.RigidVisGeom>: <4f0fd12>, idx: 58 (from entity <f35162f>, link <3602164>),
    #                         ])
    # 'gravity_compensation': <float>: 0.0
    #              'surface': <gs.options.surfaces.Default>
    #             'material': <gs.materials.Rigid>
    #    'visualize_contact': <bool>: False

    # -------------------------------------
    # build scene
    # -------------------------------------
    scene.build()

    # -------------------------------------
    # set robot joint control parameters
    # -------------------------------------
    # ロボットの関節点のインデックスを取得（正確には、関節点の自由度インデックスを取得）
    # ここでいう自由度（Degrees of Freedom, DOF）とは、ロボットで独立して動くことができる方向や軸の数のこと
    joint_names = [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
        'joint7',
        'finger_joint1',
        'finger_joint2',
    ]
    dofs_idx = [robot.get_joint(name).dof_idx_local for name in joint_names]
    print("dofs_idx: ", dofs_idx)

    # 位置ゲインの設定
    # 位置ゲイン：ロボットの位置制御において、目標位置と現在位置の差（位置誤差）をどれだけ強く補正するかを決めるパラメータ
    robot.set_dofs_kp(
        kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        dofs_idx_local = dofs_idx,
    )

    # 速度ゲインの設定
    # 速度ゲイン：ロボットの速度制御において、目標速度と現在速度の差（速度誤差）をどれだけ強く補正するかを決めるパラメータ
    robot.set_dofs_kv(
        kv             = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        dofs_idx_local = dofs_idx,
    )

    # 安全のための力の範囲設定
    robot.set_dofs_force_range(
        lower          = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        upper          = np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
        dofs_idx_local = dofs_idx,
    )

    # -------------------------------------
    # run simulation for mac (need to run in another thread in MacOS)
    # -------------------------------------
    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, camera, robot))
    camera.start_recording()
    scene.viewer.start()
