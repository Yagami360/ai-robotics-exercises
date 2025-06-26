import argparse
import genesis as gs
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def run_sim(scene, camera, robot):
    for i in range(200):
        # ロボットの関節点位置を設定
        if i < 50:
            robot.set_dofs_position(np.array([0, -1.57, 1.57, -1.57, -1.57, 0]), dofs_idx)
        elif i < 100:
            robot.set_dofs_position(np.array([1.57, -0.8, 1.0, -2.0, 1.0, 0.5]), dofs_idx)
        else:
            robot.set_dofs_position(np.array([0, -1.57, 1.57, -1.57, -1.57, 0]), dofs_idx)

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

    # Add robot (UR5e)
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file='xml/universal_robots_ur5e/ur5e.xml',
            pos   = (0.0, 0.0, 0.0),
            euler = (0, 0, 0),
        ),
    )
    print("robot: ", robot)

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
        # 'base_joint', # 固定ジョイントのため除外
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint',
        # 'ee_virtual_link_joint', # 固定ジョイントのため除外
    ]
    dofs_idx = [robot.get_joint(name).dof_idx_local for name in joint_names]
    print("dofs_idx: ", dofs_idx)

    # 位置ゲインの設定
    # 位置ゲイン：ロボットの位置制御において、目標位置と現在位置の差（位置誤差）をどれだけ強く補正するかを決めるパラメータ
    robot.set_dofs_kp(
        kp             = np.array([2000, 2000, 2000, 500, 500, 500]),
        dofs_idx_local = dofs_idx,
    )

    # 速度ゲインの設定
    # 速度ゲイン：ロボットの速度制御において、目標速度と現在速度の差（速度誤差）をどれだけ強く補正するかを決めるパラメータ
    robot.set_dofs_kv(
        kv             = np.array([100, 100, 100, 25, 25, 25]),
        dofs_idx_local = dofs_idx,
    )

    # 安全のための力の範囲設定
    robot.set_dofs_force_range(
        lower          = np.array([-150, -150, -150, -28, -28, -28]),
        upper          = np.array([ 150,  150,  150,  28,  28,  28]),
        dofs_idx_local = dofs_idx,
    )

    # -------------------------------------
    # run simulation for mac (need to run in another thread in MacOS)
    # -------------------------------------
    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, camera, robot))
    camera.start_recording()
    scene.viewer.start()
