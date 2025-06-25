import argparse
import genesis as gs
import numpy as np


def run_sim(scene, camera):
    for i in range(120):
        scene.step()
        camera.set_pose(
            pos    = (3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
            lookat = (0.0, 0.0, 0.0),
        )
        camera.render()

    camera.stop_recording(save_to_filename='video.mp4', fps=60)
    scene.viewer.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f'{key}: {value}')

    # initialize Genesis
    if args.device == 'cpu':
        gs.init(backend=gs.cpu)
    elif args.device == 'cuda':
        gs.init(backend=gs.gpu)
    else:
        raise ValueError(f'Invalid device: {args.device}')

    # create scene
    scene = gs.Scene(
        show_viewer = True,
        viewer_options = gs.options.ViewerOptions(
            res           = (1280, 960),
            camera_pos    = (2.5, 2.5, 2.5),
            camera_lookat = (0.0, 0.0, 0.0),
            camera_fov    = 40,
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
        renderer=gs.renderers.Rasterizer(),
    )

    # add camera
    # GUI = False の場合は、GUI画面はこのカメラが反映したレンダリングではなく、camera.stop_recording で保存した動画にのみ反映される
    camera = scene.add_camera(
        res    = (1280, 960),
        pos    = (2.5, 2.5, 2.5),
        lookat = (0.0, 0.0, 0.0),
        fov    = 40,
        # Mac の場合は GUI を True にするとエラーが出るので、False にする
        GUI    = False,
    )

    # add objects
    plane = scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file='xml/franka_emika_panda/panda.xml',
            pos   = (0.0, 0.0, 0.0),
            euler = (0, 0, 0),
        ),
    )

    # build scene
    scene.build()

    # run simulation for mac (need to run in another thread in MacOS)
    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, camera))
    camera.start_recording()
    scene.viewer.start()
