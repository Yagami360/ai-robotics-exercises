import genesis as gs

# Initialize Genesis
gs.init(backend=gs.cpu)

# create scene
scene = gs.Scene(
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        # for mac but only supported latest genesis version
        # run_in_thread=False,
        max_FPS=60,
    ),
)

# add objects
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

# build scene
scene.build()

# run simulation
# scene.viewer.start()

for i in range(100):
    scene.step()

# scene.viewer.stop()
