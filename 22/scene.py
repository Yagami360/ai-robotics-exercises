import mujoco
from mujoco.viewer import launch

# load object
model = mujoco.MjModel.from_xml_path('../mujoco/model/humanoid/humanoid.xml')
data = mujoco.MjData(model)

# Run simulater
launch(model, data)
