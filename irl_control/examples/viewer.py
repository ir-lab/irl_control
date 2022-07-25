from mujoco_py import load_model_from_path, MjSim
from mujoco_py.mjviewer import MjViewer

class Viewer():
    def __init__(self, xml_file):
        # Initialize the Parent class with the config file
        self.model = load_model_from_path(xml_file)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)

    def run(self):
        while True:
            self.sim.step()
            self.viewer.render()

if __name__ == "__main__":
    viewer = Viewer("/root/irl_control/src/scenes/main_dual_ur5.xml")
    viewer.run()