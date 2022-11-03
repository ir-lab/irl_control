from dual_insertion import DualInsertion

class CollectData(DualInsertion):
    """
    Implements the OSC and Dual UR5 robot
    """
    def __init__(self, robot_config_file : str =None, scene_file : str = None):
        # Initialize the Parent class with the config file
        action_config_name = 'iros2022_task.yaml'
        self.action_config = self.get_action_config(action_config_name)
        super().__init__(self.action_config["device_config"], robot_config_file, scene_file)
           
    def run(self):        
        action_object_names = ['iros2022_action_objects']
        self.action_objects = self.action_config[action_object_names[0]]
        self.initialize_action_objects()
        
        self.run_sequence(self.action_config['iros2022_pickup_sequence'])

        self.set_record(True)
        self.run_sequence(self.action_config['iros2022_demo_sequence'])

        self.set_record(False)
        self.run_sequence(self.action_config['iros2022_release_sequence'])


# Main entrypoint
if __name__ == "__main__":
    demo = CollectData(robot_config_file="iros2022.yaml", scene_file="iros2022.xml")
    demo.run()