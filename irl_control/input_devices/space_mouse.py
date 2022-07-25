import pyspacemouse
import numpy as np
import types

class SpaceMouse():
    def __init__(self, origin, increment=0.0015):
        success = pyspacemouse.open()
        if not success:
            print("Space Mouse not found!")
            raise Exception
        self.increment = increment
        self.state = types.SimpleNamespace()
        self.state.x = origin[0]
        self.state.y = origin[1]
        self.state.z = origin[2]
        self.state.roll = origin[3]
        self.state.pitch = origin[4]
        self.state.yaw = origin[5]
    
    def constrain_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def update_state(self):
        inc = self.increment
        cur_state = pyspacemouse.read()
        self.state.x += inc*cur_state.x
        self.state.y += inc*cur_state.y
        self.state.z += inc*cur_state.z
        self.state.yaw -= inc*cur_state.yaw
        self.state.yaw = self.constrain_angle(self.state.yaw)
        self.state.pitch -= inc*cur_state.pitch
        self.state.pitch = self.constrain_angle(self.state.pitch)
        self.state.roll += inc*cur_state.roll
        self.state.roll = self.constrain_angle(self.state.roll)
        return (self.state.x, self.state.y, self.state.z, self.state.roll, self.state.pitch, self.state.yaw)
