import sys
from transforms3d.euler import quat2euler, euler2quat
import time
import numpy as np
from typing import List, Tuple, Dict, Any
from enum import IntEnum, Enum
import threading
from multiprocessing import Process
from multiprocessing.managers import BaseManager

sys.path.insert(0, "/root/irl_control/libraries/psmoveapi/build")
import psmove

class Dim(IntEnum):
    X = 0
    Y = 1
    Z = 2

class MoveName(Enum):
    RIGHT = 0
    LEFT = 1

class DimRange():
    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

class DimRanges():
    def __init__(self, sim: DimRange, move: DimRange):
        self.sim = sim
        self.move = move

class MoveState():
    def __init__(self):
        self.values: Dict[str, Any] = dict()
        self.values['pos']: np.ndarray = np.zeros(3)
        self.values['quat']: np.ndarray = np.zeros(4)
        self.values['rumble']: int = 0
        self.values['trigger']: bool = False
        self.values['square']: bool = False
        self.values['triangle']: bool = False
        self.values['circle']: bool = False
    
    def get(self, key: str):
        return self.values[key]
    
    def set(self, key: str, value: Any):
        self.values[key] = value


class PSMoveInterface():
    def __init__(self, multiprocess: bool = False):
        if psmove.count_connected() < 1:
            print('No controller connected')
            sys.exit(1)
        
        move_count = psmove.count_connected()
        print('Connected controllers:', move_count)

        if multiprocess:
            BaseManager.register('MoveState', MoveState)
            manager = BaseManager()
            manager.start()

        self.tracker = psmove.PSMoveTracker()
        self.tracker.set_mirror(True)
        self.moves = dict()
        self.move_workers = []
        self.move_states = dict()
        self.running = True
        for idx in range(move_count):
            move_name = self.serial2name(psmove.PSMove(idx).get_serial())
            self.moves[move_name] = psmove.PSMove(idx)
            if self.moves[move_name].connection_type != psmove.Conn_Bluetooth:
                print('Please connect controller via Bluetooth')
                sys.exit(1)
            self.moves[move_name].enable_orientation(True)
            move_dim_ranges = self.get_dim_ranges(move_name)
            if multiprocess:
                self.move_states[move_name] = manager.MoveState()
                self.move_workers.append(Process(target=self.collect_move_state,
                    args=(self.move_states[move_name], self.moves[move_name], move_dim_ranges, self.tracker, self.running)))
            else:
                self.move_states[move_name] = MoveState()
                self.move_workers.append(threading.Thread(target=self.collect_move_state,
                    args=(self.move_states[move_name], self.moves[move_name], move_dim_ranges, self.tracker, self.running)))
            
            # Calibrate the controller with the tracker
            result = -1
            while result != psmove.Tracker_CALIBRATED:
                print('Trying to calibrate... move controller.')
                result = self.tracker.enable(self.moves[move_name])
        
        for idx in range(move_count):
            self.move_workers[idx].start()
    
    def get_dim_ranges(self, move_name: MoveName):
        if move_name == MoveName.LEFT:
            dim_ranges_dict = {
                Dim.X : DimRanges(sim=DimRange(0.2, -0.7), move=DimRange(375, 600)),
                Dim.Y : DimRanges(sim=DimRange(0.9, 0.0), move=DimRange(12, 70)),
                Dim.Z : DimRanges(sim=DimRange(0.01, 0.5), move=DimRange(-400, -20)),
            }
        elif move_name == MoveName.RIGHT:
            dim_ranges_dict = {
                Dim.X : DimRanges(sim=DimRange(0.7, -0.2), move=DimRange(150, 375)),
                Dim.Y : DimRanges(sim=DimRange(0.9, 0.0), move=DimRange(12, 70)),
                Dim.Z : DimRanges(sim=DimRange(0.01, 0.5), move=DimRange(-400, -20)),
            }
        else:
            print("Move Name is not valid!")
            raise ValueError
        
        return dim_ranges_dict

    def serial2name(self, serial) -> MoveName:
        if serial == "00:13:8a:91:f9:7e":
            move_name = MoveName.RIGHT
        elif serial == "e0:ae:5e:3e:10:24":
            move_name = MoveName.LEFT
        else:
            print("Serial " + serial + " not defined!")
            sys.exit(1)
        return move_name
    
    def collect_move_state(self, move_state, move, move_dim_ranges, tracker, running):
        tracker_positions = {
            Dim.X : 0.0,
            Dim.Y : 0.0,
            Dim.Z : 0.0
        }
       
        while running:
            while move.poll(): pass
            tracker.update_image()
            tracker.update()
            status = tracker.get_status(move)

            trigger_value = move.get_trigger()
            if trigger_value > 10:
                move_state.set('trigger', True)
            else:
                move_state.set('trigger', False)
            
            if status == psmove.Tracker_TRACKING:
                x, y, radius = tracker.get_position(move)
                tracker_positions[Dim.X] = x
                # radius is the forward/backward axis for tracker, but y is forward/backward in sim 
                tracker_positions[Dim.Y] = radius
                #y is the up/down axis for tracker, but z is up/down in sim
                tracker_positions[Dim.Z] = -1*y

            buttons = move.get_buttons()
            if buttons & psmove.Btn_SQUARE:
               move.reset_orientation()
            
            if buttons & psmove.Btn_TRIANGLE:
                move_state.set('triangle', True)
            else:
                move_state.set('triangle', False)
            
            if buttons & psmove.Btn_CIRCLE:
                move_state.set('circle', True)
            else:
                move_state.set('circle', False)
            
            move_quat = move.get_orientation()
            eul = quat2euler(move_quat)
            new_eul = [eul[0], 0, eul[1]]
            move_quat_new = euler2quat(*new_eul)
            move_state.set('quat', move_quat_new)
            
            if status == psmove.Tracker_TRACKING:
                pos_arr = np.zeros(3)
                for dim in Dim:
                    pos = tracker_positions[dim]
                    mr = move_dim_ranges[dim].move # move range
                    sr = move_dim_ranges[dim].sim # sim range
                    if pos < mr.min:
                       pos = mr.min
                    if pos > mr.max:
                       pos = mr.max
                    val = sr.min + ((pos - mr.min)/(mr.max - mr.min) * (sr.max - sr.min))
                    pos_arr[dim] = val
                move_state.set('pos', pos_arr)
            
            rv = -1*move_state.get('rumble')
            min_rumble_val = 0.1
            max_rumble_val = 0.7
            val = (rv - min_rumble_val)/(max_rumble_val - min_rumble_val)*130
            val = min(max(0, val), 130)
            move.set_rumble(int(val))
            time.sleep(0.05)
    
    def stop(self):
        self.running = False