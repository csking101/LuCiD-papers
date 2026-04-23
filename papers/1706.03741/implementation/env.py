# Grid Environment
#
# A cat in a grid world. Gymnasium-style API: reset(), step(), render().
#
# State: cat position (row, col) on an NxN grid
# Actions: 0=up, 1=down, 2=left, 3=right
# Goals: one or more target cells the cat should reach
# Obstacles: optional blocked cells
#
# There is NO built-in reward function — the reward comes from the
# learned reward model (reward_model.py). For testing, you can use
# a synthetic oracle in preferences.py that knows the "true" reward.
from enum import Enum

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Environment:

    def __init__(self,height:int ,width: int,obstacles_placement:dict,rewards_placement:dict):
        self.height = height
        self.width = width

        # 
        self.obstacles_placement = obstacles_placement
        self.rewards_placement = rewards_placement

        self.actions = {Action.UP:(0,-1),Action.DOWN:(0,1),Action.LEFT:(-1,0),Action.RIGHT:(1,0)} #Up, down, left, right

        self.position = [0,0]

    def take_action(action:Action) -> str:
        up_constraint = action == Action.UP and self.position[1] == 0
        down_constraint = action == Action.DOWN and self.position[1] == height-1






