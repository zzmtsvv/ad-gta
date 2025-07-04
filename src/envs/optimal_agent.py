import numpy as np


class OptimalAgentK2D:
    '''
        Optimal Agent class for KeyToDoor environment
    '''
    def act(self, state, env):
        if env.unwrapped.has_key:
            goal = env.unwrapped.door_pos
        else:
            goal = env.unwrapped.key_pos

        if np.all(goal == state):
            return 0

        # first up or down
        if goal[0] > state[0]:
            return 3
        if goal[0] < state[0]:
            return 1

        # then left or right
        if goal[1] > state[1]:
            return 2
        if goal[1] < state[1]:
            return 4


class OptimalAgent:
    '''
        Optimal Agent class for DarkRoom environment
    '''
    def act(self, state, env):
        goal = env.unwrapped.goal_pos
        if np.all(goal == state):
            return 0

        # first up or down
        if goal[0] > state[0]:
            return 3
        if goal[0] < state[0]:
            return 1

        # then left or right
        if goal[1] > state[1]:
            return 2
        if goal[1] < state[1]:
            return 4
