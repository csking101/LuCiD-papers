# Grid Environment
#
# A cat in a grid world. Gymnasium-style API: reset(), step().
#
# State: cat position (x, y) on a W x H grid
# Actions: 0=up, 1=down, 2=left, 3=right
# Goals: one or more terminal cells with positive/negative values
# Obstacles: optional blocked cells (walls)
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


# Movement deltas: (dx, dy) for each action
DELTAS = {
    Action.UP:    (0, -1),
    Action.DOWN:  (0,  1),
    Action.LEFT:  (-1, 0),
    Action.RIGHT: (1,  0),
}


class Environment:
    def __init__(self, height: int, width: int,
                 terminal_objects_placement: dict | None = None,
                 wall_placement: set | None = None,
                 max_steps: int = 50):
        self.height = height
        self.width = width

        # (x, y) -> reward value (+ve for goal, -ve for trap)
        self.terminal_objects_placement = terminal_objects_placement or {}
        # Set of (x, y) coordinates that are blocked
        self.wall_placement = wall_placement or set()

        self.max_steps = max_steps

        # State — set properly by reset()
        self.position = [0, 0]
        self.step_count = 0
        self.done = False

    def reset(self) -> list:
        """Reset the environment. Returns the initial observation [x, y]."""
        self.position = [0, 0]
        self.step_count = 0
        self.done = False
        return list(self.position)

    def step(self, action_int: int) -> tuple:
        """
        Take an action (int 0-3). Returns (observation, done, info).
        No reward — that comes from the learned reward model.
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")

        action = Action(action_int)
        dx, dy = DELTAS[action]

        new_x = self.position[0] + dx
        new_y = self.position[1] + dy

        info = {"event": "move"}

        # Boundary check
        if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height:
            info["event"] = "boundary"
            # Stay in place

        # Wall check
        elif (new_x, new_y) in self.wall_placement:
            info["event"] = "wall"
            # Stay in place

        else:
            # Valid move — update position
            self.position = [new_x, new_y]

            # Check if we landed on a terminal cell
            pos_tuple = tuple(self.position)
            if pos_tuple in self.terminal_objects_placement:
                info["event"] = "terminal"
                info["terminal_value"] = self.terminal_objects_placement[pos_tuple]
                self.done = True

        # Increment step count (every action counts, even boundary/wall)
        self.step_count += 1
        if self.step_count >= self.max_steps and not self.done:
            self.done = True
            info["event"] = "max_steps"

        return list(self.position), self.done, info

    def render(self) -> str:
        """Simple text rendering of the grid. Returns a string."""
        lines = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if [x, y] == self.position:
                    row.append("C")  # Cat
                elif (x, y) in self.wall_placement:
                    row.append("#")  # Wall
                elif (x, y) in self.terminal_objects_placement:
                    val = self.terminal_objects_placement[(x, y)]
                    row.append("G" if val > 0 else "T")  # Goal or Trap
                else:
                    row.append(".")  # Empty
            lines.append(" ".join(row))
        return "\n".join(lines)
