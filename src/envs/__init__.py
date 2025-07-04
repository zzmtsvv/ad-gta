import gymnasium as gym
# from src.envs.dark_key_to_door import DarkKeyToDoor, QDarkKeyToDoor
from src.envs.dark_room import DarkRoom


gym.register(
    id="Dark-Room-9x9-v0",
    entry_point="src.envs.dark_room:DarkRoom",
    max_episode_steps=100,
    kwargs={
        "size": 9,
        "terminate_on_goal": False,
    },
)

gym.register(
    id="Dark-Room-3x3-v0",
    entry_point="src.envs.dark_room:DarkRoom",
    max_episode_steps=100,
    kwargs={
        "size": 3,
        "terminate_on_goal": False,
    },
)


gym.register(
    id="Dark-Room-Hard-17x17-v0",
    entry_point="src.envs.dark_room:DarkRoom",
    max_episode_steps=20,
    kwargs={
        "size": 17,
        "terminate_on_goal": True,
    },
)


# gym.register(
#     id="Dark-Key2Door-9x9-v0",
#     entry_point="src.envs.dark_key_to_door:DarkKeyToDoor",
#     max_episode_steps=50,
#     kwargs={
#         "size": 9
#     }
# )

# gym.register(
#     id="Dark-Key2Door-3x3-v0",
#     entry_point="src.envs.dark_key_to_door:DarkKeyToDoor",
#     max_episode_steps=20,
#     kwargs={
#         "size": 3
#     }
# )

# gym.register(
#     id="Q-Dark-Key2Door-9x9-v0",
#     entry_point="src.envs.dark_key_to_door:QDarkKeyToDoor",
#     max_episode_steps=50,
#     kwargs={
#         "size": 9
#     }
# )

# gym.register(
#     id="Q-Dark-Key2Door-3x3-v0",
#     entry_point="src.envs.dark_key_to_door:QDarkKeyToDoor",
#     max_episode_steps=20,
#     kwargs={
#         "size": 3
#     }
# )