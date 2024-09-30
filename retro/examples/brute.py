"""
Implementation of the Brute from "Revisiting the Arcade Learning Environment:
Evaluation Protocols and Open Problems for General Agents" by Machado et al.
https://arxiv.org/abs/1709.06009

This is an agent that uses the determinism of the environment in order to do
pretty well at a number of retro games.  It does not save emulator state but
does rely on the same sequence of actions producing the same result when played
back.
"""

import argparse
import random

import gymnasium as gym
import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit

import retro

EXPLORATION_PARAM = 0.005

class Discretizer(gym.ActionWrapper):
    """ Wrap a gym environment and make it use discrete actions.  
    Args: combos: ordered list of lists of valid button combinations """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class JoustDiscretizer(Discretizer):
    """ Use Joust-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py """
    def __init__(self, env):
        super().__init__(env=env, 
            combos=[['LEFT'], ['RIGHT'], ['B'], ['LEFT', 'B'], ['RIGHT', 'B']]
        )

# class JoustDiscretizer(gym.ActionWrapper):
#     """
#     Wrap a gym-retro environment and make it use discrete actions for the Joust game.
#     """
#     def __init__(self, env):
#         super()
#         buttons = env.unwrapped.buttons
#         # buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
#         combos=[['LEFT'], ['RIGHT'], ['B'], ['LEFT', 'B'], ['RIGHT', 'B']]
#         # combos = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
#         #            ['DOWN', 'B'], ['B']]
#         self._actions = []
#         for action in combos:
#             arr = np.array([False] * len(env.action_space))
#             for button in action:
#                 arr[buttons.index(button)] = True
#             self._actions.append(arr)
#         self.action_space = gym.spaces.Discrete(len(self._actions))

#     def action(self, a): # pylint: disable=W0221
#         return self._actions[a].copy()


import time

class FPSWrapper(gym.Wrapper):
    def __init__(self, env, fps_calc_freq=100_000):
        super().__init__(env)
        self.fps_calc_freq = fps_calc_freq
        self.step_times = []
        self.last_print_time = time.time()
        self.total_steps = 0

    def step(self, action):
        start_time = time.time()
        observation, reward, done, truncated, info = self.env.step(action)
        end_time = time.time()
        
        self.step_times.append(end_time - start_time)
        self.total_steps += 1
        
        if len(self.step_times) >= self.fps_calc_freq:
            current_time = time.time()
            fps = self.fps_calc_freq / (current_time - self.last_print_time)
            print(f"FPS: {fps:.2f}, Total Steps: {self.total_steps}")
            self.step_times = []
            self.last_print_time = current_time
        
        return observation, reward, done, truncated, info


class Frameskip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def reset(self):
        return self.env.reset()

    def step(self, act):
        total_rew = 0.0
        terminated = False
        truncated = False
        for i in range(self._skip):
            obs, rew, terminated, truncated, info = self.env.step(act)
            total_rew += rew
            if terminated or truncated:
                break

        return obs, total_rew, terminated, truncated, info


class Node:
    def __init__(self, value=-np.inf, children=None):
        self.value = value
        self.visits = 0
        self.children = {} if children is None else children

    def __repr__(self):
        return "<Node value=%f visits=%d len(children)=%d>" % (
            self.value,
            self.visits,
            len(self.children),
        )


def select_actions(root, action_space, max_episode_steps):
    """
    Select actions from the tree

    Normally we select the greedy action that has the highest reward
    associated with that subtree.  We have a small chance to select a
    random action based on the exploration param and visit count of the
    current node at each step.

    We select actions for the longest possible episode, but normally these
    will not all be used.  They will instead be truncated to the length
    of the actual episode and then used to update the tree.
    """
    node = root

    acts = []
    steps = 0
    while steps < max_episode_steps:
        if node is None:
            # we've fallen off the explored area of the tree, just select random actions
            act = action_space.sample()
        else:
            epsilon = EXPLORATION_PARAM / np.log(node.visits + 2)
            if random.random() < epsilon:
                # random action
                act = action_space.sample()
            else:
                # greedy action
                act_value = {}
                for act in range(action_space.n):
                    if node is not None and act in node.children:
                        act_value[act] = node.children[act].value
                    else:
                        act_value[act] = -np.inf
                best_value = max(act_value.values())
                best_acts = [
                    act for act, value in act_value.items() if value == best_value
                ]
                act = random.choice(best_acts)

            if act in node.children:
                node = node.children[act]
            else:
                node = None

        acts.append(act)
        steps += 1

    return acts


def rollout(env, acts):
    """
    Perform a rollout using a preset collection of actions
    """
    total_rew = 0
    env.reset()
    steps = 0
    for act in acts:
        _obs, rew, terminated, truncated, _info = env.step(act)
        steps += 1
        total_rew += rew
        if terminated or truncated:
            break

    return steps, total_rew


def update_tree(root, executed_acts, total_rew):
    """
    Given the tree, a list of actions that were executed before the game ended, and a reward, update the tree
    so that the path formed by the executed actions are all updated to the new reward.
    """
    root.value = max(total_rew, root.value)
    root.visits += 1
    new_nodes = 0

    node = root
    for step, act in enumerate(executed_acts):
        if act not in node.children:
            node.children[act] = Node()
            new_nodes += 1
        node = node.children[act]
        node.value = max(total_rew, node.value)
        node.visits += 1

    return new_nodes


class Brute:
    """
    Implementation of the Brute

    Creates and manages the tree storing game actions and rewards
    """

    def __init__(self, env, max_episode_steps):
        self.node_count = 1
        self._root = Node()
        self._env = env
        self._max_episode_steps = max_episode_steps

    def run(self):
        acts = select_actions(
            self._root,
            self._env.action_space,
            self._max_episode_steps,
        )
        steps, total_rew = rollout(self._env, acts)
        executed_acts = acts[:steps]
        self.node_count += update_tree(self._root, executed_acts, total_rew)
        return executed_acts, total_rew


def brute_retro(
    game,
    max_episode_steps=4500,
    timestep_limit=1e8,
    state=retro.State.DEFAULT,
    scenario=None,
    record=False
):
    env = retro.make(
        game,
        state,
        # use_restricted_actions=retro.Actions.DISCRETE,
        scenario=scenario,
        record=record
    )
    env.render_mode= None
    env = FPSWrapper(env)
    env = JoustDiscretizer(env)
    env = Frameskip(env)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    brute = Brute(env, max_episode_steps=max_episode_steps)
    timesteps = 0
    best_rew = float("-inf")
    while True:
        acts, rew = brute.run()
        timesteps += len(acts)

        if rew > best_rew:
            print(f"new best reward {best_rew} => {rew}")
            best_rew = rew
            env.unwrapped.record_movie(f"best_{best_rew}.bk2")
            env.reset()
            for act in acts:
                env.step(act)
            env.unwrapped.stop_record()

        if timesteps > timestep_limit:
            print("timestep limit exceeded")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Joust-Arcade")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    brute_retro(game=args.game, state=args.state, scenario=args.scenario, max_episode_steps=500_000)


if __name__ == "__main__":
    main()
