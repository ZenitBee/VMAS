


from vmas.simulator.scenario import BaseScenario

class MyScenario(BaseScenario):

    def make_world(self, batch_dim, device, **kwargs):
        raise NotImplementedError()

    def reset_world_at(self, env_index):
        raise NotImplementedError()

    def observation(self, agent):
        raise NotImplementedError()

    def reward(self, agent):
        raise NotImplementedError()

import typing
from typing import Dict, List

import torch
from torch import Tensor

from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.dynamics.diff_drive import DiffDrive
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import (
    ANGULAR_FRICTION,
    Color,
    DRAG,
    LINEAR_FRICTION,
    ScenarioUtils,
)

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


def make_world(self, batch_dim: int, device: torch.device, **kwargs):
    ################
    # Scenario configuration
    ################
    self.plot_grid = False  # You can use this to plot a grid under the rendering for visualization purposes

    self.n_agents_holonomic = kwargs.pop(
        "n_agents_holonomic", 2
    )  # Number of agents with holonomic dynamics
    self.n_agents_diff_drive = kwargs.pop(
        "n_agents_diff_drive", 1
    )  # Number of agents with differential drive dynamics
    self.n_agents_car = kwargs.pop(
        "n_agents_car", 1
    )  # Number of agents with car dynamics
    self.n_agents = (
        self.n_agents_holonomic + self.n_agents_diff_drive + self.n_agents_car
    )
    self.n_obstacles = kwargs.pop("n_obstacles", 2)

    self.world_spawning_x = kwargs.pop(
        "world_spawning_x", 1
    )  # X-coordinate limit for entities spawning
    self.world_spawning_y = kwargs.pop(
        "world_spawning_y", 1
    )  # Y-coordinate limit for entities spawning

    self.comms_rendering_range = kwargs.pop(
        "comms_rendering_range", 0
    )  # Used for rendering communication lines between agents (just visual)
    self.lidar_range = kwargs.pop("lidar_range", 0.3)  # Range of the LIDAR sensor
    self.n_lidar_rays = kwargs.pop(
        "n_lidar_rays", 12
    )  # Number of LIDAR rays around the agent, each ray gives an observation between 0 and lidar_range

    self.shared_rew = kwargs.pop(
        "shared_rew", False
    )  # Whether the agents get a global or local reward for going to their goals
    self.final_reward = kwargs.pop(
        "final_reward", 0.01
    )  # Final reward that all the agents get when the scenario is done
    self.agent_collision_penalty = kwargs.pop(
        "agent_collision_penalty", -1
    )  # Penalty reward that an agent gets for colliding with another agent or obstacle

    self.agent_radius = kwargs.pop("agent_radius", 0.1)
    self.min_distance_between_entities = (
        self.agent_radius * 2 + 0.05
    )  # Minimum distance between entities at spawning time
    self.min_collision_distance = (
        0.005  # Minimum distance between entities for collision trigger
    )

    ScenarioUtils.check_kwargs_consumed(kwargs) # Warn if not all kwargs have been consumed


    ################
    # Make world
    ################
    world = World(
        batch_dim,  # Number of environments simulated
        device,  # Device for simulation
        substeps=5,  # Number of physical substeps (more yields more accurate but more expensive physics)
        collision_force=500,  # Paramneter to tune for collisions
        dt=0.1,  # Simulation timestep
        gravity=(-5, -9),  # Customizable gravity
        drag=DRAG,  # Physics parameters
        linear_friction=LINEAR_FRICTION,  # Physics parameters
        angular_friction=ANGULAR_FRICTION,  # Physics parameters
        # There are many more....
    )

    ################
    # Add agents
    ################
    known_colors = [
        Color.BLUE,
        Color.ORANGE,
        Color.GREEN,
        Color.PINK,
        Color.PURPLE,
        Color.YELLOW,
        Color.RED,
    ]  # Colors for the first 7 agents
    colors = torch.randn(
        (max(self.n_agents - len(known_colors), 0), 3), device=device
    )  # Other colors if we have more agents are random

    self.goals = []  # We will store our agents' goal entities here for easy access
    for i in range(self.n_agents):
        color = (
            known_colors[i]
            if i < len(known_colors)
            else colors[i - len(known_colors)]
        )  # Get color for agent

        sensors = [
            Lidar(
                world,
                n_rays=self.n_lidar_rays,
                max_range=self.lidar_range,
                entity_filter=lambda e: isinstance(
                    e, Agent
                ),  # This makes sure that this lidar only percieves other agents
                angle_start=0.0,  # LIDAR angular ranges (we sense 360 degrees)
                angle_end=2
                * torch.pi,  # LIDAR angular ranges (we sense 360 degrees)
            )
        ]  # Agent LIDAR sensor

        if i < self.n_agents_holonomic:
            agent = Agent(
                name=f"holonomic_{i}",
                collide=True,
                color=color,
                render_action=True,
                sensors=sensors,
                shape=Sphere(radius=self.agent_radius),
                u_range=[1, 1],  # Ranges for actions
                u_multiplier=[1, 1],  # Action multipliers
                dynamics=Holonomic(),  # If you got to its class you can see it has 2 actions: force_x, and force_y
            )
        elif i < self.n_agents_holonomic + self.n_agents_diff_drive:
            agent = Agent(
                name=f"diff_drive_{i - self.n_agents_holonomic}",
                collide=True,
                color=color,
                render_action=True,
                sensors=sensors,
                shape=Sphere(radius=self.agent_radius),
                u_range=[1, 1],  # Ranges for actions
                u_multiplier=[0.5, 1],  # Action multipliers
                dynamics=DiffDrive(
                    world
                ),  # If you go to its class you can see it has 2 actions: forward velocity and angular velocity
            )
        else:
            max_steering_angle = torch.pi / 4
            width = self.agent_radius
            agent = Agent(
                name=f"car_{i-self.n_agents_holonomic-self.n_agents_diff_drive}",
                linear_friction=1,
                angular_friction=1,
                drag=1,
                collide=True,
                color=color,
                render_action=True,
                sensors=sensors,
                shape=Box(length=self.agent_radius * 3, width=width),
                u_range=[1, max_steering_angle],
                u_multiplier=[0.5, 1],
                dynamics=KinematicBicycle(
                    world,
                    width=width,
                    l_f=self.agent_radius,  # Distance between the front axle and the center of gravity
                    l_r=self.agent_radius,  # Distance between the rear axle and the center of gravity
                    max_steering_angle=max_steering_angle,
                ),  # If you got to its class you can see it has 2 actions: forward velocity and steering angle
            )
        agent.pos_rew = torch.zeros(
            batch_dim, device=device
        )  # Tensor that will hold the position reward fo the agent
        agent.agent_collision_rew = (
            agent.pos_rew.clone()
        )  # Tensor that will hold the collision reward fo the agent

        world.add_agent(agent)  # Add the agent to the world

        ################
        # Add goals
        ################
        goal = Landmark(
            name=f"goal_{i}",
            collide=False,
            color=color,
        )
        world.add_landmark(goal)
        agent.goal = goal
        self.goals.append(goal)

    ################
    # Add obstacles
    ################
    self.obstacles = (
        []
    )  # We will store obstacles here for easy access
    for i in range(self.n_obstacles):
        obstacle = Landmark(
            name=f"obstacle_{i}",
            collide=True,
            color=Color.BLACK,
            shape=Sphere(radius=self.agent_radius * 2 / 3),
        )
        world.add_landmark(obstacle)
        self.obstacles.append(obstacle)

    self.pos_rew = torch.zeros(
        batch_dim, device=device
    )  # Tensor that will hold the global position reward
    self.final_rew = (
        self.pos_rew.clone()
    )  # Tensor that will hold the global done reward
    self.all_goal_reached = (
        self.pos_rew.clone()
      )  # Tensor indicating if all goals have been reached

    return world

MyScenario.make_world = make_world

def reset_world_at(self, env_index: int = None):
    ScenarioUtils.spawn_entities_randomly(
        self.world.agents
        + self.obstacles
        + self.goals,  # List of entities to spawn
        self.world,
        env_index, # Pass the env_index so we only reset what needs resetting
        self.min_distance_between_entities,
        x_bounds=(-self.world_spawning_x, self.world_spawning_x),
        y_bounds=(-self.world_spawning_y, self.world_spawning_y),
    )

    for agent in self.world.agents:
        if env_index is None:
            agent.goal_dist = torch.linalg.vector_norm(
                agent.state.pos - agent.goal.state.pos,
                dim=-1,
            )  # Tensor holding the distance of the agent to the goal, we will use it to compute the reward
        else:
            agent.goal_dist[env_index] = torch.linalg.vector_norm(
                agent.state.pos[env_index] - agent.goal.state.pos[env_index]
            )

MyScenario.reset_world_at = reset_world_at

def reward(self, agent: Agent):
    is_first = agent == self.world.agents[0]

    if is_first:
        # We can compute rewards when the first agent is called such that we do not have to recompute global components

        self.pos_rew[:] = 0  # Reset previous reward
        self.final_rew[:] = 0  # Reset previous reward

        for a in self.world.agents:
            a.agent_collision_rew[:] = 0  # Reset previous reward
            distance_to_goal = torch.linalg.vector_norm(
                a.state.pos - a.goal.state.pos,
                dim=-1,
            )
            a.on_goal = distance_to_goal < a.shape.circumscribed_radius()

            # The positional reward is the delta in distance to the goal.
            # This makes it so that if the agent moves 1m towards the goal it is rewarded
            # the same amount regardless of its absolute distance to it
            # This would not be the case if pos_rew = -distance_to_goal (a common choice)
            # This choice leads to better training
            a.pos_rew = a.goal_dist - distance_to_goal

            a.goal_dist = distance_to_goal  # Update distance to goal
            self.pos_rew += a.pos_rew  # Global pos reward

        # If all agents reached their goal we give them all a final_rew
        self.all_goal_reached = torch.all(
            torch.stack([a.on_goal for a in self.world.agents], dim=-1),
            dim=-1,
        )
        self.final_rew[self.all_goal_reached] = self.final_reward

        for i, a in enumerate(self.world.agents):
            # Agent-agent collision
            for j, b in enumerate(self.world.agents):
                if i <= j:
                    continue
                if self.world.collides(a, b):
                    distance = self.world.get_distance(a, b)
                    a.agent_collision_rew[
                        distance <= self.min_collision_distance
                    ] += self.agent_collision_penalty
                    b.agent_collision_rew[
                        distance <= self.min_collision_distance
                    ] += self.agent_collision_penalty
            # Agent obstacle collision
            for b in self.obstacles:
                if self.world.collides(a, b):
                    distance = self.world.get_distance(a, b)
                    a.agent_collision_rew[
                        distance <= self.min_collision_distance
                    ] += self.agent_collision_penalty

    pos_reward = (
        self.pos_rew if self.shared_rew else agent.pos_rew
    )  # Choose global or local reward based on configuration
    return pos_reward + self.final_rew + agent.agent_collision_rew

MyScenario.reward = reward

def observation(self, agent: Agent):
    obs = {
        "obs": torch.cat(
            [
                agent.state.pos - agent.goal.state.pos
            ]  # Relative position to goal (fundamental)
            + [
                agent.state.pos - obstacle.state.pos for obstacle in self.obstacles
            ]  # Relative position to obstacles (fundamental)
            + [
                sensor._max_range - sensor.measure() for sensor in agent.sensors
            ],  # LIDAR to avoid other agents
            dim=-1,
        ),
        "pos": agent.state.pos,
        "vel": agent.state.vel,
    }
    if not isinstance(agent.dynamics, Holonomic):
        # Non hoonomic agents need to know angular states
        obs.update(
            {
                "rot": agent.state.rot,
                "ang_vel": agent.state.ang_vel,
            }
        )
    return obs

MyScenario.observation = observation

def done(self) -> Tensor:
    return self.all_goal_reached

MyScenario.done = done

def info(self, agent: Agent) -> Dict[str, Tensor]:
    return {
        "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
        "final_rew": self.final_rew,
        "agent_collision_rew": agent.agent_collision_rew,
    }

MyScenario.info = info

def extra_render(self, env_index: int = 0) -> "List[Geom]":
    from vmas.simulator import rendering

    geoms = [
        ScenarioUtils.plot_entity_rotation(agent, env_index)
        for agent in self.world.agents
        if not isinstance(agent.dynamics, Holonomic)
    ]  # Plot the rotation for non-holonomic agents

    # Plot communication lines
    if self.comms_rendering_range > 0:
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                if agent_dist[env_index] <= self.comms_rendering_range:
                    color = Color.BLACK.value
                    line = rendering.Line(
                        (agent1.state.pos[env_index]),
                        (agent2.state.pos[env_index]),
                        width=1,
                    )
                    line.set_color(*color)
                    geoms.append(line)
    return geoms

MyScenario.extra_render = extra_render

# @title Scenario parameters
n_agents_holonomic = 0 # @param {"type":"integer"}
n_agents_diff_drive = 0 # @param {"type":"integer"}
n_agents_car = 3 # @param {"type":"integer"}
n_obstacles = 1 # @param {"type":"integer"}

vmas_device = "cpu" # @param {"type":"string"}
num_envs = 8 # @param {"type":"integer"}

from vmas import make_env
env = make_env(
    scenario=MyScenario(),
    num_envs=num_envs,
    device=vmas_device,
    seed=0,
    # Optional parameters, we use the defaults
    continuous_actions=True,
    max_steps=None, # We just use the scenario done
    dict_spaces=False, # Whether observations, infos, and rewards are dictionaries mapping agent name to data or lists. For more info see https://github.com/proroklab/VectorizedMultiAgentSimulator?tab=readme-ov-file#output-spaces
    multidiscrete_actions=False, # In case of discrete actions, whether to use multidiscrete spaces of just bigger discrete ones. Not relevant here
    grad_enabled=False, # Whether we run the simulator in a differentiable manner
    terminated_truncated=False, # Whether to separate terminate and truncated into distinct flags
    # Scenario parameters
    n_agents_holonomic=n_agents_holonomic,
    n_agents_diff_drive=n_agents_diff_drive,
    n_agents_car=n_agents_car,
    n_obstacles=n_obstacles
)

# debug prints

actions = env.get_random_actions()
#print(actions)


obs, rews, dones, info = env.step(actions)
#print(f"Obs length: {len(obs)}, observation of agent 0:\n{obs[0]}")



# print(f"Rewards length: {len(rews)}, reward of agent 0:\n{rews[0]}")
#print(dones)

from vmas.simulator.scenario import BaseScenario
from typing import Union
import time
import torch
from vmas import make_env
from vmas.simulator.core import Agent

def use_vmas_env(
    render: bool,
    num_envs: int,
    n_steps: int,
    device: str,
    scenario: Union[str, BaseScenario],
    continuous_actions: bool,
    **kwargs
):
    """Example function to use a vmas environment.

    This is a simplification of the function in `vmas.examples.use_vmas_env.py`.

    Args:
        continuous_actions (bool): Whether the agents have continuous or discrete actions
        scenario (str, BaseScenario): Name of scenario or scenario class
        device (str): Torch device to use
        render (bool): Whether to render the scenario
        num_envs (int): Number of vectorized environments
        n_steps (int): Number of steps before returning done

    """

    scenario_name = scenario if isinstance(scenario,str) else scenario.__class__.__name__

    env = make_env(
        scenario=scenario,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        seed=0,
        # Environment specific variables
        **kwargs
    )

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0

    for s in range(n_steps):
        step += 1
        print(f"Step {step}")

        actions = []
        for i, agent in enumerate(env.agents):
            action = env.get_random_action(agent)

            actions.append(action)

        obs, rews, dones, info = env.step(actions)

        if render:
            frame = env.render(mode="rgb_array")
            frame_list.append(frame)

    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} "
        f"for {scenario_name} scenario."
    )

    if render:
        from moviepy.editor import ImageSequenceClip
        fps=30
        clip = ImageSequenceClip(frame_list, fps=fps)
        clip.write_gif(f'{scenario_name}.gif', fps=fps)


use_vmas_env(
    render=True,
    num_envs=num_envs,
    n_steps=100,
    device=vmas_device,
    scenario=MyScenario(),
    continuous_actions=True,
    # Scenario kwargs
    n_agents_holonomic=n_agents_holonomic,
    n_agents_diff_drive=n_agents_diff_drive,
    n_agents_car=n_agents_car,
    n_obstacles=n_obstacles
)

from IPython.display import Image
Image(f'{MyScenario.__name__}.gif')