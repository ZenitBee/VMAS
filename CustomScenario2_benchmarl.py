# @title Rendering dependencies

# import pyvirtualdisplay
# display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
# display.start()

# @title Install GNN dependencies
# Add this in a Google Colab cell to install the correct version of Pytorch Geometric.
import torch

def format_pytorch_version(version):
  return version.split('+')[0]

TORCH_version = torch.__version__
TORCH = format_pytorch_version(TORCH_version)

def format_cuda_version(version):
  return 'cu' + version.replace('.', '')

#CUDA_version = torch.version.cuda
#CUDA = format_cuda_version(CUDA_version)

#  Copyright (c) 2025.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import typing
from typing import Dict, List

import torch

from torch import Tensor

from vmas import render_interactively
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


class MyScenario(BaseScenario):
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

        ScenarioUtils.check_kwargs_consumed(kwargs) # Warn is not all kwargs have been consumed


        ################
        # Make world
        ################
        world = World(
            batch_dim,  # Number of environments simulated
            device,  # Device for simulation
            substeps=5,  # Number of physical substeps (more yields more accurate but more expensive physics)
            collision_force=500,  # Paramneter to tune for collisions
            dt=0.1,  # Simulation timestep
            gravity=(0.0, 0.0),  # Customizable gravity
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

        self.goals = []  # We will store our agent goal entities here for easy access
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
                    ),  # If you got to its class you can see it has 2 actions: forward velocity and angular velocity
                )
            else:
                max_steering_angle = torch.pi / 4
                width = self.agent_radius
                agent = Agent(
                    name=f"car_{i-self.n_agents_holonomic-self.n_agents_diff_drive}",
                    collide=True,
                    color=color,
                    render_action=True,
                    sensors=sensors,
                    shape=Box(length=self.agent_radius * 2, width=width),
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

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents
            + self.obstacles
            + self.goals,  # List of entities to spawn
            self.world,
            env_index,  # Pass the env_index so we only reset what needs resetting
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

    def done(self) -> Tensor:
        return self.all_goal_reached

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            "agent_collision_rew": agent.agent_collision_rew,
        }

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


from benchmarl.environments import VmasTask, Smacv2Task, PettingZooTask, MeltingPotTask

# VmasTask.BALANCE  # Try deleting the enum element name and see all the available ones
# Smacv2Task.PROTOSS_10_VS_10  # Try deleting the enum element name and see all the available ones
# PettingZooTask.MULTIWALKER  # Try deleting the enum element name and see all the available ones
# MeltingPotTask.COMMONS_HARVEST__OPEN  # Try deleting the enum element name and see all the available ones

import copy
from typing import Callable, Optional
from benchmarl.environments import VmasTask
from benchmarl.utils import DEVICE_TYPING
from torchrl.envs import EnvBase, VmasEnv

def get_env_fun(
    self,
    num_envs: int,
    continuous_actions: bool,
    seed: Optional[int],
    device: DEVICE_TYPING,
) -> Callable[[], EnvBase]:
    config = copy.deepcopy(self.config)
    if (hasattr(self, "name") and self.name is "NAVIGATION") or (
        self is VmasTask.NAVIGATION
    ):  # This is the only modification we make ....
        scenario = MyScenario()  # .... ends here
    else:
        scenario = self.name.lower()
    return lambda: VmasEnv(
        scenario=scenario,
        num_envs=num_envs,
        continuous_actions=continuous_actions,
        seed=seed,
        device=device,
        categorical_actions=True,
        clamp_actions=True,
        **config,
    )

try:
    from benchmarl.environments import VmasClass
    VmasClass.get_env_fun = get_env_fun
except ImportError:
    VmasTask.get_env_fun = get_env_fun

# @title Devices
train_device = "cpu" # @param {"type":"string"}
vmas_device = "cpu" # @param {"type":"string"}

from benchmarl.experiment import ExperimentConfig

# Loads from "benchmarl/conf/experiment/base_experiment.yaml"
experiment_config = ExperimentConfig.get_from_yaml() # We start by loading the defaults

# Override devices
experiment_config.sampling_device = vmas_device
experiment_config.train_device = train_device

# experiment_config.max_n_frames = 10_000_000 # Number of frames before training ends
experiment_config.gamma = 0.99
# experiment_config.max_n_iters= 3
experiment_config.on_policy_collected_frames_per_batch = 20_000 # Number of frames collected each iteration
experiment_config.on_policy_n_envs_per_worker = 400 # Number of vmas vectorized enviornemnts (each will collect 100 steps, see max_steps in task_config -> 600 * 100 = 60_000 the number above)
experiment_config.on_policy_n_minibatch_iters = 45
experiment_config.on_policy_minibatch_size = 4096
experiment_config.evaluation = True
experiment_config.render = True
experiment_config.share_policy_params = True # Policy parameter sharing on
experiment_config.evaluation_interval = 120_000 # Interval in terms of frames, will evaluate every 120_000 / 60_000 = 2 iterations
experiment_config.evaluation_episodes = 200 # Number of vmas vectorized enviornemnts used in evaluation
experiment_config.loggers = ["csv"] # Log to csv, usually you should use wandb

# Loads from "benchmarl/conf/task/vmas/navigation.yaml"
task = VmasTask.NAVIGATION.get_from_yaml()

# We override the NAVIGATION config with ours
task.config = {
        "max_steps": 400,
        "n_agents_holonomic": 1,
        "n_agents_diff_drive": 2,
        "n_agents_car": 1,
        "n_obstacles" : 2,
        "lidar_range": 0.35,
        "comms_rendering_range": 0,
        "shared_rew": False,
}

from benchmarl.algorithms import MappoConfig

# We can load from "benchmarl/conf/algorithm/mappo.yaml"
algorithm_config = MappoConfig.get_from_yaml()

# Or create it from scratch
algorithm_config = MappoConfig(
        share_param_critic=True, # Critic param sharing on
        clip_epsilon=0.2,
        entropy_coef=0.001, # We modify this, default is 0
        critic_coef=1,
        loss_critic_type="l2",
        lmbda=0.9,
        scale_mapping="biased_softplus_1.0", # Mapping for standard deviation
        use_tanh_normal=True,
        minibatch_advantage=False,
    )

from benchmarl.models.mlp import MlpConfig

model_config = MlpConfig(
        num_cells=[256, 256], # Two layers with 256 neurons each
        layer_class=torch.nn.Linear,
        activation_class=torch.nn.Tanh,
    )

# Loads from "benchmarl/conf/model/layers/mlp.yaml" (in this case we use the defaults so it is the same)
model_config = MlpConfig.get_from_yaml()
critic_model_config = MlpConfig.get_from_yaml()

from benchmarl.experiment import Experiment

experiment_config.max_n_frames = 5_000_000 # Runs one iteration, change to 50_000_000 for full training
# experiment_config.on_policy_n_envs_per_worker = 60 # Remove this line for full training
# experiment_config.on_policy_n_minibatch_iters = 10 # Remove this line for full training

experiment = Experiment(
    task=task,
    algorithm_config=algorithm_config,
    model_config=model_config,
    critic_model_config=critic_model_config,
    seed=0,
    config=experiment_config,
)


experiment.run()



#print("Getting a directory of experiment", dir(experiment))
#print("Getting the experiments observation spec",experiment.observation_spec)
print("Getting the trained policy from the experiment.")
trained_policy = experiment.policy

#print("the policy from the experiment is: {}".format(trained_policy))
#print("the experiment.policy is ", type(experiment.policy))
#print("the policy copied from the experiment is ", type(trained_policy))



print("Saving the trained policy.")
torch.save(trained_policy.state_dict(), "trained_policy.pt")

device="cpu"

# @title Scenario parameters
n_agents_holonomic = 1 # @param {"type":"integer"}
n_agents_diff_drive = 2 # @param {"type":"integer"}
n_agents_car = 1 # @param {"type":"integer"}
n_obstacles = 2 # @param {"type":"integer"}

vmas_device = "cpu" # @param {"type":"string"}
num_envs = 8 # @param {"type":"integer"}


from vmas import make_env
import matplotlib.pyplot as plt

env = VmasEnv(
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

# print("model parameters",trained_policy.parameters())
# for s in trained_policy.parameters():
#     print(s.shape,s)

# print("parameter 0", trained_policy.parameters()[0])
# print("parameter 1", trained_policy.parameters()[1])

# print("those were the parameters of the model, now the state dict==================")
# print(trained_policy.state_dict())
#



i = 0
while(i < 1000):
    with torch.no_grad():
       env.rollout(
           max_steps=experiment.max_steps,
           policy=trained_policy,
           callback=lambda env, _: env.render(),
           auto_cast_to_device=True,
           break_when_any_done=False,
       )
    i += 1


#
# trained_policy.eval()
#
#
# obs = env.reset()
#
#
# #print("obs is a list of length",len(obs))
# # print("First Element", obs[0])
# # print("Second Element", obs[1])
# # print("Third Element", obs[2])
# #
# #
# # print("First Element shape", len(obs[0]))
# # print("Second Element shape", len(obs[1]))
# # print("Third Element shape", len(obs[2]))
#
#
#
#
#
#
#
# done = False
#
# while not done:
#     with torch.no_grad():
#         actions = trained_policy(obs)
#         # actions = trained_policy.get_actions(obs)    #suggested by pycharm
#         # actions = trained_policy(obs)                 # suggested by chatGPT
#         # actions = trained_policy(obs)                   # suggested by perplexity - same as chat GPT
#
#
#     obs, rewards, dones, info = env.step(actions)
#     dones = dones.any()
#
#     frame = env.render(mode="rgb_array")
#     plt.imshow(frame)
#     plt.pause(0.01)
# env.close()
# plt.close()




