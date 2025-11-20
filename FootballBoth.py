# Torch
import torch

# Tensordict modules
from tensordict.nn import set_composite_lp_aggregate, TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing

# for policy / probabalistic actor
from torch.distributions import Categorical

# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Utils
torch.manual_seed(0)
from matplotlib import pyplot as plt
from tqdm import tqdm

# Devices
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
vmas_device = device  # The device where the simulator is run (VMAS can run on GPU)

# print("vmas_device is ", vmas_device)

# Sampling
frames_per_batch = 30_000  # Number of team frames collected per training iteration
n_iters = 1000  # Number of sampling and training iterations
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 8  # Number of optimization steps per training iteration
minibatch_size = 5000  # Size of the mini-batches in each optimization step
lr = 3e-4  # Learning rate
max_grad_norm = 0.5  # Maximum norm for the gradients

# PPO
clip_epsilon = 0.2  # clip value for PPO loss
gamma = 0.99  # discount factor
lmbda = 0.95  # lambda for generalised advantage estimation
entropy_eps = 0.01  # coefficient of the entropy term in the PPO loss

# disable log-prob aggregation
set_composite_lp_aggregate(False).set()

max_steps = 800  # Episode steps before done
num_vmas_envs = (
    frames_per_batch // max_steps
)  # Number of vectorized envs. frames_per_batch should be divisible by this number
scenario_name = "football"

#environment parameters
n_blue_agents=5
n_red_agents=5
ai_blue_agents=False
ai_red_agents=False
ai_strength=1
ai_decision_strength=1
ai_precision_strength=1
n_traj_points=8
physically_different=False
enable_shooting=False


env = VmasEnv(
    scenario=scenario_name,
    num_envs=num_vmas_envs,
    continuous_actions=False,  # VMAS supports both continuous and discrete actions
    max_steps=max_steps,
    device=vmas_device,
    # Scenario kwargs
    n_blue_agents=n_blue_agents,  # These are custom kwargs that change for each VMAS scenario, see the VMAS repo to know more.
    n_red_agents=n_red_agents,
    ai_blue_agents=ai_blue_agents,
    ai_red_agents=ai_red_agents,
    ai_strength=ai_strength,
    ai_decision_strength=ai_decision_strength,
    ai_precision_strength=ai_precision_strength,
    n_traj_points=n_traj_points,
    physically_different=physically_different,
    enable_shooting=enable_shooting


)

# print("action_spec:", env.full_action_spec)
#print("what the fk does this do: ",env.full_action_spec[env.action_key].shape[-1])
#print("what the fk does this do: ",env.full_action_spec[env.action_keys].shape[-1])
# print("what is it?", env.action_keys)
#exit(0)

# print("reward_spec:", env.full_reward_spec)
# print("done_spec:", env.full_done_spec)
# print("observation_spec:", env.observation_spec)
# print("action_keys:", env.action_keys)
# print("reward_keys:", env.reward_keys)
# print("done_keys:", env.done_keys)
# print("env.observation spec", env.observation_spec)

#print(env.reward_keys)

env = TransformedEnv(
    env,
    RewardSum(in_keys=env.reward_keys, out_keys=[("agent_blue", "episode_reward"), ("agent_red", "episode_reward")]),
)

check_env_specs(env)

#print("env", env)
#print("env.observation spec", env.observation_spec)
# n_rollout_steps = 5
# rollout = env.rollout(n_rollout_steps)
# print("rollout of three steps:", rollout)
# print("Shape of the rollout TensorDict:", rollout.batch_size)

# POLICY / ACTOR

share_parameters_policy = True

# debugging
#print ("observation spec shape ", env.observation_spec["agent_blue", "observation"].shape[-1])
#print ("observation spec ", env.observation_spec["agent_blue", "observation"])
#print ("observation spec raw ", env.observation_spec)
#print("n_agents", env.n_agents)
#print("red ones ", n_red_agents)
#print("blue ones ",n_blue_agents)
#print(env.observation_spec["agent_blue", "observation"].shape[-1])
#print(env.observation_spec["agent_blue", "observation"].shape)

policy_net = torch.nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agent_blue", "observation"].shape[-1],  # n_obs_per_agent
        n_agent_outputs=9,  # one output per discrete action
        n_agents=n_blue_agents,
        centralised=False,  # decentralized agents using their own observations
        share_params=share_parameters_policy,
        device=device,
        depth=2,
        num_cells=128,
        activation_class=torch.nn.Tanh,
    ),
)

policy_net2 = torch.nn.Sequential(
    #DebugLayer1(),
    MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agent_red", "observation"].shape[-1],  # n_obs_per_agent
        n_agent_outputs=9,  # one output per discrete action
        n_agents=n_red_agents,
        centralised=False,  # decentralized agents using their own observations
        share_params=share_parameters_policy,
        device=device,
        depth=2,
        num_cells=128,
        activation_class=torch.nn.Tanh,
    ),
    #DebugLayer2(),
    # No NormalParamExtractor here because output is logits for discrete actions
)

policy_module = TensorDictModule(
    policy_net,
    in_keys=[("agent_blue", "observation")],
    out_keys=[("agent_blue", "logits")],  # logits for Categorical distribution
)

policy_module2 = TensorDictModule(
    policy_net2,
    in_keys=[("agent_red", "observation")],
    out_keys=[("agent_red", "logits")],  # logits for Categorical distribution
)


#print("Action Spec 1: ", env.action_spec_unbatched["agent_blue"])
#print("Action Spec 2: ", env.action_spec_unbatched["agent_blue"]["action"])


#print ("ACTION SPEC: ", env.action_spec_unbatched)
#print ("ACTION KEY", env.action_key)


#print(env.action_spec_unbatched["agent_blue"]["action"])
# what we want is  {('agent_blue', 'action')}
# error is got: {('agent_blue', 'action'), 'action'} and {('agent_blue', 'action')} respectively

from torchrl.data import CompositeSpec, DiscreteTensorSpec

policy = ProbabilisticActor(
    module=policy_module,
    # spec=env.action_spec_unbatched,
    in_keys=[("agent_blue", "logits")],  # logits for discrete actions
    out_keys=env.action_keys[0],
    distribution_class=Categorical,
    distribution_kwargs={},  # Categorical uses 'logits' from forward pass
    return_log_prob=True,
)

policy2 = ProbabilisticActor(
    module=policy_module2,
    # spec=env.action_spec_unbatched,
    in_keys=[("agent_red", "logits")],  # logits for discrete actions
    out_keys=env.action_keys[1],
    distribution_class=Categorical,
    distribution_kwargs={},  # Categorical uses 'logits' from forward pass
    return_log_prob=True,
)

#print("======= actor2 ===========")
#print(policy2.state_dict())
#print(policy2)
#exit(0)


# print("======= actor ===========")
# print(policy.state_dict())


# CRITIC

share_parameters_critic = True
mappo = True  # IPPO if False

critic_net = MultiAgentMLP(
    n_agent_inputs=env.observation_spec["agent_blue", "observation"].shape[-1],
    n_agent_outputs=1,  # 1 value per agent
    n_agents=n_blue_agents,
    centralised=mappo,
    share_params=share_parameters_critic,
    device=device,
    depth=2,
    num_cells=256,
    activation_class=torch.nn.Tanh,
)

critic_net2 = MultiAgentMLP(
    n_agent_inputs=env.observation_spec["agent_red", "observation"].shape[-1],
    n_agent_outputs=1,  # 1 value per agent
    n_agents=n_red_agents,
    centralised=mappo,
    share_params=share_parameters_critic,
    device=device,
    depth=2,
    num_cells=256,
    activation_class=torch.nn.Tanh,
)

critic = TensorDictModule(
    module=critic_net,
    in_keys=[("agent_blue", "observation")],
    out_keys=[("agent_blue", "state_value")],
)

critic2 = TensorDictModule(
    module=critic_net2,
    in_keys=[("agent_red", "observation")],
    out_keys=[("agent_red", "state_value")],
)


# DATA COLLECTOR
blue_policy = torch.load("PPO_navigation_policy.pt", weights_only= False)


# for blue start learning with a pre trained policy

combined_policy = TensorDictSequential(
    blue_policy,
    policy2,
)

collector = SyncDataCollector(
    env,
    combined_policy,
    device=vmas_device,
    storing_device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
)



#REPLAY BUFFER

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(
        frames_per_batch, device=device
    ),  # We store the frames_per_batch collected at each iteration
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size,  # We will sample minibatches of this size
)

# LOSS FUNCTION

loss_module = ClipPPOLoss(
    actor_network=policy,
    critic_network=critic,
    clip_epsilon=clip_epsilon,
    entropy_coeff=entropy_eps,
    normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
)

loss_module2 = ClipPPOLoss(
    actor_network=policy2,
    critic_network=critic2,
    clip_epsilon=clip_epsilon,
    entropy_coeff=entropy_eps,
    normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
)

loss_module.set_keys(  # We have to tell the loss where to find the keys
    reward=env.reward_keys[0],
    action=env.action_keys[0],
    value=("agent_blue", "state_value"),
    # These last 2 keys will be expanded to match the reward shape
    done=("agent_blue", "done"),
    terminated=("agent_blue", "terminated"),
)

loss_module2.set_keys(  # We have to tell the loss where to find the keys
    reward=env.reward_keys[1],
    action=env.action_keys[1],
    value=("agent_red", "state_value"),
    # These last 2 keys will be expanded to match the reward shape
    done=("agent_red", "done"),
    terminated=("agent_red", "terminated"),
)

loss_module.make_value_estimator(
    ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
)  # We build GAE

loss_module2.make_value_estimator(
    ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
)  # We build GAE2


GAE = loss_module.value_estimator
GAE2 = loss_module2.value_estimator

optim = torch.optim.Adam(loss_module.parameters(), lr)
optim2 = torch.optim.Adam(loss_module2.parameters(), lr)


pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")

#some debug put in by George - to print out the tensordict structure
# for tensordict_data in collector:
#     print("TD:",tensordict_data)

# print("=====================before===================================================")
# print(collector)
# exit(0)
episode_reward_mean_list = []
episode_reward_mean_list2 = []
for tensordict_data in collector:
    tensordict_data.set(
        ("next", "agent_blue", "done"),
        tensordict_data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_keys[0]))),
    )
    tensordict_data.set(
        ("next", "agent_red", "done"),
        tensordict_data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_keys[1]))),
    )
    tensordict_data.set(
        ("next", "agent_blue", "terminated"),
        tensordict_data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_keys[0]))),
    )
    tensordict_data.set(
        ("next", "agent_red", "terminated"),
        tensordict_data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_keys[1]))),
    )
    # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)
    # print(tensordict_data.get("agent_blue"))
    # print(tensordict_data.shape)

    with torch.no_grad():
        GAE(
            tensordict_data,
            params=loss_module.critic_network_params,
            target_params=loss_module.target_critic_network_params,
        )  # Compute GAE and add it to the data
        GAE2(
            tensordict_data,
            params=loss_module2.critic_network_params,
            target_params=loss_module2.target_critic_network_params,
        )  # Compute GAE and add it to the data

    data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
    replay_buffer.extend(data_view)



    for _ in range(num_epochs):
        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer.sample()
            #print(subdata)
            loss_vals = loss_module(subdata)
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_grad_norm
            )  # Optional

            optim.step()
            optim.zero_grad()

        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer.sample()
            #print(subdata)
            loss_vals = loss_module2(subdata)
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(
                loss_module2.parameters(), max_grad_norm
            )  # Optional

            optim2.step()
            optim2.zero_grad()

    collector.update_policy_weights_()

    # Logging
    done = tensordict_data.get(("next", "agent_blue", "done")) # or tensordict_data.get(("next", "agent_red", "done"))
    episode_reward_mean = (
        tensordict_data.get(("next", "agent_blue", "episode_reward"))[done].mean().item()
    )
    episode_reward_mean_list.append(episode_reward_mean)

    episode_reward_mean2 = (
        tensordict_data.get(("next", "agent_red", "episode_reward"))[done].mean().item()
    )
    episode_reward_mean_list2.append(episode_reward_mean2)
    pbar.set_description(f"episode_reward_mean_blue/red = {episode_reward_mean},{episode_reward_mean2}", refresh=False)
    pbar.update()


# =============training loop finished, now rollout ==========================

# torch.save(policy, "PPO_navigation_policy.pt")
#blue_policy = torch.load("PPO_navigation_policy.pt", weights_only= False)

new_combined_policy = TensorDictSequential(
    blue_policy,
    policy2,
)

torch.save(policy2, "policy_red.pt")
torch.save(blue_policy, "policy_blue.pt")

#print(type(new_policy))
new_combined_policy.eval()



plt.plot(episode_reward_mean_list)
plt.xlabel("Training iterations")
plt.ylabel("Reward")
plt.title("Episode reward mean blue")
plt.show()


plt.plot(episode_reward_mean_list2)
plt.xlabel("Training iterations")
plt.ylabel("Reward")
plt.title("Episode reward mean red")
plt.show()

i = 0
while(i < 1000):
    with torch.no_grad():
       env.rollout(
           max_steps=max_steps,
           policy=new_combined_policy,
           callback=lambda env, _: env.render(),
           auto_cast_to_device=True,
           break_when_any_done=False,
       )
    i += 1