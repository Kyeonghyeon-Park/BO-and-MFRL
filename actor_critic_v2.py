import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
import copy
import time

from grid_world_v2 import ShouAndDiTaxiGridGame

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% Generate the game
world = ShouAndDiTaxiGridGame()
np.random.seed(seed=1234)
start = time.time()


# %% Define the actor network and the critic network
class Actor(nn.Module):
    def __init__(self, net_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = len(net_size) - 1

        for i in range(self.num_layers):
            fc_i = nn.Linear(net_size[i], net_size[i + 1])
            self.layers.append(fc_i)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)

        x = F.softmax(x, dim=-1)

        return x


class Critic(nn.Module):
    def __init__(self, net_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = len(net_size) - 1

        for i in range(self.num_layers):
            fc_i = nn.Linear(net_size[i], net_size[i + 1])
            self.layers.append(fc_i)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)

        return x


# %% Define the initialization function
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)


# %% Generate the actor network and critic network and do initialization
actor_layer = [7, 32, 16, 8, 4]
critic_layer = [12, 64, 32, 16, 1]

actor = Actor(actor_layer)
critic = Critic(critic_layer)

actor.apply(init_weights)
critic.apply(init_weights)

# %% Generate the initial target network of actor and critic (update per 10 episodes)
actor_target = copy.deepcopy(actor)
critic_target = copy.deepcopy(critic)

# %% Parameter setting
LEARNING_RATE = 0.001
optimizerA = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
optimizerC = optim.Adam(critic.parameters(), lr=LEARNING_RATE)
discount_factor = 1
designer_alpha = 1.5
epsilon = 0.5

buffer = []  # initialize replay buffer B, [o, a, r, a_bar, next_o]
buffer_max_size = 50
K = 4
mean_action_sample_number = 5

obj_weight = 3 / 5

# outcome of train episode
ORR_train = []
OSC_train = []
avg_reward_train = []
obj_ftn_train = []

# outcome of test episode
ORR_test = []
OSC_test = []
avg_reward_test = []
obj_ftn_test = []


# %% Define the actor and critic input generation(conversion) function
def get_actor_input(observation):
    # [0, 1, 2, 3 : location / 4, 5, 6 : time]
    actor_input_numpy = np.zeros(7)
    location = observation[0]
    current_time = observation[1]
    actor_input_numpy[location] = 1
    if current_time > 2:
        actor_input_numpy[6] = 1
    else:
        actor_input_numpy[4 + current_time] = 1
    # actor_input_numpy[4] = current_time
    actor_input = torch.FloatTensor(actor_input_numpy).unsqueeze(0)

    return actor_input


def get_critic_input(observation, action, mean_action):
    # [0, 1, 2, 3 : location / 4, 5, 6 : time / 7, 8, 9, 10 : action / 11 : mean action]
    critic_input_numpy = np.zeros(12)
    location = observation[0]
    current_time = observation[1]
    critic_input_numpy[location] = 1
    if current_time > 2:
        critic_input_numpy[6] = 1
    else:
        critic_input_numpy[4 + current_time] = 1
    critic_input_numpy[4] = current_time
    critic_input_numpy[7 + action] = 1
    critic_input_numpy[11] = np.min([mean_action, 1])
    critic_input = torch.FloatTensor(critic_input_numpy).unsqueeze(0)

    return critic_input


# %% Define the action distribution generation function
def get_action_dist(actor_network, observation):
    actor_input = get_actor_input(observation)
    action_prob = actor_network(actor_input)
    if observation[0] == 0:
        available_action_torch = torch.tensor([1, 1, 1, 0])
    elif observation[0] == 1:
        available_action_torch = torch.tensor([1, 1, 0, 1])
    elif observation[0] == 2:
        available_action_torch = torch.tensor([1, 0, 1, 1])
    else:
        available_action_torch = torch.tensor([0, 1, 1, 1])
    action_dist = distributions.Categorical(torch.mul(action_prob, available_action_torch))

    return action_dist


# %% Define the outcome function
def get_outcome(ORR_list, OSC_list, avg_reward_list, obj_ftn_list, demand, overall_fare):
    # Order response rate / do not consider no demand case in the game
    total_request = demand[:, 3].shape[0]
    fulfilled_request = np.sum(demand[:, 3])
    ORR_list.append(fulfilled_request / total_request)

    # Overall service charge ratio
    if overall_fare[1] != 0:
        OSC_list.append(overall_fare[0] / overall_fare[1])
    else:
        OSC_list.append(0)

    # Average reward of all agents
    avg_reward_list.append((overall_fare[1] - overall_fare[0]) / world.number_of_agents)
    obj_ftn_list.append(obj_weight * ORR_list[-1] + (1 - obj_weight) * (1 - OSC_list[-1]))


# %% Train (generate samples and update networks)
def train():
    global buffer
    world.initialize_game(random_grid=False)
    global_time = 0
    overall_fare = np.array([0, 0], 'float')

    while global_time is not world.max_episode_time:

        available_agent = world.get_available_agent(global_time)

        joint_action = []  # available agents' joint action
        for agent_id in available_agent:
            available_action_set = world.get_available_action(agent_id)
            exploration = np.random.rand(1)[0]
            if exploration < epsilon:
                random_action = np.random.choice(available_action_set)
                joint_action.append(random_action.item())
            else:
                action_dist = get_action_dist(actor_target, world.joint_observation[agent_id])
                # 이 부분에서 특정 확률이 너무 작아지면 runtime error 발생
                action = action_dist.sample()
                joint_action.append(action.item())

        # step 후 replay buffer B에 (o, a, r, a_bar, o_prime) 추가
        if len(available_agent) != 0:
            buffer, overall_fare = world.step(available_agent, joint_action, designer_alpha, buffer, overall_fare,
                                              train=True)
            buffer = buffer[-buffer_max_size:]
        global_time += 1
    # Get outcome of train episode
    get_outcome(ORR_train, OSC_train, avg_reward_train, obj_ftn_train, world.demand, overall_fare)

    # Update the network
    sample_id_list = np.random.choice(len(buffer), K, replace=True)
    actor_loss = torch.tensor([[0]])
    critic_loss = torch.tensor([[0]])

    update_count = 0

    for sample_id in sample_id_list:
        sample = buffer[sample_id]
        for agent_id in range(world.number_of_agents):
            if sample[1][agent_id] is not None:
                actor_loss = actor_loss + calculate_actor_loss(sample, agent_id)
                critic_loss = critic_loss + calculate_critic_loss(sample, agent_id) ** 2
                update_count += 1
            else:
                continue

    # sample 내에 available agent 수만큼 각각 loss 더하므로 K가 아님
    actor_loss = actor_loss / update_count
    critic_loss = critic_loss / update_count

    optimizerA.zero_grad()
    optimizerC.zero_grad()
    actor_loss.backward()
    critic_loss.backward()
    optimizerA.step()
    optimizerC.step()


# %% Define the function for expectation over mean action using sampling
def get_location_agent_number_and_prob(joint_observation, current_time):
    agent_num = []
    action_dist_set = []
    for loc in range(4):
        agent_num.append(np.sum((joint_observation[:, 0] == loc) & (joint_observation[:, 1] == current_time)))
        action_dist = get_action_dist(actor_target, [loc, current_time])
        action_dist_set.append(action_dist)

    return agent_num, action_dist_set


def get_q_expectation_over_mean_action(observation, action, agent_num, action_dist_set, mean_action_sample_number):
    q_observation_action = 0

    temp_observation = world.move_agent(observation, action)
    temp_location = temp_observation[0]
    temp_time = temp_observation[1]
    local_demand = np.argwhere((world.demand[:, 0] == temp_time) & (world.demand[:, 1] == temp_location))[:, 0]
    local_demand_num = local_demand.shape[0]

    sample_number = mean_action_sample_number

    for expectation_over_mean_action in range(sample_number):
        loc_agent_num = 0
        for loc in range(4):
            num = agent_num[loc]
            prob = action_dist_set[loc].probs[0][action].detach().numpy()
            loc_agent_num = loc_agent_num + np.random.binomial(num, prob)
        mean_action_sample = local_demand_num / (loc_agent_num + 1)

        critic_input = get_critic_input(observation, action, mean_action_sample)

        q_observation_action = q_observation_action + critic_target(critic_input) / sample_number

    return q_observation_action


# %% Define loss function for one sample and agent id
def calculate_actor_loss(sample, agent_id):
    observation = sample[0][agent_id]
    action = sample[1][agent_id]
    with torch.no_grad():
        agent_num, action_dist_set = get_location_agent_number_and_prob(sample[0], observation[1])
        q_observation = get_q_expectation_over_mean_action(observation, action, agent_num, action_dist_set,
                                                           mean_action_sample_number)

        v_observation = 0

        expected_action_dist = get_action_dist(actor_target, observation)

        available_action_set = world.get_available_action_from_location(observation[0])
        for expected_action in available_action_set:
            expected_q = get_q_expectation_over_mean_action(observation, expected_action, agent_num, action_dist_set,
                                                            mean_action_sample_number)
            v_observation = v_observation + expected_action_dist.probs[0][expected_action] * expected_q

    action = torch.tensor(action)

    action_dist = get_action_dist(actor, observation)
    actor_loss = - (q_observation - v_observation) * action_dist.log_prob(action)

    return actor_loss


def calculate_actor_loss_test(sample, agent_id):
    observation = sample[0][agent_id]
    action = sample[1][agent_id]
    reward = sample[2][agent_id]
    next_observation = sample[4][agent_id]
    with torch.no_grad():
        for i in ["observation", "next_observation"]:
            if i == "observation":
                obs = observation
                joint_obs = sample[0]
            else:
                obs = next_observation
                joint_obs = sample[4]
            if obs[1] != world.max_episode_time:
                v_obs = 0
                agent_num, action_dist_set = get_location_agent_number_and_prob(joint_obs, obs[1])
                expected_action_dist = get_action_dist(actor_target, obs)
                available_action_set = world.get_available_action_from_location(obs[0])
                for expected_action in available_action_set:
                    expected_q = get_q_expectation_over_mean_action(obs, expected_action, agent_num, action_dist_set,
                                                                    mean_action_sample_number)
                    v_obs = v_obs + expected_action_dist.probs[0][expected_action] * expected_q
            else:
                v_obs = 0

            if i == "observation":
                v_observation = v_obs
            else:
                v_next_observation = v_obs

    action = torch.tensor(action)
    action_dist = get_action_dist(actor, observation)
    actor_loss = - (reward + v_next_observation - v_observation) * action_dist.log_prob(action)
    return actor_loss


def calculate_critic_loss(sample, agent_id):
    observation = sample[0][agent_id]
    action = sample[1][agent_id]
    reward = sample[2][agent_id]
    mean_action = sample[3][agent_id]
    next_observation = sample[4][agent_id]
    with torch.no_grad():
        if next_observation[1] != world.max_episode_time:
            available_action_set = world.get_available_action_from_location(next_observation[0])

            q_next_observation = []
            # get each location's agent numbers and action distributions from next_joint_observation
            agent_num, action_dist_set = get_location_agent_number_and_prob(sample[4], next_observation[1])

            # available action의 location으로 가려는 agent가 몇명이 되는지 sampling
            for available_action in available_action_set:
                q_next_observation_action = get_q_expectation_over_mean_action(next_observation, available_action,
                                                                               agent_num, action_dist_set,
                                                                               mean_action_sample_number)
                q_next_observation.append(q_next_observation_action)
            # max_q_next_observation = (np.max(q_next_observation)).clone().detach()
            max_q_next_observation = np.max(q_next_observation)
        else:
            max_q_next_observation = 0
        #### temporal test
        max_q_next_observation = 0

        critic_input = get_critic_input(observation, action, mean_action)
    reward = torch.tensor(reward)

    critic_loss = reward + discount_factor * max_q_next_observation - critic(critic_input)

    return critic_loss


# %% Evaluate (using trained actor network)
def evaluate():
    global buffer
    world.initialize_game(random_grid=False)
    global_time = 0
    overall_fare = np.array([0, 0], 'float')

    while global_time is not world.max_episode_time:
        available_agent = world.get_available_agent(global_time)
        joint_action = []
        for agent_id in available_agent:
            action_dist = get_action_dist(actor, world.joint_observation[agent_id])
            if global_time == 0 and agent_id in [0, len(available_agent) - 1]:
                print(agent_id, action_dist.probs)
            action = action_dist.sample()  # 이 부분에서 runtime error 발생
            joint_action.append(action.item())
        if len(available_agent) != 0:
            buffer, overall_fare = world.step(available_agent, joint_action, designer_alpha, buffer, overall_fare,
                                              train=False)
        global_time += 1

    # Get outcome of test episode
    get_outcome(ORR_test, OSC_test, avg_reward_test, obj_ftn_test, world.demand, overall_fare)


# %% Define draw_plt print_updated_Q, print_action_distribution functions
def draw_plt():
    plt.figure(figsize=(16, 14))

    plt.subplot(2, 2, 1)
    plt.plot(avg_reward_train, label='Avg reward train')
    plt.ylim([0, 6])
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(avg_reward_test, label='Avg reward test')
    plt.ylim([0, 6])
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(ORR_train, label='ORR train')
    plt.plot(OSC_train, label='OSC train')
    plt.plot(obj_ftn_train, label='Obj train')
    plt.ylim([0, 1.1])
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(ORR_test, label='ORR test')
    plt.plot(OSC_test, label='OSC test')
    plt.plot(obj_ftn_test, label='Obj test')
    plt.ylim([0, 1.1])
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()

    plt.show()


def print_updated_Q():
    np.set_printoptions(precision=2, linewidth=np.inf)
    for location in range(4):
        for agent_time in range(3):
            print("Q at (#", location, ", ", agent_time, ")")
            for action in range(4):
                Q = []
                for mean_action in np.arange(0.1, 1.1, 0.1):
                    critic_input = get_critic_input([location, agent_time], action, mean_action)
                    Q_value = critic(critic_input)
                    Q.append(Q_value.item())
                Q = np.array(Q)
                print(Q)


def print_action_distribution():
    for location in range(4):
        for agent_time in range(3):
            action_dist = get_action_dist(actor, [location, agent_time])
            print("Action distribution at (#", location, ", ", agent_time, ") : ", action_dist.probs[0].numpy())


# %% Run episode
max_episode_number = 250
episode = 0
update_period = 10
draw_period = 5
actor_network_save = []
critic_network_save = []
for episode in range(max_episode_number):
    # while episode is not max_episode_number:
    train()

    actor_network_save.append(actor)
    critic_network_save.append(critic)
    if episode == 100:
        print(100)
    with torch.no_grad():
        evaluate()

    if (episode + 1) % update_period == 0:
        actor_target = copy.deepcopy(actor)
        critic_target = copy.deepcopy(critic)
        # epsilon = np.max([epsilon - 0.025, 0.01])
        # LEARNING_RATE = np.max([LEARNING_RATE - 0.00005, 0.0001])
        # LEARNING_RATE = 0.7 * LEARNING_RATE
        # decaying 적용 안되고 있었음
    if (episode + 1) % draw_period == 0:
        print(f"| Episode : {episode:4} | total time : {time.time() - start:5.2f} |")
        print(f"| train ORR : {ORR_train[episode]:5.2f} | train OSC : {OSC_train[episode]:5.2f} |"
              f" train Obj : {obj_ftn_train[episode]:5.2f} | train avg reward : {avg_reward_train[episode]:5.2f} |")
        print(f"|  test ORR : {ORR_test[episode]:5.2f} |  test OSC : {OSC_test[episode]:5.2f} |"
              f"  test Obj : {obj_ftn_test[episode]:5.2f} |  test avg reward : {avg_reward_test[episode]:5.2f} |")
        draw_plt()
        with torch.no_grad():
            print_updated_Q()
            print_action_distribution()
#
