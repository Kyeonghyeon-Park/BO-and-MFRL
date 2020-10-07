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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Game 생성
world = ShouAndDiTaxiGridGame()
np.random.seed(seed=1234)
start = time.time()

#%% Actor network와 critic network 정의
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


#%% initialization function
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)


#%% Actor network와 critic network 생성 및 initialization
actor_layer = [2, 32, 16, 8, 4]
critic_layer = [4, 64, 32, 16, 1]

actor = Actor(actor_layer)
critic = Critic(critic_layer)

actor.apply(init_weights)
critic.apply(init_weights)

#%% initial target network 생성 (나중에 10 episode마다 update)
actor_target = copy.deepcopy(actor)
critic_target = copy.deepcopy(critic)

#%% parameter setting
LEARNING_RATE = 0.001
optimizerA = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
optimizerC = optim.Adam(critic.parameters(), lr=LEARNING_RATE)
discount_factor = 1
designer_alpha = 0.5
epsilon = 0.5

buffer = []  # initialize replay buffer B, [o, a, r, a_bar, next_o]
K = 4

obj_weight = 3/5
ORR_train = []  # indicator
OSC_train = []  # indicator
avg_reward_train = []  # indicator
obj_ftn_train = []

ORR_test = []  # indicator
OSC_test = []  # indicator
avg_reward_test = []  # indicator
obj_ftn_test = []


#%%
def get_action_dist(actor_network, observation):
    actor_input = torch.FloatTensor(observation).unsqueeze(0)
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


#%% Actor network와 critic network update # Q. 여기에 input으로 actor, critic 들어갔는데 내부에서 업데이트가 되는가?
def train():
    global buffer
    # dropout이나 batchnorm과 같이 traning과 evaluation이 다를 때 유용. 이 모델에선 쓸모 없음
    # actor.train()
    # critic.train()

    world.initialize_game(random_grid=True)
    global_time = 0
    overall_fare = np.array([0, 0], 'float')

    while global_time is not world.max_episode_time:

        available_agent = world.get_available_agent(global_time)
        exploration = np.random.rand(1)[0]
        joint_action = []  # available agent들의 joint action임
        for agent_id in available_agent:
            available_action_set = world.get_available_action(agent_id)
            if exploration < epsilon:
                random_action = np.random.choice(available_action_set)
                joint_action.append(random_action)
            else:
                # dim 2인 input을 받아서 unsqueeze 필요?
                action_dist = get_action_dist(actor_target, world.joint_observation[agent_id])
                action = action_dist.sample()
                # 여기도 문제 생길 수도
                joint_action.append(action.item())

        # step 후 replay buffer B에 (o_i, a_i, r_i, a_i_bar, o_i_prime) 추가
        if len(available_agent) != 0:
            buffer, overall_fare = world.step(available_agent, joint_action, designer_alpha, buffer, overall_fare, train=True)
            buffer = buffer[-2000:]
        global_time += 1

    # Order response rate / do not consider no demand case in the game
    total_request = world.demand[:, 3].shape[0]
    fulfilled_request = np.sum(world.demand[:, 3])
    ORR_train.append(fulfilled_request / total_request)

    # Overall service charge ratio
    if overall_fare[1] != 0:
        OSC_train.append(overall_fare[0] / overall_fare[1])
    else:
        OSC_train.append(0)

    # Average reward of all agents
    avg_reward_train.append((overall_fare[1] - overall_fare[0]) / world.number_of_agents)
    obj_ftn_train.append(obj_weight * ORR_train[-1] + (1 - obj_weight) * (1 - OSC_train[-1]))

    # buffer에서 sample하여 update / 여기서부터 tensor 정의되어야 할 듯?
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
    #######################
    optimizerA.zero_grad()
    optimizerC.zero_grad()
    actor_loss.backward()
    critic_loss.backward()
    optimizerA.step()
    optimizerC.step()


#%% mean action sampling을 위한 part
mean_action_sample_number = 5


def get_location_agent_number_and_prob(joint_observation, time):
    agent_num = []
    action_dist_set = []
    for loc in range(4):
        agent_num.append(np.sum((joint_observation[:, 0] == loc) & joint_observation[:, 1] == time))
        action_dist = get_action_dist(actor_target, [loc, time])
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

        critic_input = torch.FloatTensor([observation[0], observation[1], action, mean_action_sample]).unsqueeze(0)

        q_observation_action = q_observation_action + critic_target(critic_input) / sample_number

    return q_observation_action


#%% loss 계산
def calculate_actor_loss(sample, agent_id):
    observation = sample[0][agent_id]
    action = sample[1][agent_id]
    agent_num, action_dist_set = get_location_agent_number_and_prob(sample[0], observation[1])
    q_observation = get_q_expectation_over_mean_action(observation, action, agent_num, action_dist_set, mean_action_sample_number)

    v_observation = 0

    expected_action_dist = get_action_dist(actor_target, observation)

    available_action_set = world.get_available_action_from_location(observation[0])
    for expected_action in available_action_set:
        expected_q = get_q_expectation_over_mean_action(observation, expected_action, agent_num, action_dist_set,
                                                           mean_action_sample_number)
        v_observation = v_observation + expected_action_dist.probs[0][expected_action] * expected_q

    q_observation = q_observation.detach()  # tensor를 tensor에 넣어서 생기는 문제 나올 예정
    v_observation = v_observation.detach()
    action = torch.tensor(action)

    action_dist = get_action_dist(actor, observation)
    actor_loss = - (q_observation - v_observation) * action_dist.log_prob(action)

    return actor_loss


def calculate_critic_loss(sample, agent_id):
    observation = sample[0][agent_id]
    action = sample[1][agent_id]
    reward = sample[2][agent_id]
    mean_action = sample[3][agent_id]
    next_observation = sample[4][agent_id]
    # o_i'에서의 available action

    if next_observation[1] != world.max_episode_time:
        available_action_set = world.get_available_action_from_location(next_observation[0])

        q_next_observation = []
        # next_joint_observation에서 각 location에 몇명씩 있는지
        agent_num, action_dist_set = get_location_agent_number_and_prob(sample[4], next_observation[1])

        # available action의 location으로 가려는 agent가 몇명이 되는지 sampling
        for available_action in available_action_set:
            q_next_observation_action = get_q_expectation_over_mean_action(next_observation, available_action,
                                                                           agent_num, action_dist_set, mean_action_sample_number)
            q_next_observation.append(q_next_observation_action)
        max_q_next_observation = (np.max(q_next_observation)).clone().detach()
    else:
        max_q_next_observation = 0

    critic_input = torch.FloatTensor([observation[0], observation[1], action, mean_action]).unsqueeze(0)
    reward = torch.tensor(reward).detach()

    # sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True) rather than torch.tensor(sourceTensor)
    critic_loss = reward + discount_factor * max_q_next_observation - critic(critic_input)

    return critic_loss


#%% Trained actor network로 실제 case에 대해 실행
def evaluate():

    global buffer
    world.initialize_game(random_grid=False)
    global_time = 0
    overall_fare = np.array([0, 0], 'float')

    while global_time is not world.max_episode_time:
        available_agent = world.get_available_agent(global_time)
        joint_action = []  # available agent들의 joint action임
        # print(f"global time : {global_time}")
        for agent_id in available_agent:
            action_dist = get_action_dist(actor, world.joint_observation[agent_id])
            if global_time == 0 and agent_id in [0, len(available_agent) - 1]:
                print(agent_id, action_dist.probs)
            # action = torch.argmax(action_dist.probs, dim=-1)
            action = action_dist.sample()
            # 이 부분에서 runtime error 발생
            joint_action.append(action.item())
        if len(available_agent) != 0:
            buffer, overall_fare = world.step(available_agent, joint_action, designer_alpha, buffer, overall_fare, train=False)
        global_time += 1

    # global ORR_test  # 굳이 할 필요는 없는듯. 여기서 실행하면.
    # global OSC_test
    # global avg_reward_test
    # Order response rate / do not consider no demand case in the game
    total_request = world.demand[:, 3].shape[0]
    fulfilled_request = np.sum(world.demand[:, 3])
    ORR_test.append(fulfilled_request / total_request)

    # Overall service charge ratio
    if overall_fare[1] != 0:
        # print(overall_fare)
        OSC_test.append(overall_fare[0] / overall_fare[1])
    else:
        OSC_test.append(0)

    # Average reward of all agents
    avg_reward_test.append((overall_fare[1] - overall_fare[0]) / world.number_of_agents)

    obj_ftn_test.append(obj_weight * ORR_test[-1] + (1 - obj_weight) * (1 - OSC_test[-1]))


#%%
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


#%% print updated Q
def print_updated_Q():
    print("Q at (#1, 0)")
    for action in range(4):
        Q =[]
        for mean_action in np.arange(0.1, 2.1, 0.1):
            Q_value = critic(torch.FloatTensor([1, 0, action, mean_action]).unsqueeze(0))
            Q.append(Q_value.item())
        Q = np.array(Q)
        np.set_printoptions(precision=2, linewidth=np.inf)
        print(Q)
    print("Q at (#2, 0)")
    for action in range(4):
        Q = []
        for mean_action in np.arange(0.1, 2.1, 0.1):
            Q_value = critic(torch.FloatTensor([2, 0, action, mean_action]).unsqueeze(0))
            Q.append(Q_value.item())
        Q = np.array(Q)
        # np.set_printoptions(precision=2, linewidth=np.inf)
        print(Q)





#%%
max_episode_number = 250
episode = 0
update_period = 10
draw_period = 5

for episode in range(max_episode_number):
# while episode is not max_episode_number:
    train()
    with torch.no_grad():
        print_updated_Q()
        evaluate()

    if (episode + 1) % update_period == 0:
        actor_target = copy.deepcopy(actor)
        critic_target = copy.deepcopy(critic)
        epsilon = np.max([epsilon - 0.05, 0.01])
        # LEARNING_RATE = np.max([LEARNING_RATE - 0.00005, 0.0001])
        LEARNING_RATE = 0.7 * LEARNING_RATE
    if (episode + 1) % draw_period == 0:
        print(f"| Episode : {episode:4} | total time : {time.time() - start:5.2f} |")
        print(f"| train ORR : {ORR_train[episode]:5.2f} | train OSC : {OSC_train[episode]:5.2f} |"
              f" train Obj : {obj_ftn_train[episode]:5.2f} | train avg reward : {avg_reward_train[episode]:5.2f} |")
        print(f"|  test ORR : {ORR_test[episode]:5.2f} |  test OSC : {OSC_test[episode]:5.2f} |"
              f"  test Obj : {obj_ftn_test[episode]:5.2f} |  test avg reward : {avg_reward_test[episode]:5.2f} |")
        draw_plt()
#