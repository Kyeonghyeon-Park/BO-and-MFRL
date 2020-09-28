import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
import copy
import time

from grid_world import ShouAndDiTaxiGridGame

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
actor_layer = [2, 32, 16, 8, 5]
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

buffer = []  # initialize replay buffer B, [o_i, a_i, r_i, a_i_bar, next_o_i]
K = 256

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
        available_action_torch = torch.tensor([1, 0, 0, 1, 1])
    elif observation[0] == 1:
        available_action_torch = torch.tensor([1, 0, 1, 1, 0])
    elif observation[0] == 2:
        available_action_torch = torch.tensor([1, 1, 0, 0, 1])
    else:
        available_action_torch = torch.tensor([1, 1, 1, 0, 0])
    action_dist = distributions.Categorical(torch.mul(action_prob, available_action_torch))

    return action_dist


#%% Actor network와 critic network update # Q. 여기에 input으로 actor, critic 들어갔는데 내부에서 업데이트가 되는가?
def train():
    # dropout이나 batchnorm과 같이 traning과 evaluation이 다를 때 유용. 이 모델에선 쓸모 없음
    global buffer
    actor.train()
    critic.train()

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
                joint_action.append(action.numpy())

        # step 후 replay buffer B에 (o_i, a_i, r_i, a_i_bar, o_i_prime) 추가
        # global assignment 필요할까?
        if len(available_agent) != 0:
            buffer, overall_fare = world.step(available_agent, joint_action, designer_alpha, buffer, overall_fare, train=True)
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
    sample_id = np.random.choice(len(buffer), K, replace=True)
    actor_loss = torch.tensor([[0]])
    critic_loss = torch.tensor([[0]])

    for i in sample_id:
        sample = buffer[i]
        actor_loss = actor_loss + calculate_actor_loss(sample)
        critic_loss = critic_loss + calculate_critic_loss(sample)**2

    actor_loss = actor_loss / K
    critic_loss = critic_loss / K

    optimizerA.zero_grad()
    optimizerC.zero_grad()
    actor_loss.backward()
    critic_loss.backward()
    optimizerA.step()
    optimizerC.step()


def calculate_actor_loss(sample):
    observation = sample[0]
    action = sample[1]
    local_demand = np.argwhere((world.demand[:, 0] == observation[1]) & (world.demand[:, 1] == observation[0]))[:, 0]
    local_demand_num = local_demand.shape[0]
    available_mean_action_set = local_demand_num / (np.arange(world.number_of_agents) + 1)
    q_observation = 0
    for available_mean_action in available_mean_action_set:
        critic_input = torch.FloatTensor([observation[0], observation[1], action, available_mean_action]).unsqueeze(0)
        q_observation = q_observation + critic_target(critic_input) / world.number_of_agents

    v_observation = 0

    action_dist = get_action_dist(actor, observation)
    expected_action_dist = get_action_dist(actor_target, observation)

    for expected_action in range(5):
        expected_q = 0
        for available_mean_action in available_mean_action_set:
            critic_input = torch.FloatTensor([observation[0], observation[1], expected_action, available_mean_action]).unsqueeze(0)
            expected_q = expected_q + critic_target(critic_input) / world.number_of_agents
        v_observation = v_observation + expected_action_dist.probs[0][expected_action] * expected_q
    q_observation = q_observation.detach()  # tensor를 tensor에 넣어서 생기는 문제 나올 예정
    v_observation = v_observation.detach()
    action = torch.tensor(action)
    actor_loss = - (q_observation - v_observation) * action_dist.log_prob(action)

    return actor_loss


def calculate_critic_loss(sample):
    observation = sample[0]
    action = sample[1]
    reward = sample[2]
    mean_action = sample[3]
    next_observation = sample[4]
    # o_i'에서의 available action
    if next_observation[0] == 0:
        available_action_set = [0, 3, 4]
    elif next_observation[0] == 1:
        available_action_set = [0, 2, 3]
    elif next_observation[0] == 2:
        available_action_set = [0, 1, 4]
    else:
        available_action_set = [0, 1, 2]

    q_next_observation = []
    for available_action in available_action_set:
        q_next_observation_action = 0
        temp_observation = world.move_agent(next_observation, available_action)
        temp_location = temp_observation[0]
        temp_time = temp_observation[1]
        local_demand = np.argwhere((world.demand[:, 0] == temp_time) & (world.demand[:, 1] == temp_location))[:, 0]
        local_demand_num = local_demand.shape[0]
        available_mean_action_set = local_demand_num / (np.arange(world.number_of_agents) + 1)
        # mean action에 대한 expectation 구하기 위함
        for available_mean_action in available_mean_action_set:
            critic_input = torch.FloatTensor([temp_location, temp_time, available_action, available_mean_action]).unsqueeze(0)
            q_next_observation_action = q_next_observation_action + critic_target(critic_input) / world.number_of_agents

        q_next_observation.append(q_next_observation_action)

    critic_input = torch.FloatTensor([observation[0], observation[0], action, mean_action]).unsqueeze(0)
    reward = torch.tensor(reward).detach()
    max_q_next_observation = torch.tensor(np.max(q_next_observation)).detach()
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
            joint_action.append(action.numpy())
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


#%%
max_episode_number = 250
episode = 0
update_period = 10

for episode in range(max_episode_number):
# while episode is not max_episode_number:
    train()
    evaluate()

    if (episode + 1) % update_period == 0:
        actor_target = copy.deepcopy(actor)
        critic_target = copy.deepcopy(critic)
        epsilon = np.max([epsilon - 0.05, 0.01])
        # LEARNING_RATE = np.max([LEARNING_RATE - 0.00005, 0.0001])
        LEARNING_RATE = 0.7 * LEARNING_RATE
        print(f"| Episode : {episode:4} | total time : {time.time() - start:5.2f} |")
        print(f"| train ORR : {ORR_train[episode]:5.2f} | train OSC : {OSC_train[episode]:5.2f} |"
              f" train Obj : {obj_ftn_train[episode]:5.2f} | train avg reward : {avg_reward_train[episode]:5.2f} |")
        print(f"|  test ORR : {ORR_test[episode]:5.2f} |  test OSC : {OSC_test[episode]:5.2f} |"
              f"  test Obj : {obj_ftn_test[episode]:5.2f} |  test avg reward : {avg_reward_test[episode]:5.2f} |")
        draw_plt()
#