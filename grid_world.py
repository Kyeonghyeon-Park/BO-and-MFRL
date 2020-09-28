import numpy as np
import copy
import random


class ShouAndDiTaxiGridGame:
    # Define the taxi grid world setting
    def __init__(self):
        self.grid_size = 2
        self.max_episode_time = 3  # maximum global time
        self.number_of_grids = self.grid_size * self.grid_size
        self.fare = np.zeros((self.number_of_grids, self.number_of_grids))  # fare of demand (O-D)
        self.travel_time = np.ones((self.number_of_grids, self.number_of_grids), int) # travel time between grids
        self.number_of_agents = 100
        self.demand = np.empty((0, 4))  # (time, origin, destination, fulfilled)
        self.joint_observation = np.zeros((self.number_of_agents, 2), int)  # (location, current time)
        # self.joint_action = np.zeros((self.number_of_agents, 3), int)

    # Initialize(reset) the joint observation and the initial position (joint observation = state)
    # joint observation's row = each agent's observation : (location, agent's current time)
    def initialize_joint_observation(self, random_grid):
        ########
        # training 시에는 random starting grid, test 시 given starting grid? 뒤 experiment로 확인 필요
        ########
        self.joint_observation = np.zeros((self.number_of_agents, 2), int)
        if random_grid:

            for i in range(self.number_of_agents):
                self.joint_observation[i][0] = np.random.randint(4)
        else:
            for i in range(self.number_of_agents):
                if i < self.number_of_agents/2:
                    self.joint_observation[i][0] = 1
                else:
                    self.joint_observation[i][0] = 2

    # Initialize the demand / demand's row :  (time, origin, destination, fulfilled or not)
    def initialize_demand(self):
        total_demand = []
        for i in range(int(self.number_of_agents/2)):
            total_demand.append([1, 3, 1, 0])
        for j in range(int(self.number_of_agents/5)):
            total_demand.append([1, 0, 2, 0])
        self.demand = np.asarray(total_demand)

    # Initialize the fare structure (Origin-Destination)
    def initialize_fare(self):
        self.fare[3, 1] = 10
        self.fare[0, 2] = 4.9

    # Initialize the travel time matrix (Origin-Destination) / at least 1
    def initialize_travel_time(self):
        for i in range(self.number_of_grids):
            self.travel_time[i, 3-i] = 2

    # Initialize the game
    def initialize_game(self, random_grid):
        self.initialize_joint_observation(random_grid)
        self.initialize_demand()
        self.initialize_fare()
        self.initialize_travel_time()

    # Get available agents whose current time is same as the global time
    # [available_agent_id, available_agent_id, ...]
    def get_available_agent(self, global_time):
        ###############
        available_agent = np.argwhere(self.joint_observation[:, 1] == global_time)[:, 0]
        ###############
        return available_agent  # numpy.ndarray

    # Get available actions for each agent / action : (0 : stay, 1 : up, 2 : left, 3 : down, 4 : right)
    def get_available_action(self, agent_id):
        if self.joint_observation[agent_id][0] == 0:
            available_action_set = [0, 3, 4]
        elif self.joint_observation[agent_id][0] == 1:
            available_action_set = [0, 2, 3]
        elif self.joint_observation[agent_id][0] == 2:
            available_action_set = [0, 1, 4]
        else:
            available_action_set = [0, 1, 2]

        return np.asarray(available_action_set, int)  # numpy.ndarray

    # # Get random joint action of available agents (나중에 actor network 구현하면 안 쓰일 예정)
    # def get_random_joint_action(self, available_agent):
    #     random_joint_action = []
    #     for agent_id in available_agent:
    #         available_action_set = self.get_available_action(agent_id)
    #         random_action = available_action_set[np.random.randint(0, available_action_set.shape[0])]
    #         random_joint_action.append(random_action)
    #
    #     return np.asarray(random_joint_action)  # numpy.ndarray

    # individual agent에 대해 current observation과 action으로 move (before order matching)
    # 원래는 travel time 해서 해야하지만 time = 1이므로 이렇게 진행
    def move_agent(self, observation, action):
        if observation[0] == 0:
            if action in [0, 1, 2]:
                temp_observation = observation + np.array([0, 1])
            elif action == 3:
                temp_observation = observation + np.array([2, 1])
            else:  # action == 4
                temp_observation = observation + np.array([1, 1])
        elif observation[0] == 1:
            if action in [0, 1, 4]:
                temp_observation = observation + np.array([0, 1])
            elif action == 3:
                temp_observation = observation + np.array([2, 1])
            else:  # action == 2
                temp_observation = observation + np.array([-1, 1])
        elif observation[0] == 2:
            if action in [0, 2, 3]:
                temp_observation = observation + np.array([0, 1])
            elif action == 1:
                temp_observation = observation + np.array([-2, 1])
            else:  # action == 4
                temp_observation = observation + np.array([1, 1])
        else:  # observation[0] == 3
            if action in [0, 3, 4]:
                temp_observation = observation + np.array([0, 1])
            elif action == 1:
                temp_observation = observation + np.array([-2, 1])
            else:  # action == 2
                temp_observation = observation + np.array([-1, 1])

        return temp_observation

    # available agent와 그것들의 joint action으로 move (before order matching)
    def move_available_agent(self, available_agent, joint_action):
        temp_joint_observation = []
        for agent in range(available_agent.shape[0]):
            agent_id = available_agent[agent]
            observation = self.joint_observation[agent_id]
            action = joint_action[agent]
            temp_joint_observation.append(self.move_agent(observation, action))

        return np.asarray(temp_joint_observation)

    # move 후 도착한 location에서 발생한 demand와 agent들 matching
    def match_agent_and_demand(self, available_agent, temp_joint_observation):
        temp_time = temp_joint_observation[0, 1]
        order_matching = []
        for location in range(4):
            local_demand = np.argwhere((self.demand[:, 0] == temp_time) & (self.demand[:, 1] == location))[:, 0]
            np.random.shuffle(local_demand)
            local_supply = available_agent[temp_joint_observation[:, 0] == location]
            np.random.shuffle(local_supply)
            if local_demand.shape[0] == 0:
                continue
            for demand_id in local_demand:
                if local_supply.shape[0] != 0:
                    order_matching.append([local_supply[0], demand_id])
                    local_supply = local_supply[1:]
                    self.demand[demand_id, 3] = 1
                else:
                    break

        return np.asarray(order_matching)

    # (matching된 agent에 대해) demand에 의한 move
    def move_with_demand(self, observation, order):
        destination = order[2]
        arrival_time = observation[1] + self.travel_time[order[1], order[2]]
        observation = np.array([destination, arrival_time])

        return observation

    # available agents들의 move (before matching) 후 location의 demand to supply ratio
    def get_demand_to_supply_ratio(self, temp_joint_observation):
        temp_time = temp_joint_observation[0, 1]
        DS_ratio = []
        for location in range(4):
            local_demand = np.argwhere((self.demand[:, 0] == temp_time) & (self.demand[:, 1] == location))[:, 0]
            num_agents = np.sum(temp_joint_observation[:, 0] == location)
            num_orders = local_demand.shape[0]
            if num_agents == 0:
                DS_ratio.append(float("inf"))
            else:
                DS_ratio.append(num_orders / num_agents)

        return np.asarray(DS_ratio)

    # available agents들의 move (before matching) 후 location의 service charge ratio
    def get_service_charge_ratio(self, designer_alpha, DS_ratio):
        SC_ratio = []
        for location in range(4):
            if DS_ratio[location] <= 1:
                SC_ratio.append(designer_alpha * (1 - DS_ratio[location]))
            else:
                SC_ratio.append(0)

        return np.asarray(SC_ratio)

    # Given available agent와 joint action에 대해, state transition을 진행하고 buffer에 저장하는 function
    # action에 대한 travel time이 무조건 1이라는 전제 하에 가정된 모델
    def step(self, available_agent, joint_action, designer_alpha, buffer, overall_fare, train=True):
        # buffer와 overall fare는 world 외부에서 정의됨

        temp_joint_observation = self.move_available_agent(available_agent, joint_action)
        order_matching = self.match_agent_and_demand(available_agent, temp_joint_observation)
        DS_ratio = self.get_demand_to_supply_ratio(temp_joint_observation)
        SC_ratio = self.get_service_charge_ratio(designer_alpha, DS_ratio)

        for agent in range(available_agent.shape[0]):
            agent_id = available_agent[agent]
            current_observation = copy.deepcopy(self.joint_observation[agent_id])
            action = joint_action[agent]
            temp_observation = temp_joint_observation[agent]
            if len(order_matching) != 0 and agent_id in order_matching[:, 0]:
                order_id = order_matching[order_matching[:, 0] == agent_id][0, 1]
                order = self.demand[order_id]
                next_observation = self.move_with_demand(temp_observation, order)
                reward = self.fare[order[1], order[2]] * (1 - SC_ratio[temp_observation[0]])
                overall_fare += np.array([self.fare[order[1], order[2]] * SC_ratio[temp_observation[0]],
                                          self.fare[order[1], order[2]]])
            else:
                next_observation = temp_observation
                reward = 0
            mean_action = DS_ratio[temp_observation[0]]
            if train:
                buffer.append([current_observation, action, reward, mean_action, next_observation])
            self.joint_observation[available_agent[agent]] = next_observation

        return buffer, overall_fare
