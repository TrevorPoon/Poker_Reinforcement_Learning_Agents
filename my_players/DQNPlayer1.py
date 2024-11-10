from .QLearningPlayer import QLearningPlayer
import random as rand
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

# hyper-parameters
batch_size = 64
learning_rate = 1e-4
gamma = 0.99
exp_replay_size = 10000
epsilon = 0.5
learn_start = 1000
target_net_update_freq = 1000
decay_factor = 0.9999
final_epsilon = 0.1

class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return rand.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Deep Q Network
class DQN(nn.Module):

    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.fc1 = nn.Linear(self.input_shape[0], 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, self.num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DQNPlayer1(QLearningPlayer):

    def __init__(self, model_path, optimizer_path, training):
        """
        State: hole_card, community_card, self.stack, opponent_player.action
        """
        # training device: cpu > cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nb_player = self.player_id = None
        self.loss = 0
        self.episode = 0
        self.declare_memory()
        #self.oponent_action = None
        self.loss = []
        #self.oponent = None

        # hyper-parameter for Deep Q Learning
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.experience_replay_size = exp_replay_size
        self.batch_size = batch_size
        self.learn_start = learn_start
        self.target_net_update_freq = target_net_update_freq
        # training-required game attribute
        self.stack = 100
        self.hole_card = None
        self.model_path = model_path
        self.optimizer_path = optimizer_path
        self.update_count = 0
        self.history = []
        self.training = training
        # declare DQN model
        self.num_actions = 11
        self.num_feats = (46, )
        self.declare_networks()
        try:
            self.policy_net.load_state_dict(torch.load(self.model_path))
        except:
            pass
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net = self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        try:
            self.optimizer.load_state_dict(torch.load(self.optimizer_path))
        except:
            pass
        self.losses = []
        self.sigma_parameter_mag = []
        if self.training:
            self.policy_net.train()
            self.target_net.train()
        else:
            self.policy_net.eval()
            self.target_net.eval()

        self.update_count = 0
        
        self.hand_count = 0
        self.VPIP = 0
        self.last_vpip_action = None
        self.PFR = 0
        self.last_pfr_action = None
        self.three_bet = 0
        self.last_3_bet_action = None
        self.raisesizes = [0.25, 0.33, 0.5, 0.75, 1, 1.25, 1.5]
        self.accumulated_reward = 0
        self.state = []


    def declare_networks(self):
        self.policy_net = DQN(self.num_feats, self.num_actions)
        self.target_net = DQN(self.num_feats, self.num_actions)

    def declare_memory(self):
        self.memory = ExperienceReplayMemory(exp_replay_size)
        return self.memory

    def append_to_replay(self, s, a, r, s_):
        self.memory.push((s, a, r, s_))

    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions = self.memory.sample(self.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1,) + self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device,
                                      dtype=torch.bool)
        try:  # sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device,
                                                 dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values

    def save_sigma_param_magnitudes(self):
        tmp = []
        for name, param in self.policy_net.named_parameters():
            if param.requires_grad:
                if 'sigma' in name:
                    tmp += param.data.cpu().numpy().ravel().tolist()
        if tmp:
            self.sigma_parameter_mag.append(np.mean(np.abs(np.array(tmp))))

    def save_loss(self, loss):
        self.losses.append(loss)

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values = batch_vars

        # estimate
        current_q_values = self.policy_net(batch_state).gather(1, batch_action)

        # target
        with torch.no_grad():
            # To prevent tracking history of gradient
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                max_next_q_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, max_next_action)
            expected_q_values = batch_reward + (self.gamma * max_next_q_values)
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(current_q_values, expected_q_values)
        self.loss = loss.detach().item()
        # print(loss)
        # diff = (expected_q_values - current_q_values)
        # loss = self.huber(diff)
        # loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, episode=0):
        if not self.training:
            return None

        self.append_to_replay(s, a, r, s_)

        if episode < self.learn_start:
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        # self.save_loss(loss.item())
        # self.save_sigma_param_magnitudes()

    def update_target_model(self):
        """
        to use in fix-target
        """
        if self.update_count % self.target_net_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # print("update target: ", self.update_count)
        
        self.epsilon = max(final_epsilon, self.epsilon * decay_factor)

    def get_max_next_state_action(self, next_states):
        return self.target_net(next_states).max(dim=1)[1].view(-1, 1)

    @staticmethod
    def huber(x):
        cond = (x.abs() < 1.0).to(torch.float)
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)

    @staticmethod
    def bce_loss(x, y):
        """
        :return: binary entropy loss between x and y
        """
        _x = 1 / (1 + torch.exp(-x))
        _y = 1 / (1 + torch.exp(-y))
        return -(_y * torch.log(_x) + (1 - _y) * torch.log(1 - _x))

    @staticmethod
    def card_to_int(card):
        """convert card to int, card[0]:suit, card[1]:rank"""
        suit_map = {'H': 0, 'S': 1, 'D': 2, 'C': 3}
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11,
                    'A': 12}
        return suit_map[card[0]] * 13 + rank_map[card[1]]

    def community_card_to_tuple(self, community_card):
        """
        :param community_card: round_state['community_card']
        :return: tuple of int (0..52)
        """
        new_community_card = []
        for i in range(0, len(community_card)):
            new_community_card.append(self.card_to_int(community_card[i]))
        for i in range(0, 5 - len(community_card)):
            # if community card num <5, append 52 to fill out the rest
            new_community_card.append(52)
        return tuple(new_community_card)

    def eps_greedy_policy(self, s, opponent, valid_actions, eps):
        with torch.no_grad():
            X = torch.tensor([s], device=self.device, dtype=torch.float)
            q_values = self.policy_net(X).cpu().numpy().reshape(-1)

            # Create a mask of valid actions
            valid_action_mask = np.ones(self.num_actions, dtype=bool)

            # Adjust mask based on game state
            if opponent['state'] == 'allin' or valid_actions[2]['amount']['max'] == -1:
                valid_action_mask[2:] = False  # Only 'fold' and 'call' are valid
            if valid_actions[1]['amount'] == 0:
                valid_action_mask[0] = False  # Cannot 'fold' when 'check' is available

            # Apply mask to Q-values
            masked_q_values = np.where(valid_action_mask, q_values, -np.inf)

            if np.random.random() >= eps or not self.training:
                action = np.argmax(masked_q_values)
            else:
                valid_actions_indices = np.where(valid_action_mask)[0]
                action = np.random.choice(valid_actions_indices)

            return action

    @staticmethod
    def process_state(s):
        new_s = list(s)
        for i in range(0, 7):
            new_s[i] = new_s[i] / 26.0 - 1
        for i in range(7, 39):
            new_s[i] = (new_s[i] - 100) / 100
        return tuple(new_s)
    
    def get_state(self, community_card, round_state):

        self.state = self.hole_card + community_card 

        # Bet Related
        main_pot = round_state['pot']['main']['amount']
        self.state += (main_pot,)

        player_stacks = tuple(seat['stack'] for seat in round_state['seats'])
        self.state += player_stacks  

        self.state += (round_state['seats'][self.player_id]['stack'], )

        last_amounts = {seat['uuid']: [0, 0, 0, 0] for seat in round_state['seats']}  # 4 rounds per player

        for i, street in enumerate(['preflop', 'flop', 'turn', 'river']):
            actions = round_state['action_histories'].get(street, [])
            for action in actions:
                uuid = action['uuid']
                last_amounts[uuid][i] = action.get('amount', 0)

        for player_id in last_amounts:
            self.state += tuple(last_amounts[player_id])

        # Boolean
        player_statuses = tuple(1 if seat['state'] == 'participating' else 0 for seat in round_state['seats'])
        self.state += player_statuses

        for player in round_state['seats']:
            if player['uuid'] == self.uuid:
                my_position = int(player['name'][-1])
                break

        dealer_position = round_state['dealer_btn']
        player_position = (my_position - dealer_position) % len(round_state['seats'])
        self.state += (player_position, )
        
        return self.state

    def declare_action(self, valid_actions, hole_card, round_state):
        """
        state: hole_card, community_card, self.stack
        """

        # preprocess variable in states
        self.episode += 1
        self.update_count += 1

        hole_card_1 = self.card_to_int(hole_card[0])
        hole_card_2 = self.card_to_int(hole_card[1])
        self.hole_card = (hole_card_1, hole_card_2)
        community_card = self.community_card_to_tuple(round_state['community_card'])
        
        self.state = self.get_state(community_card, round_state)

        self.state = self.process_state(self.state)

        pot_size = round_state['pot']['main']['amount']
        side_pots = round_state['pot'].get('side', [])
        for side_pot in side_pots:
            pot_size += side_pot['amount']

        for action in valid_actions:
            if action['action'] == 'raise':

                min_raise = action['amount']['min']
                max_raise = action['amount']['max']

                for raisesize in self.raisesizes:
            
                    raiseamount = raisesize * pot_size
                    if raiseamount > min_raise and raiseamount < max_raise:
                        action['amount'][raisesize] = raiseamount
                    elif raiseamount <= min_raise: 
                        action['amount'][raisesize] = min_raise
                    elif raiseamount >= max_raise: 
                        action['amount'][raisesize] = max_raise

                amounts = action['amount']
                # Sort the 'min' and 'max'

                min_value = {'min': amounts.pop('min')}
                max_value = {'max': amounts.pop('max')}

                sorted_amount = dict(sorted(amounts.items(), key=lambda item: item[1]))

                final_amount = {**min_value, **sorted_amount, **max_value}
                # Update the original data with sorted values
                action['amount'] = final_amount

        action = self.eps_greedy_policy(self.state, round_state['seats'][(self.player_id + 1) % 6], valid_actions,
                                        self.epsilon)
        
        # record the action
        self.history.append(self.state + (action,))

        if action > 1:
            amount = list(valid_actions[2]["amount"].items())[action - 2][1]
            action = "raise"
        elif action == 1:
            amount = valid_actions[1]["amount"]
            action = "call"
        else:
            amount = 0
            action = "fold"

        print(action, amount)

        if round_state["street"] == 'preflop':

            pre_flop_action = round_state['action_histories']['preflop']

            if action in ["call", "raise"]:

                # VPIP Logic
                if amount > 0:
                    if not self.last_vpip_action:
                        self.VPIP += 1
                        self.last_vpip_action = action
    
                # PFR Logic
                if action == "raise" and not self.last_pfr_action:
                    self.PFR += 1
                    self.last_pfr_action = action

                # 3-Bet Logic
                if action == "raise" and not self.last_3_bet_action:
                    # Check for any previous raise in the action history
                    for previous_action in reversed(pre_flop_action[:-1]):  # Go backwards to find the first raise
                        if previous_action["action"].lower() == "raise":
                            self.three_bet += 1
                            self.last_3_bet_action = action
                            break  # Exit loop after the first raise found

        return action, math.floor(amount)

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']
        for i in range(0, len(game_info['seats'])):
            if self.uuid == game_info['seats'][i]['uuid']:
                self.player_id = i
                break
        self.stack = 100

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.last_vpip_action = None
        self.last_pfr_action = None
        self.last_3_bet_action = None
        self.hand_count += 1
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    @staticmethod
    def round_int_to_string(round_int):
        m = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river'}
        return m[round_int]

    def receive_round_result_message(self, winners, hand_info, round_state):

        if len(self.history) >= 1 and self.training:

            for players in round_state['seats']: 
                
                if players['uuid'] == self.uuid:
                    if self.stack != 0:
                        reward = (players['stack'] - self.stack) / self.stack 
                    else: reward = 0
                    self.stack = players['stack']
                    break
            
            self.accumulated_reward += reward / 100

            #if winners[0]['uuid'] == self.uuid:
                #reward += 0.05

            action_num = len(round_state['action_histories'])
            if round_state['action_histories'][self.round_int_to_string(action_num - 1)] == []:
                action_num -= 1
            if action_num == 1:
                community_card = self.community_card_to_tuple([])
            elif action_num == 2:
                community_card = self.community_card_to_tuple(round_state['community_card'][:3])
            elif action_num == 3:
                community_card = self.community_card_to_tuple(round_state['community_card'][:4])
            elif action_num == 4:
                community_card = self.community_card_to_tuple(round_state['community_card'][:5])

            self.state = self.get_state(community_card, round_state)
            last_state = self.process_state(self.state)
            # append the last state to history
            self.history.append(last_state + (None,))

            # update using reward
            for i in range(len(self.history) - 1):
                h = self.history[i]
                next_h = self.history[i + 1]
                self.update(h[:-1], h[-1], reward, next_h[:-1], self.episode)
            # clear history
            self.history = []
            #self.save_model()

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_path)
        torch.save(self.optimizer.state_dict(), self.optimizer_path)