from pypokerengine.api.game import setup_config, start_poker
from my_players.RandomPlayer import RandomPlayer
from my_players.AllCall import AllCallPlayer
from my_players.QLearningPlayer import QLearningPlayer
from my_players.HumanPlayer import ConsolePlayer
from my_players.DQNPlayer import DQNPlayer
from my_players.cardplayer import cardplayer
from my_players.A2CPlayer import A2CPlayer
from my_players.HonestPlayer import HonestPlayer
from scipy.stats import t
import math
import os
import matplotlib.pyplot as plt

num_episode = 10000
count = 0
log_interval = 100

vpip_history = []

dqn_paths = {
        'model': os.getcwd() + f'/model/dqn.dump',
        'optimizer': os.getcwd() + f'/model/dqn_optim.dump'
    }

training_agent = DQNPlayer(dqn_paths['model'], dqn_paths['optimizer'], True)


# Set up configuration
config = setup_config(max_round=6, initial_stack=100, small_blind_amount=5)


config.register_player(name=f"p1", algorithm=training_agent)

for i in range(2, 7):
    config.register_player(name=f"p{i}", algorithm=HonestPlayer())

for i in range(0, num_episode):
    count += 1
    game_result = start_poker(config, verbose=0)

    if count % log_interval == 0:
        print(count)
        vpip_rate = training_agent.VPIP / training_agent.hand_count * 100 
        vpip_history.append(vpip_rate)
        print(f"VPIP over each episode: {vpip_rate:.2f}%")
        training_agent.VPIP , training_agent.hand_count = 0, 0
        config.players_info[0]['algorithm'].save_model()

# Plotting the VPIP history
plt.figure(figsize=(12, 6))
plt.plot(vpip_history, label='DQN Player VPIP')
    
plt.title('VPIP History of All Agents')
plt.xlabel('Episodes (every 10 episodes)')
plt.ylabel('VPIP (%)')
plt.legend()
plt.grid()
plt.savefig('images/DQN_vs_Honest_vpip_history.png')
plt.show()