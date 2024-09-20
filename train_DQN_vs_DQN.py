from pypokerengine.api.game import setup_config, start_poker
from my_players import (DQNPlayer1, DQNPlayer2, DQNPlayer3, DQNPlayer4, 
                        DQNPlayer5, DQNPlayer6, HonestPlayer)
import os
import matplotlib.pyplot as plt
import gc
import pandas as pd

num_episode = 1000000
win = 0
sample_mean = 0
SXX = 0
sample_std = 0
count = 0
log_interval = 100
log = []
confidence_level = 0.05
Num_of_agents = 6

# Plotting the VPIP history
def plotting_VPIP (vpip_history):
    plt.figure(figsize=(12, 6))
    for j in range(1, 7):
        plt.plot(vpip_history[j], label=f'Player {j} VPIP')
    plt.title('VPIP History of All Agents')
    plt.xlabel('Episodes (every 100 episodes)')
    plt.ylabel('VPIP (%)')
    plt.legend()
    plt.grid()
    plt.savefig('images/vpip_history.png')

def plotting_PFR (pfr_history):
# Plotting the PFR history
    plt.figure(figsize=(12, 6))
    for j in range(1, 7):
        plt.plot(pfr_history[j], label=f'Player {j} PFR')
    plt.title('PFR History of All Agents')
    plt.xlabel('Episodes (every 100 episodes)')
    plt.ylabel('PFR (%)')
    plt.legend()
    plt.grid()
    plt.savefig('images/pfr_history.png')

def plotting_3bet (three_bet_history):
    # Plotting the 3-Bet history
    plt.figure(figsize=(12, 6))
    for j in range(1, 7):
        plt.plot(three_bet_history[j], label=f'Player {j} 3-Bet %')
    plt.title('3-Bet Percentage History of All Agents')
    plt.xlabel('Episodes (every 100 episodes)')
    plt.ylabel('3-Bet %')
    plt.legend()
    plt.grid()
    plt.savefig('images/three_bet_history.png')


vpip_history = {i: [] for i in range(1, Num_of_agents+1)}
pfr_history = {i: [] for i in range(1, Num_of_agents+1)}
three_bet_history = {i: [] for i in range(1, Num_of_agents+1)}

dqn_paths = {}
for i in range(1, Num_of_agents+1):
    dqn_paths[i] = {
        'model': os.getcwd() + f'/model/dqn{i}.dump',
        'optimizer': os.getcwd() + f'/model/dqn{i}_optim.dump'
    }


training_agents = [None] + [DQNPlayer1(dqn_paths[1]['model'], dqn_paths[1]['optimizer'], True),
                            DQNPlayer2(dqn_paths[2]['model'], dqn_paths[2]['optimizer'], True),
                            DQNPlayer3(dqn_paths[3]['model'], dqn_paths[3]['optimizer'], True),
                            DQNPlayer4(dqn_paths[4]['model'], dqn_paths[4]['optimizer'], True),
                            DQNPlayer5(dqn_paths[5]['model'], dqn_paths[5]['optimizer'], True),
                            DQNPlayer6(dqn_paths[6]['model'], dqn_paths[6]['optimizer'], True)]


# Set up configuration
config = setup_config(max_round=36, initial_stack=100, small_blind_amount=5)

# Register each player with their respective DQNPlayer instance
for i in range(1, Num_of_agents+1):
    config.register_player(name=f"p{i}", algorithm=training_agents[i])

for i in range(Num_of_agents+1, 7):
    config.register_player(name=f"p{i}", algorithm=HonestPlayer())

vpip_df = pd.DataFrame()
pfr_df = pd.DataFrame()
three_bet_df = pd.DataFrame()

if os.path.exists("DQN_Stat/vpip_history.csv"):
    vpip_df = pd.read_csv("DQN_Stat/vpip_history.csv")
if os.path.exists("DQN_Stat/pfr_history.csv"):
    pfr_df = pd.read_csv("DQN_Stat/pfr_history.csv")
if os.path.exists("DQN_Stat/three_bet_history.csv"):
    three_bet_df = pd.read_csv("DQN_Stat/three_bet_history.csv")

for i in range(0, num_episode):
    count += 1
    game_result = start_poker(config, verbose=0)

    if count % log_interval == 0:
        print(count)
        for j in range(1, Num_of_agents+1):

            vpip_rate = training_agents[j].VPIP / training_agents[j].hand_count * 100 
            vpip_history[j].append(vpip_rate)
            print(f"VPIP over each episode: {vpip_rate:.2f}%")

            # Calculate and log PFR
            pfr_rate = training_agents[j].PFR / training_agents[j].hand_count * 100 
            pfr_history[j].append(pfr_rate)
            print(f"PFR over each episode: {pfr_rate:.2f}%")

            # Calculate and log 3-Bet Percentage
            three_bet_rate = training_agents[j].three_bet / training_agents[j].hand_count * 100
            three_bet_history[j].append(three_bet_rate)
            print(f"3-Bet Percentage over each episode: {three_bet_rate:.2f}%")

            # Resetting the counts for the next episode
            training_agents[j].VPIP, training_agents[j].PFR, training_agents[j].three_bet, training_agents[j].hand_count = 0, 0, 0, 0
            config.players_info[j-1]['algorithm'].save_model()
        
        plotting_VPIP(vpip_history)
        plotting_PFR(pfr_history)
        plotting_3bet(three_bet_history)

        new_vpip = pd.DataFrame(vpip_history)
        new_pfr = pd.DataFrame(pfr_history)
        new_three_bet = pd.DataFrame(three_bet_history)

        vpip_df = pd.concat([vpip_df, new_vpip], ignore_index=True)
        pfr_df = pd.concat([pfr_df, new_pfr], ignore_index=True)
        three_bet_df = pd.concat([three_bet_df, new_three_bet], ignore_index=True)

        vpip_df.to_csv("DQN_Stat/vpip_history.csv", index=False)
        pfr_df.to_csv("DQN_Stat/pfr_history.csv", index=False)
        three_bet_df.to_csv("DQN_Stat/three_bet_history.csv", index=False)

        if count % 10000 == 0:
            gc.collect()
        

