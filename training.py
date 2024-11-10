from pypokerengine.api.game import setup_config, start_poker
from my_players.DQNPlayer import DQNPlayer
from my_players.DQNPlayer1 import DQNPlayer1
from my_players.DQNPlayer2 import DQNPlayer2
from my_players.DQNPlayer3 import DQNPlayer3
from my_players.DQNPlayer4 import DQNPlayer4
from my_players.DQNPlayer5 import DQNPlayer5
from my_players.DQNPlayer6 import DQNPlayer6
from my_players.HonestPlayer import HonestPlayer
from my_players.cardplayer import cardplayer
from my_players.AllCall import AllCallPlayer
import os
import matplotlib.pyplot as plt
import gc
import pandas as pd
import numpy as np 

# User's Input
num_episode = 10000000
count = 0
log_interval = 100
Num_of_agents = 1
Title = 'DQN_vs_AllCall' # 'DQN_vs_AllCall' 'DQN_vs_DQN'


def moving_average(data, window_size=10):
    """Calculate the moving average manually."""
    if len(data) < window_size:
        return data  # Not enough data to smooth
    averages = []
    for i in range(len(data) - window_size + 1):
        avg = sum(data[i:i + window_size]) / window_size
        averages.append(avg)
    return averages

# Apply a valid, professional style (like ggplot or seaborn-whitegrid)
plt.style.use('ggplot')

def apply_professional_formatting():
    """Apply professional formatting to the current plot."""
    # Set the font and size for titles, labels, and legend
    plt.title(plt.gca().get_title(), fontsize=16, fontweight='bold', fontname='Arial', color='#333333')
    plt.xlabel(plt.gca().get_xlabel(), fontsize=14, fontweight='bold', fontname='Arial', color='#333333')
    plt.ylabel(plt.gca().get_ylabel(), fontsize=14, fontweight='bold', fontname='Arial', color='#333333')
    
    # Customizing ticks and grid
    plt.xticks(fontsize=12, fontname='Arial', color='#333333')
    plt.yticks(fontsize=12, fontname='Arial', color='#333333')
    
    # Customizing grid
    plt.grid(True, which='both', color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    
    # Customizing legend
    plt.legend(loc='upper right', fontsize=8, frameon=True, fancybox=True, framealpha=0.8, shadow=True)
    
    # Light grey background
    plt.gcf().set_facecolor('#f9f9f9')
    plt.gca().set_facecolor('#ffffff')

def plot_metric(df, metric_name, ylabel, file_suffix):
    """Plots both raw and smoothed graphs for a given metric."""
    # Plot raw data
    plt.figure(figsize=(12, 6))
    for j in range(Num_of_agents):
        plt.plot(df.iloc[:, j], label=f'Player {j+1} {metric_name}', linewidth=2)
    plt.title(f'{metric_name} History of All Agents')
    plt.xlabel('Episodes (every 100 episodes)')
    plt.ylabel(ylabel)
    apply_professional_formatting()
    plt.savefig(f'images/{Title}_{file_suffix}.png')
    plt.close()
    
    # Plot smoothed data
    plt.figure(figsize=(12, 6))
    for j in range(Num_of_agents):
        smoothed_data = moving_average(df.iloc[:, j])
        plt.plot(range(len(smoothed_data)), smoothed_data, linestyle='--', 
                 label=f'Player {j+1} Smoothed {metric_name}', alpha=0.8, linewidth=2)
    plt.title(f'Smoothed {metric_name} History of All Agents')
    plt.xlabel('Episodes (every 100 episodes)')
    plt.ylabel(ylabel)
    apply_professional_formatting()
    plt.savefig(f'images/{Title}_{file_suffix}_smoothed.png')
    plt.close()

# Declaration
vpip_history, pfr_history, three_bet_history, loss_history, reward_history = [], [], [], [], []

vpip_df, pfr_df, three_bet_df, loss_df, reward_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Load data from CSV files if they exist
file_paths = {
    'vpip': f"DQN_Stat/{Title}_vpip_history.csv",
    'pfr': f"DQN_Stat/{Title}_pfr_history.csv",
    'three_bet': f"DQN_Stat/{Title}_three_bet_history.csv",
    'loss': f"DQN_Stat/{Title}_loss_history.csv",
    'reward': f"DQN_Stat/{Title}_reward_history.csv"
}

dataframes = {
    'vpip': 'vpip_df',
    'pfr': 'pfr_df',
    'three_bet': 'three_bet_df',
    'loss': 'loss_df',
    'reward': 'reward_df'
}

for key, path in file_paths.items():
    if os.path.exists(path):
        try:
            globals()[dataframes[key]] = pd.read_csv(path)
        except Exception as e:
            print(f"Warning: Could not load {path}. Error: {e}")

dqn_paths = {}
if Num_of_agents == 1:

    dqn_paths = {
        'model': os.getcwd() + f'/model/dqn_{Title}.dump',
        'optimizer': os.getcwd() + f'/model/dqn_optim_{Title}.dump'
    }

    training_agents =  [DQNPlayer(dqn_paths['model'], dqn_paths['optimizer'], True)]
else:
    for i in range(Num_of_agents):
        dqn_paths[i] = {
            'model': os.getcwd() + f'/model/dqn{i+1}.dump',
            'optimizer': os.getcwd() + f'/model/dqn{i+1}_optim.dump'
        }

    training_agents =  [DQNPlayer1(dqn_paths[0]['model'], dqn_paths[0]['optimizer'], True),
                        DQNPlayer2(dqn_paths[1]['model'], dqn_paths[1]['optimizer'], True),
                        DQNPlayer3(dqn_paths[2]['model'], dqn_paths[2]['optimizer'], True),
                        DQNPlayer4(dqn_paths[3]['model'], dqn_paths[3]['optimizer'], True),
                        DQNPlayer5(dqn_paths[4]['model'], dqn_paths[4]['optimizer'], True),
                        DQNPlayer6(dqn_paths[5]['model'], dqn_paths[5]['optimizer'], True)]

# Set up configuration
config = setup_config(max_round=6, initial_stack=100, small_blind_amount=0.5)

# Register each player with their respective DQNPlayer instance
for i in range(Num_of_agents):
    config.register_player(name=f"p{i+1}", algorithm=training_agents[i])

for i in range(Num_of_agents+1, 7):
    config.register_player(name=f"p{i+1}", algorithm=AllCallPlayer())

accum_reward = 0

# Game_Simulation
for i in range(0, num_episode):
    count += 1
    game_result = start_poker(config, verbose=0)

    if count % log_interval == 0:
        print(count)
        loss_switch = 1
        for j in range(Num_of_agents):

            vpip_rate = training_agents[j].VPIP / training_agents[j].hand_count * 100 
            vpip_history.append(vpip_rate)
            print(f"VPIP over each episode: {vpip_rate:.2f}%")

            # Calculate and log PFR
            pfr_rate = training_agents[j].PFR / training_agents[j].hand_count * 100 
            pfr_history.append(pfr_rate)
            print(f"PFR over each episode: {pfr_rate:.2f}%")

            # Calculate and log 3-Bet Percentage
            three_bet_rate = training_agents[j].three_bet / training_agents[j].hand_count * 100
            three_bet_history.append(three_bet_rate)
            print(f"3-Bet Percentage over each episode: {three_bet_rate:.2f}%")

            try:
                model_loss = training_agents[j].loss
                loss_history.append(model_loss)
                print(f"Model Loss: {model_loss:.5f}")
            except:
                loss_switch = 0
            
            accum_reward = training_agents[j].accumulated_reward
            reward_history.append(accum_reward)
            print(f"Reward: {accum_reward:.2f}")

            # Resetting the counts for the next episode
            training_agents[j].VPIP, training_agents[j].PFR, training_agents[j].three_bet, training_agents[j].hand_count = 0, 0, 0, 0
            config.players_info[j]['algorithm'].save_model()


        new_vpip = pd.DataFrame([vpip_history])
        new_pfr = pd.DataFrame([pfr_history])
        new_three_bet = pd.DataFrame([three_bet_history])
        new_loss = pd.DataFrame([loss_history])
        new_reward = pd.DataFrame([reward_history])

        if not vpip_df.empty:
            new_vpip.columns = vpip_df.columns
        
        if not pfr_df.empty:
            new_pfr.columns = pfr_df.columns
        
        if not three_bet_df.empty:
            new_three_bet.columns = three_bet_df.columns

        if not loss_df.empty:
            new_loss.columns = loss_df.columns

        if not reward_df.empty:
            new_reward.columns = reward_df.columns

        # Reset index before concatenation
        vpip_df = pd.concat([vpip_df, new_vpip], ignore_index=True)  # No ignore_index here
        pfr_df = pd.concat([pfr_df, new_pfr], ignore_index=True)      # No ignore_index here
        three_bet_df = pd.concat([three_bet_df, new_three_bet],ignore_index=True)
        if loss_switch == 1:
            loss_df = pd.concat([loss_df, new_loss], ignore_index=True)
        reward_df = pd.concat([reward_df, new_reward], ignore_index=True)


        vpip_df.to_csv(f"DQN_Stat/{Title}_vpip_history.csv", index=False)
        pfr_df.to_csv(f"DQN_Stat/{Title}_pfr_history.csv", index=False)
        three_bet_df.to_csv(f"DQN_Stat/{Title}_three_bet_history.csv", index=False)
        loss_df.to_csv(f"DQN_Stat/{Title}_loss_history.csv", index=False)
        reward_df.to_csv(f"DQN_Stat/{Title}_reward_history.csv", index=False)

        vpip_history, pfr_history, three_bet_history, loss_history, reward_history= [], [], [], [], []

        plot_metric(vpip_df, 'VPIP', 'VPIP (%)', 'vpip_history')
        plot_metric(pfr_df, 'PFR', 'PFR (%)', 'pfr_history')
        plot_metric(three_bet_df, '3-Bet %', '3-Bet %', 'three_bet_history')
        plot_metric(reward_df, 'Reward', 'Reward', 'Reward')
        if loss_switch == 1:
            plot_metric(loss_df, 'Model Loss', 'Loss', 'model_loss')
        

        if loss_switch == 1:
            if round(model_loss, 5) == 0:
                break

        if count % 10000 == 0:
            gc.collect()
        

