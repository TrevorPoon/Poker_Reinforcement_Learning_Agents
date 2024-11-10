import pandas as pd
import os
import matplotlib.pyplot as plt
def moving_average(data, window_size=10):
    """Calculate the moving average manually."""
    if len(data) < window_size:
        return data  # Not enough data to smooth
    averages = []
    for i in range(len(data) - window_size + 1):
        avg = sum(data[i:i + window_size]) / window_size
        averages.append(avg)
    return averages
def plotting_VPIP(vpip_df):
    plt.figure(figsize=(12, 6))
    for j in range(6):
        plt.plot(vpip_df.iloc[:, j], label=f'Player {j+1} VPIP')
        # Smoothing with a moving average
    plt.title('VPIP History of All Agents')
    plt.xlabel('Episodes (every 100 episodes)')
    plt.ylabel('VPIP (%)')
    plt.legend()
    plt.grid()
    plt.savefig('images/vpip_history.png')
    plt.close()
    plt.figure(figsize=(12, 6))
    for j in range(6):
        smoothed_vpip = moving_average(vpip_df.iloc[:, j])
        plt.plot(range(len(smoothed_vpip)), smoothed_vpip, linestyle='--', 
                label=f'Player {j+1} Smoothed VPIP', alpha=0.7)
    plt.title('VPIP History of All Agents')
    plt.xlabel('Episodes (every 100 episodes)')
    plt.ylabel('VPIP (%)')
    plt.legend()
    plt.grid()
    plt.savefig('images/vpip_history_smoothed.png')
    plt.close()
        
def plotting_PFR(pfr_df):
    plt.figure(figsize=(12, 6))
    for j in range(6):
        plt.plot(pfr_df.iloc[:, j], label=f'Player {j+1} PFR')
    
    plt.title('PFR History of All Agents')
    plt.xlabel('Episodes (every 100 episodes)')
    plt.ylabel('PFR (%)')
    plt.legend()
    plt.grid()
    plt.savefig('images/pfr_history.png')
    plt.close()
    plt.figure(figsize=(12, 6))
    for j in range(6):
        # Smoothing with a moving average
        smoothed_pfr = moving_average(pfr_df.iloc[:, j])
        plt.plot(range(len(smoothed_pfr)), smoothed_pfr, linestyle='--', 
                 label=f'Player {j+1} Smoothed PFR', alpha=0.7)
    
    plt.title('PFR History of All Agents')
    plt.xlabel('Episodes (every 100 episodes)')
    plt.ylabel('PFR (%)')
    plt.legend()
    plt.grid()
    plt.savefig('images/pfr_history_smoothed.png')
    plt.close()
def plotting_3bet(three_bet_df):
    plt.figure(figsize=(12, 6))
    for j in range(6):
        plt.plot(three_bet_df.iloc[:, j], label=f'Player {j+1} 3-Bet %')
    plt.title('3-Bet Percentage History of All Agents')
    plt.xlabel('Episodes (every 100 episodes)')
    plt.ylabel('3-Bet %')
    plt.legend()
    plt.grid()
    plt.savefig('images/three_bet_history.png')
    plt.close()
    plt.figure(figsize=(12, 6))
    for j in range(6):
        # Smoothing with a moving average
        smoothed_3bet = moving_average(three_bet_df.iloc[:, j])
        plt.plot(range(len(smoothed_3bet)), smoothed_3bet, linestyle='--', 
                 label=f'Player {j+1} Smoothed 3-Bet %', alpha=0.7)
    plt.title('3-Bet Percentage History of All Agents')
    plt.xlabel('Episodes (every 100 episodes)')
    plt.ylabel('3-Bet %')
    plt.legend()
    plt.grid()
    plt.savefig('images/three_bet_history_Smoothed.png')
    plt.close()
vpip_df = pd.DataFrame()
pfr_df = pd.DataFrame()
three_bet_df = pd.DataFrame()
if os.path.exists("DQN_Stat/vpip_history.csv"):
    vpip_df = pd.read_csv("DQN_Stat/vpip_history.csv")
if os.path.exists("DQN_Stat/pfr_history.csv"):
    pfr_df = pd.read_csv("DQN_Stat/pfr_history.csv")
if os.path.exists("DQN_Stat/three_bet_history.csv"):
    three_bet_df = pd.read_csv("DQN_Stat/three_bet_history.csv")
print(vpip_df)
plotting_VPIP(vpip_df)
plotting_PFR(pfr_df)
plotting_3bet(three_bet_df)