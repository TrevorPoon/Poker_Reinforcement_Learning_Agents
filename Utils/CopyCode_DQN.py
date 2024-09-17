import shutil
import os

# Specify the source file (from the parent folder)
source_file = '/Users/trevorpoon/Desktop/Coding Projects/Poker Reinforcement Learning Agents/my_players/DQNPlayer1.py'  # Adjust the path to your source file

# Specify the target directory (subfolder)
target_directory = 'my_players'

# Create a list of target files in the subfolder
target_files = [os.path.join(target_directory, f'DQNPlayer{i}.py') for i in range(2, 7)]

# Copy the content from the source file to each target file
for target_file in target_files:
    shutil.copyfile(source_file, target_file)
    print(f'Copied content from {source_file} to {target_file}')