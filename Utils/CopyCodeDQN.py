import shutil
import os
# Specify the source file (from the parent folder)
source_file = os.path.join('my_players', 'DQNPlayer.py')  # Adjust the path to your source file
# Specify the target directory (subfolder)
target_directory = 'my_players'
# Create a list of target files in the subfolder
target_files = [os.path.join(target_directory, f'DQNPlayer{i}.py') for i in range(1, 7)]
# Copy the content from the source file to each target file
for i, target_file in enumerate(target_files, start=1):
    shutil.copyfile(source_file, target_file)
    print(f'Copied content from {source_file} to {target_file}')
    
    # Now replace the class name in the copied file
    with open(target_file, 'r') as file:
        file_contents = file.read()
        
    # Replace 'DQNPlayer' with 'DQNPlayer{i}' in the content
    new_contents = file_contents.replace('class DQNPlayer(QLearningPlayer)', f'class DQNPlayer{i}(QLearningPlayer)')
    
    # Write the modified content back to the file
    with open(target_file, 'w') as file:
        file.write(new_contents)
    
    print(f'Replaced class name in {target_file} to DQNPlayer{i}')