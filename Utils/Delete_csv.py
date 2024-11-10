import os
# Specify the path to the 'Model' directory
model_directory = os.path.join(os.getcwd(), 'DQN_Stat')
print(model_directory)
# Check if the directory exists
if os.path.exists(model_directory):
    # Iterate over all files in the directory
    for filename in os.listdir(model_directory):
        file_path = os.path.join(model_directory, filename)
        # Check if it's a file and delete it
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f'Deleted: {file_path}')
else:
    print("The specified directory does not exist.")