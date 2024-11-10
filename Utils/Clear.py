import subprocess
import time

# List of Python scripts to run
scripts = ["Utils/CopyCodeDQN.py", "Utils/Delete_csv.py", "Utils/Delete_Models.py"]

# Function to run a script and restart if it fails
def run_script(script_path):
    while True:
        try:
            # Run the script
            result = subprocess.run(["python", script_path], check=True)
            print(f"{script_path} completed successfully.")
            break  # Exit loop if the script ran successfully
        except subprocess.CalledProcessError:
            print(f"An error occurred in {script_path}. Restarting...")
            time.sleep(1)  # Optional delay before restarting

# Loop through each script and run them sequentially
for script in scripts:
    run_script(script) 
