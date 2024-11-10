import subprocess

# Define the command as a list of strings
command = ["longjob", "-28day", "-c", "./run_training.sh"]

# Run the command using subprocess
try:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print("Command output:", result.stdout)
except subprocess.CalledProcessError as e:
    print("Error:", e.stderr)
    print("Command failed with return code:", e.returncode)
