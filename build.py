import os
import subprocess
import sys


def run_command(command):
	process = subprocess.Popen(command, shell=True)
	process.wait()
	return process.returncode


# Create a virtual environment
if len(sys.argv) < 2:
	venv_path = "venv"
else:
	venv_path = sys.argv[1]

if not os.path.exists(venv_path):
	print("Creating virtual environment...")
	return_code = run_command(f"{sys.executable} -m venv {venv_path}")
	if return_code != 0:
		print("Failed to create virtual environment.")
		sys.exit(1)
	else:
		print("Virtual environment created successfully.")
print("Virtual environment already exists.")

# Activate the virtual environment
print("Activating virtual environment...")
activate_script = "activate.bat" if sys.platform == "win32" else "activate"
activate_path = os.path.join(venv_path, "Scripts" if sys.platform == "win32" else "bin", activate_script)
activate_command = f"{activate_path}"

if sys.platform != "win32":
	activate_command = f". {activate_command}"

# Install requirements
print("Installing requirements...")
print("Installing wheel...")
install_command = f"{activate_command} && pip install wheel"
return_code = run_command(install_command)
if return_code != 0:
	print("Failed to install requirements.")
	sys.exit(1)

print("Installing JAX...")
install_command = f"{activate_command} && pip install --upgrade jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
return_code = run_command(install_command)
if return_code != 0:
	print("Failed to install requirements.")
	sys.exit(1)

print("Installing Additional Libraries...")
with open("req.txt", "w") as f:
	f.write("clu\n")
	f.write("matplotlib\n")
	f.write("flax\n")
	f.write("datasets\n")
	f.write("joblib\n")

install_command = f"{activate_command} && pip install -r req.txt"
return_code = run_command(install_command)
if return_code != 0:
	print("Failed to install requirements.")
	sys.exit(1)

print("Requirements installed successfully.")
print("Project setup complete.")
