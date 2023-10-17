# Install

This project requires Python>=3.7.0 and PyTorch>=1.7.  

## Python version
To check your python version if you have already installed it, 
type the following in the [command line](https://en.wikiversity.org/wiki/Command_Prompt/Open)
or [powershell](https://learn.microsoft.com/en-us/powershell/scripting/windows-powershell/starting-windows-powershell?view=powershell-7.3) for windows, 
or in the [terminal](https://support.apple.com/en-gb/guide/terminal/apd5265185d-f365-44cb-8b09-71a064a42125/mac) for macOS.
```bash
python -V
```
If it does not recognise python as a command, this might be because you have not added python to your `PATH` variables, 
then please refer to this [tutorial](https://realpython.com/add-python-to-path/).  

## Install Python
If your python version does not match the requirements or you do not have python installed, 
please install [Python 3.9](https://www.python.org/downloads/release/python-390/) with this 
[tutorial](https://realpython.com/installing-python/) to help. 
(Make sure that you download an installer compatible with your operating system from the Python 3.9 webpage, 
check `Add Python 3.9 to PATH` when running the installer and installing `pip` in the optional features)

## Install PyTorch
After you have python installed, install PyTorch.
```bash
pip install torch
```
If it does not recognise pip as a command, please refer to this 
[tutorial](https://www.alphr.com/pip-is-not-recognized-as-an-internal-or-external-command/).

You can check the PyTorch version.
```bash
python -c "import torch; print(torch.__version__)"
```

## Install Git
Refer to this [tutorial](https://github.com/git-guides/install-git)
if you have not already installed git.

## Install Project
Clone this repo and install 
[requirements.txt](https://github.com/teethoe/BirdTracking/blob/master/requirements.txt).
```bash
git clone https://github.com/teethoe/BirdTracking  # clone
cd BirdTracking
pip install -r requirements.txt  # install
```
Take a note of the path for this project, this will be used for returning to work on the project.
```bash
pwd
```

## Returning to work on the project
To work on this project, go to the project's directory by running the following 
command and replacing `[project path]`with the path returned from the previous `pwd` command.
```bash
cd {project path}
```
Git pull for project updates.
```bash
git pull
```