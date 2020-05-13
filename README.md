# Intro to ML Notebooks

## Environment (Optional) 
Create a conda environment where everything should run. 
```bash
$ conda create -n intro-ml python=3.7 anaconda
$ conda activate intro-ml
```

## Install requirements
To install the requirements
```bash
$ pip install -r requirements.txt
```

## Run a jupyter notebook server
Start a jupyter notebook server by running 
```bash
$ jupyter notebook 
```

## Run a jupyter notebook in a remote cluster
Go to Server using

```bash
$ ssh username@ip_address 
```

On remote terminal run:
```bash
$ jupyter notebook --no-browser --port=7800 
```

On your local terminal run [explained](https://explainshell.com/explain?cmd=ssh+-N+-f+-L+localhost%3A8001%3Alocalhost%3A7800+username%40ip_address):
```bash
$ ssh -N -f -L localhost:8001:localhost:7800 username@ip_address 
```

Open web browser on local machine and go to http://localhost:8001/
