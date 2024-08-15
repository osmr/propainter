# Streaming Video Inpainting via ProPainter

A package for using ProPainter in streaming mode.

### Prerequisites
python >= 3.10

### Deployment 
```
pip install propainter
```

### Deployment for testing from repo
1. Install prerequisites:
```
sudo apt-get update
sudo apt-get update && sudo apt-get upgrade -y && sudo apt-get autoremove
```
2. Install virtualenv (actual [instructions](https://virtualenv.pypa.io/en/latest/installation.html)):
```
sudo -H pip install --upgrade pip setuptools wheel
sudo -H pip install virtualenv
```
3. Clone repo, create and activate environment:
```
git clone --recurse-submodules git@github.com:osmr/propainter.git
cd propainter
virtualenv venv
source venv/bin/activate
```
4. Update preinstalled packages:
```
pip install --upgrade pip setuptools wheel
```
5. Install package:
```
pip install -e .
```
6. Optionally install pytest:
```
pip install pytest
```

### Usage

1. Investigate the `example.py` script.
2. Investigate pytest scripts in `tests` directory.
