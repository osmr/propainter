# Streaming Video Inpainting via ProPainter

A package for using [ProPainter](https://arxiv.org/pdf/2309.03897) in streaming mode (e.g. for a very long video).

### Prerequisites
python >= 3.10

### Deployment 
```
pip install propainter
```

### Deployment for testing from repo
1. Install prerequisites:
```
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
3. Launch the `example.py` script on your data: 
```
python example.py --frames=<frame_dir> --masks=<mask_dir> --output=<output_dir> --resize_ratio=1.0 --save
```

### License
Please note that ProPainter is subject to a non-commercial [S-Lab license](https://github.com/sczhou/ProPainter/blob/main/LICENSE).

### Acknowledgement
This code is based on [ProPainter](https://github.com/sczhou/ProPainter). Thanks for this awesome work.
