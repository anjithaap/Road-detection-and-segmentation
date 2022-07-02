# Image segmentation with FCN-8 and FCN-32
Segmenting road from satellite images using FCN-8 and FCN-32 neural networks with Keras


## Download the Dataset
* An archive file containing satellite images as JPEG files within respective directories
* The image files have dimension as `512 X 512` pixels
```bash
pip install -U gdown                           # Install gdown to download GDrive files
gdown 1u4WJLjYrbZHwdvFOHQXJqDTtco6F5hJ-        # Download Dataset.zip file from Google Drive
```

## Prepare workspace
```bash
cd ~/
git clone https://github.com/anjithaap/Road-detection-and-segmentation.git workspace
cd workspace/
gdown 1u4WJLjYrbZHwdvFOHQXJqDTtco6F5hJ-
unzip -q Dataset.zip
rm -rf epochs; mkdir epochs

# Start Training
python3 train.py

# Predict from saved model
python3 predict.py
```
