<p align="center">

<h1 align="center">EC-classifier-training-framework</h1>
<p align="center">
Hongqing Liu, Stanley Gu

</p>
    
## Run
Here you can run EC-Fusion-RGBIR yourself using terminal command.

Training model

```bash
python -W ignore run.py config/RGB_image.yaml
python -W ignore run.py config/IR_image.yaml
```

Testing Trained model on collected dataset
```bash
python -W ignore test.py config/RGB_image.yaml
python -W ignore test.py config/IR_image.yaml
```

### configuration file
Here you can change configuration file to load different datasets and using different network

```bash
dataset: 'RGB_image_single' # choose the datasets you want to use.
# Option: "single_RGB_image", "single_fusion_image", "multi_IR_image", ...
mode: 'visible' # 'IR', 'Fusion'
...
```

### network
If you want to use network, you can add it in src/models.

## Datasets
Here I show how we store input dataset for single object detection, you can create a new file under file "datasets".
The data folder that has been normalized by the temperature interval is named thermal_N.
Dataset link: https://drive.google.com/drive/folders/1vfvrZN-OC0U4gPugdU_IL844YZfxF3rV?usp=sharing
```
  DATAROOT
  └── datasets
      └── file_name
          ├── thermal
          │   ├── 00000.jpg
          │   ├── 00001.jpg
          │   ├── 00002.jpg
          │   ├── ...
          │   └── ...
          ├── thermal
          │   ├── 00000.jpg
          │   ├── 00001.jpg
          │   ├── 00002.jpg
          │   ├── ...
          │   └── ...    
          ├── visible
          │   ├── 00000.jpg
          │   ├── 00001.jpg
          │   ├── 00002.jpg
          │   ├── ...
          │   └── ...
          └── label.csv

```
