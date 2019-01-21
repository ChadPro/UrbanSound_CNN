# UrbanSound_CNN.
## 0.Requirements
**Tensorflow-1.12.0**  
**Python2**

## 1. Download UrbanSoud data
You can download data from office website https://urbansounddataset.weebly.com/.  
数据文件目录如下所示:  
UrbanSound8K/  
├── audio  
│   ├── fold10  
│   ├── train  
│   └── val  
└── metadata  

## 2. Extract Features
We use LibROSA package read audio, and get mfcc features.Use python scipt *extract_features.py*.
```python
python extract_features.py
```
And after that we get four filse, *train_x.dat*, *train_y.dat*, *val_x.dat*, *val_y.dat*.  
We can also use *check_features.py* check the features files.
```python
python check_features.py
```

## 3. Train

## 4. Use Model Test Audio
