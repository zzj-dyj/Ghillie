# Ghillie
Implementation of "Ghost Imaging in the Dark: A Multi-Illumination Estimation Network for Low-Light Image Enhancement"



# Introduction

It is well known that the diverse causes of low-light images challenge the adaptability of enhancement algorithms in uncertain environments. Most deep learning-based algorithms only learn single illuminance estimation or mapping relationship, which inhibit the generalization ability of the model. To address this, we propose a novel multi-illumination estimation framework based on ghost imaging theory, dubbed Ghillie. Specifically, we consider low-light enhancement as a re-imaging process for objects in dark scenes. First, the light modulation network (LMN) is designed to modulate a series of estimated lights following a normal light distribution. These lights “illuminate” the low-light image and the enhanced illuminance image can be reconstructed by a differential ghost imaging algorithm. Then, a gradient-guided denoising network (GDN) is constructed to eliminate noise and enhance details. Finally, we employ the color adaption network (CAN) to restore the color degradation. Additionally, a novel mean structural similarity loss (AM-SSIM) is proposed to guide the model to address the uneven image illumination. The qualitative and quantitative experimental results show that our enhanced methods outperform state-of-the-art methods on the vast majority of publicly available datasets.


# Environment
We are good in the environment:

python 3.8

Pytorch >= 1.7.0

## Datasets

- Low-light dataset: [LOL](https://daooshee.github.io/BMVC2018website/)
- MIT-Adobe FiveK dataset: [MIT](https://drive.google.com/drive/folders/144GTFl8SLygM_yWfNnzkk-RWSXu4eypt?usp=sharing)
- LSRW dataset: [LSRW](https://github.com/JianghaiSCU/R2RNet)


# Run a demo


To quickly run a demo, you can :


```
python test.py --data_path path/to/lowlight_dir --LMN_model ./weights/LMN_weights.pt --DNN_model ./weights/DNN_weights.pt --CAN_model ./weights/CAN_weights.pt

```


## Training

The directory structure of training datasets is :
```shell
data/
└── train
   ├── high
   └── low
```

To train LMN model, you can run:

```
python train_LMN.py --train_data_path ./data/train --M 32 --epoch 500

```

To train DNN model, you can run:

```
python train_DNN.py --train_data_path ./data/train --LMN_model ./weights/LMN_weights.pt --M 32 --epoch 500

```

To train CAN model , you can run:

```
python train_CAN.py --train_data_path ./data/train --epoch 500

```
The trained weights can be find in “./EXP/Train_LMN(_DNN_CAN)/Train{_date}/model_epochs”

# Acknowledgment
Our code is built on

 [SCI](https://github.com/tengyu1998/SCI)

 We thank the authors for sharing their codes!
