
# Battus Museum Data 

#### CVAT Labeling and Data Setup
- Labels: tags, ruler, lepidopteran 
    ```json
    [
      { "name": "tags",         "id": 1, "color": "#00ff00", "type": "mask", "attributes": [] },
      { "name": "ruler",        "id": 2, "color": "#ffff00", "type": "mask", "attributes": [] },
      { "name": "lepidopteran", "id": 3, "color": "#ff0000", "type": "mask", "attributes": [] }
    ]
    ```
    - [Insert Screenshot]
- Use SAM for semi-automated mask labeling
    - [Insert Screenshot]
- Export the project as CamVid from Battus labeling project 
    - [Insert Screenshot]
    - Copy training and validation images to the battus100 folders
    - Run the [./prep_images.py](./prep_images.py) script to setup the grayscale and label images 


### FastAI Training Notebook
- UNET model for semantic segmentation [`train.ipynb`](./train.ipynb)
  - References:
    - FastAI - https://colab.research.google.com/github/visual-layer/vl-datasets/blob/main/notebooks/train-fastai.ipynb
    - FastAI - https://github.com/ashutoshraj/Dynamic-Unet/blob/master/caravan-work.ipynb
- Train the models on Battus100 dataset [`train.py`](./train.py)
  - Resnet34, Batch 4, Epoch 50 with image resolution as 1200x800 
  ```log
    /mothra/data$ python train.py
    Unique labels: {'background': 0, 'lepidopteran': 1, 'tags': 2, 'ruler': 3}
    2024-04-09 18:15:16.092956       Total Images: 80        Sample:  /home/rahul/workspace/vision/eeb/mothra/data/battus100/training_images/images/IMG_4810.JPG
    2024-04-09 18:15:21.760610       Creating Learner, loading the model
    2024-04-09 18:15:43.701323       Train model for epochs= 50
    epoch     train_loss  valid_loss  acc_camvid  time
    0         0.806768    0.459883    0.636961    02:33
    epoch     train_loss  valid_loss  acc_camvid  time
    0         0.149808    0.128320    0.808339    09:12
    1         0.115419    0.079170    0.896724    10:41
    2         0.095822    0.079611    0.954942    14:17
    3         0.084676    0.051534    0.953409    03:59
    4         0.072986    0.038313    0.943552    04:43
    5         0.062362    0.033027    0.948094    07:52
    6         0.057291    0.036135    0.953725    05:45
    7         0.052212    0.039786    0.919670    16:04
    8         0.046736    0.026606    0.967843    05:24
    9         0.043211    0.033908    0.986602    12:12
    10        0.040234    0.046850    0.973135    07:54
    11        0.039036    0.030431    0.944569    06:53
    12        0.037563    0.026196    0.958800    05:09
    13        0.035784    0.028760    0.974625    06:14
    14        0.033103    0.027583    0.980055    07:27
    15        0.032790    0.044074    0.959578    11:16
    16        0.032898    0.028118    0.976511    07:45
    17        0.032550    0.029431    0.950204    14:27
    18        0.036883    0.044919    0.911377    06:30
    19        0.037076    0.026376    0.971383    07:01
    20        0.033519    0.027041    0.972268    18:33
    21        0.031397    0.023548    0.968206    05:32
    22        0.029965    0.023719    0.970279    07:47
    23        0.028803    0.023212    0.966944    05:48
    24        0.027795    0.023248    0.977940    08:06
    25        0.026741    0.028469    0.944889    05:45
    26        0.025991    0.023879    0.960224    11:14
    27        0.024507    0.023116    0.964569    05:33
    28        0.024004    0.023602    0.966349    04:52
    29        0.023738    0.021643    0.962317    06:40
    30        0.023573    0.024443    0.965278    08:04
    31        0.023717    0.036110    0.953635    08:12
    32        0.023424    0.022199    0.974851    07:46
    33        0.023092    0.023003    0.968156    05:39
    34        0.023080    0.021869    0.967127    06:15
    35        0.021702    0.028368    0.957976    07:21
    36        0.020507    0.022182    0.968752    09:23
    37        0.019944    0.025000    0.963409    14:15
    38        0.019134    0.026316    0.962307    06:44
    39        0.018094    0.032151    0.964344    06:25
    40        0.017827    0.025473    0.964131    05:09
    41        0.017791    0.026449    0.967641    04:25
    42        0.017935    0.029951    0.964231    04:54
    43        0.017605    0.034360    0.963989    04:32
    44        0.017242    0.028126    0.965754    05:38
    45        0.017434    0.028271    0.964929    05:06
    46        0.017079    0.029187    0.965438    05:48
    47        0.017186    0.029525    0.965751    06:16
    48        0.017077    0.029003    0.966066    06:46
    49        0.016688    0.030077    0.965412    04:39
    2024-04-10 00:42:35.306840       Export Model to *.pkl
    2024-04-10 00:42:36.041505       Done
  ```
  - Resnet18, Batch 8, Epoch 50 with image resolution as 1200x800 
  ```log
  Wed Apr 10 11:28:40 2024
  +-----------------------------------------------------------------------------+
  | NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7     |
  |-------------------------------+----------------------+----------------------+
  | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
  |                               |                      |               MIG M. |
  |===============================+======================+======================|
  +-------------------------------+----------------------+----------------------+
  |   4  NVIDIA RTX A6000    On   | 00000000:81:00.0 Off |                  Off |
  | 30%   38C    P8    20W / 300W |  46976MiB / 49140MiB |      0%      Default |
  |                               |                      |                  N/A |
  +-------------------------------+----------------------+----------------------+

  +-----------------------------------------------------------------------------+
  | Processes:                                                                  |
  |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
  |        ID   ID                                                   Usage      |
  |=============================================================================|
  |    4   N/A  N/A   2229241      C   python                          46973MiB |
  +-----------------------------------------------------------------------------+
  ```
  ```log
  (mothra) /mothra/data$ time python train.py
  2024-04-10 11:20:08.600215       Unique labels: {'background': 0, 'lepidopteran': 1, 'tags': 2, 'ruler': 3}
  2024-04-10 11:20:08.600613       Loading segmentation masks data for images: 80          Sample:  /home/rahul/workspace/vision/eeb/mothra/data/battus100/training_images/images/IMG_4810.JPG
  2024-04-10 11:20:23.726984       Creating Learner, initialized the UNET segmentation model
  2024-04-10 11:22:24.357764       Train model for epochs= 50
  epoch     train_loss  valid_loss  acc_camvid  time
  0         1.441202    1.229148    0.787606    04:13
  epoch     train_loss  valid_loss  acc_camvid  time
  0         0.673551    0.307634    0.575747    18:31
  Epoch 2/50 : |██████████████████████████████████████████████████--------------------------------------------------| 50.00% [1/2 04:46<04:46]
  ```
