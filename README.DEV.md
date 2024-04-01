
Mothra analyzes images of lepidopterans — mainly butterflies and moths — and measures their wing lengths. Using binarization techniques and calculating the resolution of ruler ticks, we read in images of lepidopterans and output the millimeter lengths of their wings.

##### Goal
- Adapted for UF Museum samples
- extract specific details and measurements of body sizes for each image in the UF Museum dataset "../UF_museum_data_2023/231017_Battus_philenor_polydamas_FLMNH.csv"



##### Dev Environment

```
conda env remove -n mothra
conda create -n mothra python=3.8 opencv jupyterlab -c conda-forge
conda activate mothra
pip install -r requirements.txt

```
```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
- binarization
```

```

##### Test on UF Battus samples

```
python pipeline_battus.py --detailed_plot -ar -i ./data/val_images/images -o ./data/results/val_images -csv ./data/results/val_images/results.csv
```
Issues: [TODO]
- Re-trained the segmentation model with new data "models/battus10_segmentation_test-4classes-resnet18-b2-e20.pkl" for clean ruler segments
- Ruler segments analysis code is not working. [In-Progress] 

```
PYTHONPATH=. pytest

```


##### Test on Original Mothra input_images

```
(mothra) rahul@karmax:/mnt/c/workspace/eebio/battus-museum/mothra-og/mothra-og$ python pipeline.py --detailed_plot -ar -i ./input_images/ -o ./input_images/result/ -csv  ./input_images/result/results.csv

Image 1/1 : BMNHE_500607.JPG
Couldn't determine EXIF image angle
Processing U-net...
Ruler row(min,max)=2569,3455, col(min,max)=0,5183
Focus shape (y,x)- (886, 5183)
T space - 91.12087912087912
x single - [3701, 3792.120879120879]
x multpl - [3701, 4612.208791208791]
Ruler row(min,max)=2569,3455, col(min,max)=0,5183
Focus shape (y,x)- (886, 5183)
T space - 91.12087912087912
x single - [3701, 3792.120879120879]
x multpl - [3701, 4612.208791208791]
Processing U-net...
Ruler row(min,max)=2569,3455, col(min,max)=0,5183
Focus shape (y,x)- (886, 5183)
T space - 91.12087912087912
x single - [3701, 3792.120879120879]
x multpl - [3701, 4612.208791208791]
Processing U-net...
Ruler row(min,max)=2569,3455, col(min,max)=0,5183
Focus shape (y,x)- (886, 5183)
T space - 91.12087912087912
x single - [3701, 3792.120879120879]
x multpl - [3701, 4612.208791208791]
Measurements:
* left_wing: 13.75 mm
* right_wing: 13.69 mm
* left_wing_center: 17.48 mm
* right_wing_center: 17.35 mm
* wing_span: 29.41 mm
* wing_shoulder: 3.92 mm
Identifying position and gender...
* Could not calculate position and gender
```


###### Output/Results

#### Validate accuracy
- Add wingspan, left, right measurements in "mm" to the battus 100 dataset 
    - subset the larger metadata excel sheet based on the filename [Rahul]
    - share the set with Vineesha for adding the actual measurement as ground truth information [Vineesha]
        - Add columns [image_id	left_wing (mm)	right_wing (mm)	left_wing_center (mm)	right_wing_center (mm)	wing_span (mm)	wing_shoulder (mm)]
- Run the analysis for all 100 images using mothra [Rahul]
    - evaluate the results accuracy based on the human labeled ground truth (length in mm)
    - tune or tweak the model to obtain the best accuracy possible given the measurements
- Run the model on all musuem specimens to generate results [Rahul]
- Spot check the results on random 100 images to verify the measurements correctness [Vineesha]