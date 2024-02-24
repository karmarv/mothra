
Mothra analyzes images of lepidopterans — mainly butterflies and moths — and measures their wing lengths. Using binarization techniques and calculating the resolution of ruler ticks, we read in images of lepidopterans and output the millimeter lengths of their wings.

##### Goal
- Adapted for UF Museum samples
- extract specific details and measurements of body sizes for each image in the UF Museum dataset "../UF_museum_data_2023/231017_Battus_philenor_polydamas_FLMNH.csv"



##### Dev Environment

```
conda create -n mothra python=3.8 opencv jupyterlab -c conda-forge
conda activate mothra
pip install -r requirements.txt
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