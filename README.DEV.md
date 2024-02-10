
Mothra analyzes images of lepidopterans — mainly butterflies and moths — and measures their wing lengths. Using binarization techniques and calculating the resolution of ruler ticks, we read in images of lepidopterans and output the millimeter lengths of their wings.

##### Goal
- Adapted for UF Museum samples
- extract specific details and measurements of body sizes for each image in the UF Museum dataset "../UF_museum_data_2023/231017_Battus_philenor_polydamas_FLMNH.csv"

```
python misc.py

0  --> Gray: 0    : Back 
1  --> Gray: 54   : Lepidopteran
2  --> Gray: 182  : Tag
3  --> Gray: 237  : Ruler
```

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
- Ruler segments are not working out. Need to train the segmentation model with new data

```
PYTHONPATH=. pytest

```