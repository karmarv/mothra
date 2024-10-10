
This is a collaboration between Chaturvedi Lab at Tulane EEB and Rahul Vishwakarma, and image and vision processing scientist at Hitachi America Ltd. Mothra analyzes images of lepidopterans — mainly butterflies and moths — and measures their wing lengths. Using binarization techniques and calculating the resolution of ruler ticks, we read in images of lepidopterans and output the millimeter lengths of their wings. 

> Reference: Mothra [README.md](./README.OG.md) from https://github.com/machine-shop/mothra

## Goal
- Measure Battus in UF Museum samples
- Extract specific details and measurements of body sizes for each image in the UF Museum dataset "../UF_museum_data_2023/231017_Battus_philenor_polydamas_FLMNH.csv"

**Approach**: Mothra
>  Vision Dataset (CVAT) > Segmentation Model (UNet) > Measurement Pipeline (T-space Fourier)

## Tasks: 
- [x] Curate the Battus100 dataset
  - Image level segmentation mask for segmentation vision model training [Feb/26/2024]
  - Image level measurements for pipeline evaluation [Apr/08/2024] 
- [x] Ruler segments analysis code changes [Mar/20/2024] 
  - Updated the tick-space computation to measure the pixel distance between the `mm` gradings
- [x] Re-train the segmentation model with new data [Apr/08/2024][Apr/10/2024]
  - Latest model with Resnet18 to be downloaded in [./model/*.pkl](./models/battus100_segm_c4_resnet18_b8_e50_s1200x800.pkl) folder
  - Model training instructions in [./data/README.md](./data/README.md)
- [x] Compute the vision model foreground accuracy metrics on test images [Apr/11/2024] [(reference)](#c-evaluate-vision-model)
  - Edit and evaluate using [./data/test.py](./data/test.py)
- [x] Run measurement pipeline on test images and report results [Apr/11/2024] [(reference)](#a-evaluate-measurement-pipeline)
  - Updated results in [./data/battus100/results/val_images/results.csv](./data/battus100/results/val_images/results.csv)
- [ ] Compare predictions with the manual measured ground truth [Apr/12/2024] [(reference)](#b-compare-the-predicted-results-to-manual-labels)
  - Tested on 3 evaluation ground truth samples using the [./result_plotting.py](./result_plotting.py)
  - Evaluate for outliers on Battus100 ground truth samples
- [ ] Run the model on all museum specimens to generate results 
- [ ] Spot check the results on random images for verification

---

### Environment Setup
- Python 3.8.18 Installation
  - Instructions via `Miniconda v23.1.0` - https://docs.conda.io/projects/miniconda/en/latest/ 
  - Create a virtual environment named "`mothra`" for this analysis
    ```bash
    conda env remove -n mothra
    conda create -n mothra python=3.8
    conda activate mothra
    ```
- Clone the current codebase - `git clone https://github.com/karmarv/mothra.git && cd mothra`
- Install pre-requisite packages in the activated python virtual environment using -  ` pip install -r requirements.txt`
- Download the model
  ```bash
  wget https://github.com/karmarv/mothra/releases/download/v0.2/battus100_segm_c4_resnet18_b8_e50_s1200x800.pkl -P ./models
  ```


### (A.) Evaluate measurement pipeline
- Overall measurement pipeline test run on [./data/battus10/val_images/](./data/battus10/val_images/)
  ```
  time python pipeline_battus.py --detailed_plot -ar -i ./data/battus10/val_images/images -o ./data/battus10/results/val_images -csv ./data/battus10/results/val_images/results.csv
  ```
  - Result output in 4m29s
    ```log
    Image 1/2 : IMG_3870.JPG
    Cannot evaluate orientation for ./data/battus10/val_images/images/IMG_3870.JPG.
    Couldn't determine EXIF image angle
    Skip weight check and use the local weights:  models/battus100_segm_c4_resnet18_b8_e50_s1200x800.pkl
    Processing U-net...
    Estimated ruler t-unit space in pixels - 31.27
    Measurements:
    * left_wing: 50.72 mm
    * right_wing: 50.31 mm
    * left_wing_center: 60.55 mm
    * right_wing_center: 60.57 mm
    * wing_span: 82.88 mm
    * wing_shoulder: 3.2 mm
    Identifying position and gender...
    * Could not calculate position and gender

    ```
  - ![alt text](./data/battus10/results/val_images/IMG_3870.JPG "Sample 1")
    ```log
    Image 2/2 : IMG_4541.JPG
    Cannot evaluate orientation for ./data/battus10/val_images/images/IMG_4541.JPG.
    Couldn't determine EXIF image angle
    Skip weight check and use the local weights:  models/battus100_segm_c4_resnet18_b8_e50_s1200x800.pkl
    Processing U-net...
    Estimated ruler t-unit space in pixels - 31.06
    Measurements:
    * left_wing: 45.17 mm
    * right_wing: 44.75 mm
    * left_wing_center: 53.6 mm
    * right_wing_center: 54.01 mm
    * wing_span: 81.7 mm
    * wing_shoulder: 4.71 mm
    Identifying position and gender...
    * Could not calculate position and gender
    ```
  - ![alt text](./data/battus10/results/val_images/IMG_4541.JPG "Sample 1")

- Pipeline run on Battus100 validation images
  ```
  time python pipeline_battus.py --detailed_plot -ar -i ./data/battus100/val_images/images -o ./data/battus100/results/val_images -csv ./data/battus100/results/val_images/results.csv
  ```
  - Check the pipeline run log at [./data/battus100/results/val_images/eval_b100.log](./data/battus100/results/val_images/eval_b100.log)


### (B.) Compare the predicted results to manual labels
- Use the available result plotting utility 
  ```
  python result_plotting.py --actual data/battus10/val_images/manual_labels.csv  --name "image_id" --left "left_wing (mm)" --right "right_wing (mm)" --predicted data/battus10/results/val_images/results.csv --comparison --outliers
  ```
  ```log
  DIFFERENCE STATISTICS
      Mean Differences: -9.501666666666667
      Differences SD: 11.37668290359228.
      Lower Bound (-2 SD) of Differences: -32.25503247385123
      Upper Bound (+2 SD) of Differences: 13.251699140517895
      Number of outlying measurements: 0
      Number of images with outlying measurements: 0

  Saved plot of differences to result_plot.png
  Saved all differences to comparison.csv
  Saved 0 rows to outliers.csv
  ```


### (C.) Evaluate the core image segmentation model
- Segmentation model test on [./data/battus10/val_images/](./data/battus10/val_images/)
  - configure the image and mask folder path in the [./data/test.py](./data/test.py) code
  ```
  python test.py
  ```
  - Score meaasured is the foreground segmentation accuracy
    ```log
    2024-04-11 10:40:36.542821       Loading U-net model from  /home/rahul/workspace/vision/eeb/mothra/data/battus100/training_images/battus100_segm_c4_resnet18_b8_e50_s1200x800.pkl
    2024-04-11 10:40:36.950398       Test ->  /home/rahul/workspace/vision/eeb/mothra/data/battus10/val_images/images/IMG_2895.JPG   Mask ->  /home/rahul/workspace/vision/eeb/mothra/data/battus10/val_images/labels/IMG_2895.png
    2024-04-11 10:41:03.363294       - Score: 0.969103194841159
    2024-04-11 10:41:03.363399       Test ->  /home/rahul/workspace/vision/eeb/mothra/data/battus10/val_images/images/IMG_3870.JPG   Mask ->  /home/rahul/workspace/vision/eeb/mothra/data/battus10/val_images/labels/IMG_3870.png
    2024-04-11 10:41:28.144009       - Score: 0.9921218477296475
    2024-04-11 10:41:28.144164       Test ->  /home/rahul/workspace/vision/eeb/mothra/data/battus10/val_images/images/IMG_4541.JPG   Mask ->  /home/rahul/workspace/vision/eeb/mothra/data/battus10/val_images/labels/IMG_4541.png
    2024-04-11 10:41:40.032401       - Score: 0.9903594628986084
    ```

--- 


