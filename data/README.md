
# Battus Museum Data 

#### CVAT Labeling and Data Setup
- Labels: tags, ruler, lepidopteran 
    ```
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
- UNET model for semantic segmentation `train.ipynb`
    - Learner export Model saved upon training on Battus10 dataset
      - Epoch10 `training_images/battus10_segmentation_test-4classes-resnet18-b2-e10.pkl`
      - Epoch20 `training_images/battus10_segmentation_test-4classes-resnet18-b2-e20.pkl`
    - Train the models on Battus100 dataset for epoch50
      - Epoch50 [TODO]
      