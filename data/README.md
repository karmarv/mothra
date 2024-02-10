
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
- Export the project as CamVid 
    - [Insert Screenshot]


### FastAI Training Notebook
- UNET model for semantic segmentation `train.ipynb`
