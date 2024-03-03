
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
- UNET model for semantic segmentation [`train.ipynb`](./train.ipynb)
  - References:
    - FastAI - https://colab.research.google.com/github/visual-layer/vl-datasets/blob/main/notebooks/train-fastai.ipynb
    - FastAI - https://github.com/ashutoshraj/Dynamic-Unet/blob/master/caravan-work.ipynb
  - Learner export Model saved upon training on Battus10 dataset
    - Epoch10 `training_images/battus10_segmentation_test-4classes-resnet18-b2-e10.pkl`
    - Epoch20 `training_images/battus10_segmentation_test-4classes-resnet18-b2-e20.pkl`
- Train the models on Battus100 dataset for epoch50 [`train.py`](./train.py)
  - Epoch50 [TODO]
  ```
  (mothra) rahul@gpu-server:~/workspace/vision/eeb/mothra/data$ python train.py
    Unique labels: {'background': 0, 'lepidopteran': 1, 'tags': 2, 'ruler': 3}
    2024-03-03 13:26:09.117124       Total Images: 8         Sample:  /home/rahul/workspace/vision/eeb/mothra/data/battus10/training_images/images/IMG_1763.JPG
    2024-03-03 13:26:17.428410       Creating Learner, loading the model
    2024-03-03 13:27:04.826933       Train model for epochs= 20
    epoch     train_loss  valid_loss  time
    Traceback (most recent call last):-------------------------------------------------------------------------------| 0.00% [0/7 00:00<?]
    File "train.py", line 39, in <module>
        learner.fine_tune(num_epochs)
    File "/home/rahul/miniconda/envs/mothra/lib/python3.8/site-packages/fastai/callback/schedule.py", line 165, in fine_tune
        self.fit_one_cycle(freeze_epochs, slice(base_lr), pct_start=0.99, **kwargs)
    File "/home/rahul/miniconda/envs/mothra/lib/python3.8/site-packages/fastai/callback/schedule.py", line 119, in fit_one_cycle
        self.fit(n_epoch, cbs=ParamScheduler(scheds)+L(cbs), reset_opt=reset_opt, wd=wd, start_epoch=start_epoch)
    File "/home/rahul/miniconda/envs/mothra/lib/python3.8/site-packages/fastai/learner.py", line 264, in fit
        self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)
    File "/home/rahul/miniconda/envs/mothra/lib/python3.8/site-packages/fastai/learner.py", line 199, in _with_events
        try: self(f'before_{event_type}');  f()
    File "/home/rahul/miniconda/envs/mothra/lib/python3.8/site-packages/fastai/learner.py", line 253, in _do_fit
        self._with_events(self._do_epoch, 'epoch', CancelEpochException)
    File "/home/rahul/miniconda/envs/mothra/lib/python3.8/site-packages/fastai/learner.py", line 199, in _with_events
        try: self(f'before_{event_type}');  f()
    File "/home/rahul/miniconda/envs/mothra/lib/python3.8/site-packages/fastai/learner.py", line 247, in _do_epoch
        self._do_epoch_train()
    File "/home/rahul/miniconda/envs/mothra/lib/python3.8/site-packages/fastai/learner.py", line 239, in _do_epoch_train
        self._with_events(self.all_batches, 'train', CancelTrainException)
    File "/home/rahul/miniconda/envs/mothra/lib/python3.8/site-packages/fastai/learner.py", line 199, in _with_events
        try: self(f'before_{event_type}');  f()
    File "/home/rahul/miniconda/envs/mothra/lib/python3.8/site-packages/fastai/learner.py", line 205, in all_batches
        for o in enumerate(self.dl): self.one_batch(*o)
    File "/home/rahul/miniconda/envs/mothra/lib/python3.8/site-packages/fastai/learner.py", line 235, in one_batch
        self._with_events(self._do_one_batch, 'batch', CancelBatchException)
    File "/home/rahul/miniconda/envs/mothra/lib/python3.8/site-packages/fastai/learner.py", line 199, in _with_events
        try: self(f'before_{event_type}');  f()
    File "/home/rahul/miniconda/envs/mothra/lib/python3.8/site-packages/fastai/learner.py", line 216, in _do_one_batch
        self.pred = self.model(*self.xb)
    File "/home/rahul/miniconda/envs/mothra/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
    File "/home/rahul/miniconda/envs/mothra/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
        return forward_call(*args, **kwargs)
    File "/home/rahul/miniconda/envs/mothra/lib/python3.8/site-packages/fastai/layers.py", line 409, in forward
        nres = l(res)
    File "/home/rahul/miniconda/envs/mothra/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
    File "/home/rahul/miniconda/envs/mothra/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
        return forward_call(*args, **kwargs)
    File "/home/rahul/miniconda/envs/mothra/lib/python3.8/site-packages/fastai/vision/models/unet.py", line 50, in forward
        x = F.interpolate(x, x.orig.shape[-2:], mode=self.mode)
    File "/home/rahul/miniconda/envs/mothra/lib/python3.8/site-packages/torch/nn/functional.py", line 4001, in interpolate
        return torch._C._nn.upsample_nearest2d(input, output_size, scale_factors)
    torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 34.17 GiB. GPU 0 has a total capacity of 47.54 GiB of which 19.10 GiB is free. Including non-PyTorch memory, this process has 28.44 GiB memory in use. Of the allocated memory 26.76 GiB is allocated by PyTorch, and 257.77 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)  
  ```
  