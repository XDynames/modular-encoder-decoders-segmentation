# Pytorch Lightning for Segmentation Models
## Run
An example of running a training job can be found in `train_segmentation.sh`.
Make sure to include your w&b API key and adjust any filepaths and hyperparameters here.

## Docker
Modify the `docker/run.sh` file to mount your data to the container.
Build the runtime container using the helper script `./docker/build.sh`.
Run the container with `docker/run.sh`.

## Custom Datasets
To add a custom dataset you need to do a few things.
First define the dataset class and add it to the `build_dataset()` function in `datasets/segmentation.py` by adding it as a return in the switch statement.
Next add the revelant information to the `db_info` dictionary in `datasets/segmentation.py`.
An example of a valid entry would be:
```
"cityscapes": {
    "n_classes": 19,
    "size": [768, 768], # [height, width]
    "ignore_index": -100,
    "class_labels": CITYSCAPES_SEG_CLASSES,
    "normalisation": [[0.288, 0.327, 0.286], [0.190, 0.190, 0.187]],
}
```
You can leave the `"class_labels"` field for the dataset equal to `None`, this will log each classes IoU as a number from [0, n_classes-1].
If you want to have IoU logged with the pixel classifications natural language name you can define a dictionary in containing the details of each class.
An example of PascalVOC can be found in `/datasets/constants.py`.
Once the dataset if defined, added to `build_dataset()` and has an entry in `db_info` you can use it in `train_segmentation.sh` as the `--dataset-name`.