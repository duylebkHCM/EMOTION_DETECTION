# EMOTION_DETECTION
> Using deeplearning, CNN, OpenCV to recognize emotion on human face

### Dataset
[FER-2013](https://www.kaggle.com/deadskull7/fer2013)

## Usage

1. Clone project into a folder
2. Create another folder call fer2013, include: output, hdf5, fer2013
3. Download dataset and put it inside /fer2013/fer2013

### Build dataset
Use command
```
$ python build_dataset.py
```
### Train model
Create a folder call 'checkpoints'

Use command

```
$ python train_recognizer.py --model checkpoints
```

### Evaluate model

Use command
```
$ python test_recognizer.py --model checkpoints
```

### Test on image

Use command
```
$ python emotion_image_detector.py --testImage <path_to_test_folder> --model checkpoints
```
> Some result

![Image](https://github.com/duylebkHCM/EMOTION_DETECTION/blob/master/resultexample/Screenshot%20from%202020-05-20%2022-31-47.png?raw=true)
![Image](https://github.com/duylebkHCM/EMOTION_DETECTION/blob/master/resultexample/Screenshot%20from%202020-05-20%2022-32-03.png)

### Test video stream
Use command
```
$ python emotion_detector.py --model checkpoints
```
## Experiment

After training for 100 epochs
![Image](https://github.com/duylebkHCM/EMOTION_DETECTION/blob/master/Duynet_emotion.png)
