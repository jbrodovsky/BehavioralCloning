# Behavioral Cloning Project

Overview
---

In this project, I used what I learned about deep neural networks and convolutional neural networks to clone driving behavior. I trained, validated, and tested a model using Keras. The model outputs a steering angle to an autonomous vehicle. Training data was collected using a simulator where I steered a car around a track. I used image data and steering angles to train a neural network and then used this model to drive the car autonomously around the track.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

## Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

## Architecture and Training Documentation

The network selected was the NVIDIA model. This model has been used for self-driving cars in the past and proved to work well for this project. There were a few modifications made to the model. Initially, a Lambda layer was added prior to the network proper to assist in standardizing and normalizing the images into the format that the NVIDIA model expects. An Adam optimizer was used with a learning rate of 0.0001.

The raw training data consisted of a single lap around the track. During this manual drive, I attempted to keep the car as close to the center of the track as possible. 20% of this data will be split off for validation purposes. The ‘test’ set will consist of the live autonomous test.

During training, the raw data is preprocessed and augmented. The preprocessing cropped out the sky and the front of the car, resized the image to 66 x 200, and converted the color encoding to YUV to bring this in line with what the NVIDIA model is expecting for input.

Next the data was augmented. For a left (or right) image the steering angle was increased (or decreased) by 0.2. Next, the data was randomly flipped about the vertical axis and the steering angle reversed. Finally, the data was randomly translated. The amount of translation was also randomized and the steering angle adjusted proportionally to the amount of translation.

## Model Architecture and Training Strategy

After an initial failed attempt to autonomously drive around the track (run1.mp4), a dropout layer was added after the final convolution that kept 50% of the data. Initially, this model was implemented with RELU activation and no Python generator. This initial implementation was moderately successful however there were some turns that had a tire leave the driving surface.

On the second attempt, the additional dropout layer was added after the final convolution and the activation was changed to ELU for each of the convolution layers. A Python generator was used to speed up the training process by only bringing a single batch of images at a time. The generated used a batch size of 32 and took 1000 steps per epoch. The final model that was submitted successfully drove the car around the track fully keeping the car safely on the road surface. The final model (model.h5) and video of the run (run2.mp4) are included in this repo.
