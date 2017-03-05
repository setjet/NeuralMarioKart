# NeuralMarioKart
A set of scripts to record a dataset of actions and pixels fetched from a nintendo 64 emulator and xbox controller, which are then used to train a neural network that is able to play mariokart.

It is able to autonomously drve around the tracks with a human level performance
![MarioRaceway.gif](https://giphy.com/gifs/l4FGAwbQKvWDwHxba)

![Jungle.gif](https://giphy.com/gifs/l4FGrIIvbzkJS0ni8)

![RoyalRaceway.gif](https://giphy.com/gifs/3oKIPsKaFQEsk8jR3a)

A complete game can be found here:
https://youtu.be/g_ZIsCe1COk

## Neural network
The neuralnet used is a LeCunn style feed forward network with a couple convolution layers. Because we are dealing with a live video game, it is key that the network is able to provide an action given an input image almost instantly, even running on a CPU. To achieve this, the network is rather small not only because it not very deep, but also because the input image is sampled down.

## Setup
Simply running the bash script in the setup directory will set everything up. Simply calling play.py will start the emulator. You can then calibrate the capturing window by tweaking the parameters in the config.

## Build data set
To build a data set, simply run play.py and the build_dataset.py simultaneously.

## Train network and play
The model.py script will train the neural network for you. Once its completed, simply start the agent by calling 'python play.py NN'. You can override the agent by pressing RB on your xbox controller. 

## References
Mupen64plus emulator, N65 plugin by Kevin Hughes 
