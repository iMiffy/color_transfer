Color transfer between images
==============

There are 2 implementations of <code>color_transfer</code>:
1. Color Transfer between Images by Reinhard et al. (2001). The algorithm itself is extremely efficient (much faster than histogram based methods), requiring only the mean and standard deviation of pixel intensities for each channel in the L\*a\*b\* color space.
2. Image Style Transfer Using Convolutional Neural Networks by Gatys et al. (2016).A pre-trained VGG-19 network is used as a feature extractor for the content and style images via correlations betwen the different filter responses over the spatal extent of the respective feature maps. Tuning the hyperparameters drastically will yield effects similar to color transfer results by Reinhard et al.

#Requirements
- opencv
- numpy
- pillow
- pytorch


#Examples
Below are some example showing how to run the <code>main.py</code> and the associated color transfers between images.

<code>$ python main.py --source images/autumn.jpg --target images/fallingwater.jpg --mode reinhard --output output/autumn_water_reinhard.jpg</code>
<code>$ python main.py --source images/autumn.jpg --target images/fallingwater.jpg --mode vgg --out-size 1000 --output output/autumn_water_vgg.jpg</code>

![Reinhard water screenshot](/output/autumn_water_reinhard.jpg?raw=true)
![VGG water screenshot](/output/autumn_water_vgg.jpg?raw=true)

<code>$ python main.py --source images/ocean_sunset.jpg --target images/ocean_day.jpg --mode reinhard --output output/ocean_reinhard.jpg</code>
<code>$ python main.py --source images/ocean_sunset.jpg --target images/ocean_day.jpg --mode vgg --out-size 669 --output output/ocean_vgg.jpg</code>

![Reinhard ocean screenshot](/output/ocean_reinhard.jpg?raw=true)
![VGG ocean screenshot](/output/ocean_vgg.jpg?raw=true)
