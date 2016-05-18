# Model 2

## Design

Got a little further in implementing resnet inception v4. Now up to reduction B.  We are now training only one plane at a time. Uses silence to ignore the planes not in use.

## Training

### Plane 0

#### Attempt 1: weight decay 0.001, batch size 10. Seems to have trained.

![Plane 0,Attempt1](https://github.com/LArbys/ubv4/blob/master/models/002/plane0/training_plot_plane0.png)


### Plane 1

#### Attempt 1: weight decay 0.01, batch size 24.  

To start, ssing a bigger weight decay.  Tried plane 1 with 0.01 and plane 0 with 0.001. In progress.

Seems to have gotten stucked. So stopped it to lower weight decay to 0.001.

![Plane 1, Attempt1](https://github.com/LArbys/ubv4/blob/master/models/002/training_plot_plane1_attempt1.png)

#### Attempt 2: weight decay 0.0001, batch size 32

This time I finished the inception resnet-v2 model.  Only 3 copies of modules B and C, however. Also, I added a 7x7, stride 3 filter at the very beginning. This served to extract bigger features and to shrink the output to the size of images that inception-resnet-v2 was working with in their paper.  This saved a lot of memory and allowed me to make the network deeper while using a bigger batch size. I also trained the data using a truncated ADC range, [0.45,3.55], and event-by-event smearing at the 0.01 level.  Note that I did not turn on pixel-by-pixel smearing.

The result is that the network learned, but overtrainined. I am thinking that maybe adding further augmentation might help, just like it did in the 12 channel experiment (which kind of worked).

![Plane 1, Attempt 2](https://github.com/LArbys/ubv4/blob/master/models/002/plane1/training_plot_plane1.png)

### Plane 2

#### Attempt 1: weight decay 0.001, batch size 10. Still in Progress.

Still training, albeit slowly.

![Plane 2, Attemp 1](https://github.com/LArbys/ubv4/blob/master/models/002/plane2/training_plot_plane2.png)

## Results

### Validation Set Neutrino Score Distributions

#### Plane 1

For attempt 2. Looks OK.  That long neutrino tail is probably events that I need to filter (single proton events).  Also, maybe the neutrino distribution looks a little overtrained (maybe it's a bit too spikey?).


<img src=https://github.com/LArbys/ubv4/blob/master/models/002/plane1/model2_plane1_attempt2_valscores.png width=400>

## Take aways

Probably won't work.
