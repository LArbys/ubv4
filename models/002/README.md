# Model 2

## Design

Got a little further in implementing resnet inception v4. Now up to reduction B.  We are now training only one plane at a time. Uses silence to ignore the planes not in use.

## Training

Using a bigger weight decay.  Tried plane 1 with 0.01 and plane 0 with 0.001. In progress.

### Plane 1, Attempt 1: weight decay 0.01, batch size 24.  

Seems to have gotten stucked. So stopped it to lower weight decay to 0.01.

![Plane 1, Attempt1](https://github.com/LArbys/ubv4/blob/master/models/002/training_plot_plane1_attempt1.png)

## Results

Coming.

## Take aways

Probably won't work.
