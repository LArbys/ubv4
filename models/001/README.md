# Model 1

## Design 

This used a truncated ResNet inception v4 network.  
Only got up to reduction A due to time before I had to leave for Lindley's group party.  
It uses all three views and merges them very late.

## Training

It trained to >85% accuracy on the validation set.  The training set was 95% or maybe higher.

![Training](https://github.com/LArbys/ubv4/blob/master/models/001/training_plot.png)

## Results

Couldn't really find neutrinos in data.
In the EXTBNB set, there was a fairly big spike at 1.0.  Performed placed-in analyses of EXTBNB, BNB and validation set neutrino events.
Can't know for sure of course, but network really relied on U plane (assuming BGR coloring by OpenCV 3).  
Collection plane (red) played a role in ID some events. The V plane did not.
The parts where the red plane really actiated the network was on strong looking muon tracks, not really near the vertex.
In general, there were a lot of non-vertex, cosmic regions that looked neutrino-like to the network.

### MC Patch study

Did a study where only a patch is given to the network.  Gave patches for one plane at a time.  The color shading indicated which plane patch provided a high neutrino score. Processed 100 MC Neutrino+Cosmic overlay events. You can find them in the images_placedin_numc folder. *NOTE: bounding boxes are fubar*

* Example of the network doing something reasonable:
    Notice how the blue (U) and red (Y) planes have color patches. These areas gave high neutrino score.  But why no green patch (from the V plane).  The lack of green patch over obvious neutrinos is a pattern.
    ![NuMC OK](https://github.com/LArbys/ubv4/blob/master/models/001/images_placedin_numc/NUMC_1.PNG)

* Example of network doing something not-so-good:
   This seems to be a single proton event. Not the easiest thing to try and find. And here, it doesn't find it. (Note it's just the blue again)
   ![Single proton event](https://github.com/LArbys/ubv4/blob/master/models/001/images_placedin_numc/NUMC_22.PNG)

* Finally the network looks at a green track -- but's a stopping muon
   ![Stopping Mu on V plane](https://github.com/LArbys/ubv4/blob/master/models/001/images_placedin_numc/NUMC_99.PNG)

* V place network activates on some noise feature
     Events like this make me think that high V plane ADC values maybe throw the network off? More weight decay to fix this?
     ![V plane noise](https://github.com/LArbys/ubv4/blob/master/models/001/images_placedin_numc/NUMC_3.PNG)
    


## Take aways for next try

The off-regions, could they be do to too much overtaining?  In principle, these off-vertex events should have rare patterns.
Could increase weight decay to mitigate them.

Also, how to get all three planes involved?
Could try training them separately, then fine-tuning a FC layer that concats the features from the three planes.
Is the quick size reduction of the image be hurting recognition?
