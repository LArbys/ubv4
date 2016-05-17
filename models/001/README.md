# Model 1

## Design 

This used a truncated ResNet inception v4 network.  
Only got up to reduction A due to time before I had to leave for Lindley's group party.  
It uses all three views and merges them very late.

## Training

It trained to >85% accuracy on the validation set.  The training set was 95% or maybe higher.


## Results

Couldn't really find neutrinos.
In the EXTBNB set, there was a fairly big spike at 1.0.  Performed placed-in analyses of EXTBNB, BNB and validation set neutrino events.
Can't know for sure of course, but network really relied on U plane (assuming BGR coloring by OpenCV 3).  
Collection plane (red) played a role in ID some events. The V plane did not.
The parts where the red plane really actiated the network was on strong looking muon tracks, not really near the vertex.
In general, there were a lot of non-vertex, cosmic regions that looked neutrino-like to the network.

## Take aways for next try

The off-regions, could they be do to too much overtaining?  In principle, these off-vertex events should have rare patterns.
Could increase weight decay to mitigate them.

Also, how to get all three planes involved?
Could try training them separately, then fine-tuning a FC layer that concats the features from the three planes.
Is the quick size reduction of the image be hurting recognition?
