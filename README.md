# Lifting Tracker
The goal of this project is to analyse the path of a bar in a video recording of a lifting movement (e.g. deadlift, squat, bench press) with the results then visualised as charts and key values.

## Implementation
The video processing is done using OpenCV in Python and uses colour detection to track the desired objects movement. The current development method is using a tennis ball as the 'known object' as it has a distinct colour to track, and the known size allows for speed and distance calculations to be derived.