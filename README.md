# Pedestrian Intension Detection with Hybrid Feature Fusion
The project is based on [Pedestrian Stop and Go Forecasting with Hybrid Feature Fusion](https://github.com/vita-epfl/hybrid-feature-fusion).
## Introduction
The goal of pedestrian intention prediction is to determine, for each prediction timestep, whether a given pedestrian will be crossing or non-crossing the road based on a sequence of past observations. 

![Task](figure/task.png)

### Contribution Overview
Drawing inspiration from the *Pedestrian Stop and Go Forecasting with Hybrid Feature Fusion* [1], which focuses on state transition prediction, we adapt their model and approach to our specific crossing/non-crossing task. The model including four modules: **visual information** (encode with CNN), **position and relative velocity** (bounding box), **pedestrain behavior**, **scene description**.
![Model](figure/model.png)
The model employs individual LSTM units for three of modalities and applies a hybrid fusion technique, combining linear projections and concatenations, to integrate the multi-modal embeddings and obtain the final prediction.

Furthermore, we substitute the CNN backbone (Resnet18) with MobileNet to leverage the efficiency of separable convolutions. This prioritizes improving the model's real-time performance by increasing speed rather than emphasizing quality.

## Experimental Setup

## Dataset
**JAAD** [2] has been selected as the dataset. **JAAD** focuses on investigating pedestrian road crossing behaviors using a dataset comprising 346 videos which encompassing a range of weather and lighting conditions. Each pedestrian in the dataset is annotated with bounding boxes, behavioral data, and demographic information. 
### Label
- **crosing/non-crossing**: {0: 'not-crossing', 1: 'crossing'}
The label assigned to a given sequence of past observations corresponds to the cross/non-cross label of the given prediction timestep.
### Input
- **visual context**: (channel x image height x image width)
a sequence of RGB images cropped with corresponding pedestrian bounding box and backbround information. 
- **bounding boxes and relative velocities**: (x<sub>t</sub>, y<sub>t</sub>, H<sub>t</sub>, W<sub>t</sub>, $\Delta x$<sub>t</sub>, $\Delta y$<sub>t</sub>, $\Delta H$<sub>t</sub>, $\Delta W$<sub>t</sub>)
P<sub>t</sub> = (x<sub>t</sub>, y<sub>t</sub>, H<sub>t</sub>, W<sub>t</sub>) -> bounding box of corresponding pedestrian
V<sub>t</sub> = ($\Delta$x<sub>t</sub>, $\Delta$y<sub>t</sub>, $\Delta$H<sub>t</sub>, $\Delta$W<sub>t</sub>) = (x<sub>t</sub> - x<sub>t-1</sub>, y<sub>t</sub> - y<sub>t-1</sub>, H<sub>t</sub> - H<sub>t-1</sub>, W<sub>t</sub> - W<sub>t-1</sub>) -> relative velocity of the pedestrian
- **behavioural data**: (b<sup>t</sup><sub>look</sub>, b<sup>t</sup><sub>nod</sub>, b<sup>t</sup><sub>hand</sub>)
b<sup>t</sup><sub>look</sub> -> {0: 'not-looking', 1: 'looking'}
b<sup>t</sup><sub>nod</sub> -> {0: 'undefined', 1: 'nodding'}
b<sup>t</sup><sub>hand</sub> -> {0: 'undefined', 1: 'greet',2: 'yield', 3: 'rightofway', 4: 'other'}
** To avoid information leakage, we removed the binary label indicating walking/non-walking.
- **scene descriptions**:  (s<sub>tl</sub>, s<sub>in</sub>, s<sub>de</sub>, s<sub>si</sub>, s<sub>td</sub>)
s<sub>tl</sub> -> number of traffic lanes
s<sub>in</sub> ->  if is an intersection
s<sub>de</sub> -> if an intersection is designated with a zebra crossing or a traffic signal
s<sub>si</sub> ->  if an intersection is signalized
s<sub>td</sub> ->  traffic direction (one/two-way) 
** To avoid information leakage, we removed the label indicating pedestrian
motion direction (lateral or longitudinal).
### Output
- **p<sub>t</sub>**: the probability of crossing for each of the prediction timesteps



## Installation

## Train

## Inference

## Results

## Conclusion

## Reference
[1] Dongxu Guo, Taylor Mordan, and Alexandre Alahi. *Pedestrian Stop and Go Forecasting with Hybrid Feature Fusion*. 2022. [arXiv: 2203.02489 [cs.CV]](https://arxiv.org/abs/2203.02489) .
[2] Amir Rasouli, Iuliia Kotseruba, and John K. Tsotsos. “Are They Going to Cross? A Benchmark Dataset and Baseline for Pedestrian Crosswalk Behavior”. In: 2017 *IEEE International Conference on Computer Vision Workshops (ICCVW)*. 2017, pp. 206–213. doi: [10.1109/ICCVW.2017.33](https://ieeexplore.ieee.org/document/8265243).