**Automated Glacial Lake Detection using DeepLabV3+**



Team Information

Team Name: **Trixel**



**Team Members**



1. Hariram J
2. Manoj Kumar S
3. Praveen A M



**Institution**

Thiagarajar College of Engineering



**Project Overview**



This project presents an automated framework for glacial lake detection from satellite imagery using DeepLabV3+ with an EfficientNet-B4 backbone.



The proposed system performs semantic segmentation to identify glacial lakes while minimizing confusion with snow, glacier ice, terrain shadows, and debris-covered regions.



**Objectives**

* Automated Glacial Lake Detection
* Semantic Segmentation of Lake Regions
* Lake Mapping and Visualization
* Lake Area Estimation
* Support for GLOF Monitoring



**Methodology**



Satellite Images



↓



Manual Annotation



↓



Binary Mask Generation



↓



Data Augmentation



↓



DeepLabV3+ Training



↓



Lake Detection and Mapping



**Model Architecture**

* DeepLabV3+
* EfficientNet-B4 Backbone
* BCE + Tversky Loss
* AdamW Optimizer
* Cosine Annealing Scheduler



**Performance**

**Metric	Value**



* Accuracy	99.62%
* Precision	98.09%
* Recall	93.49%
* F1 Score	95.74%
* IoU	91.82%
* Cohen's Kappa	0.955

YouTube Presentation Link:

https://youtu.be/mtdjzp9I9RM?si=xKIOm7rVx6Llb4nn

**Repository Structure**



* train.py : Training pipeline
* inference.py : Prediction pipeline
* model\_architecture.py : Model definition
* utils.py : Utility functions
* notebook.ipynb : Full implementation
* best\_glacial\_model.pth : Trained model
* masks/ : Segmentation masks
* outputs/ : Prediction outputs
* Installation
* pip install -r requirements.txt
* Challenge



GLOFeagles'26 Challenge



**Authors**



1. Hariram J
2. Manoj Kumar S
3. Praveen A M

