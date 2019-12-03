Unsupervised Conditional Reflex Learning Based on Convolutional Spiking Neural Network and Reward Modulation
============
## Model
### Structure
![structure](images/structure.jpg)

## Requirements
1. [NEST](https://www.nest-simulator.org/)
2. [Pytorch](https://pytorch.org/)
3. [SkypeTorch](https://github.com/miladmozafari/SpykeTorch) 
4. [V-REP (simulation platform)](http://www.coppeliarobotics.com/)
5. [ROS(interface between simulation environment and the script)](https://www.ros.org/)

## How to run
1. generate noise-like data with opencv and put the data into *./images*
2. train the feature-extract unit with the generated noise-like data in an unsupervised manner:
    ```angular2html
    python ./feature_extract_unit/train_with_noiselikedata.py
    ```
3. train the decision-making unit:
    ```angular2html
    python ./decision_making_unit/train.py  
    ```
    - parameters of the network can be changed in *./decision_making_unit/parameters.py*
    
    - **note:** *./decision_making_unit/environment_1.py* corresponds to scenario 1~3 and *./decision_making_unit/environment_2.py* corresponds to 
    scenario 4.



## Some Results
### A. Visualization of the feature-extraction unit
![visualization1](images/visualization1.jpg)
### B. Visualization of the decision-making unit
![visualiztion2](images/visualization2.jpg)
### C. Training period
![train](images/training_period.jpg)
### D. Testing performance
![test_comparison](images/testing_comparison.jpg)
![test_visualiztion](images/testing_visualiztion.jpg)
### E. Generalization evaluation
![generaliztion](images/generalization_evaluation.jpg)
### F. Robustness evaluation
![robustness](images/robustness.jpg)

## Acknowledge
Our work is based on the following works:
1. SkykeTorch  
    [[Paper](https://www.frontiersin.org/articles/10.3389/fnins.2019.00625/full) 
    | 
    [Project](https://github.com/miladmozafari/SpykeTorch)]
2. R-STDP  
    [[Paper](https://ieeexplore.ieee.org/document/8460482/)
    | 
    [Project](https://github.com/clamesc/Training-Neural-Networks-for-Event-Based-End-to-End-Robot-Control)]