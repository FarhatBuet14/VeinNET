# Human-Identification-through-Matching-Dorsal-Vein
This project is to detect the dorsal vein using an infrared illuminator and then enhancement of vein images is done by using different techniques for identifying the person. The proposed work is related to the implementation of Infrared vein detection and matching system for person identification.

## [Video Link](https://youtu.be/0xPcVjBJbuc)


## Description - 

![vein-01.jpg](https://github.com/FarhatBuet14/Human-Identification-through-Matching-Dorsal-Vein/blob/master/Images/vein-01.jpg)
![vein-02.jpg](https://github.com/FarhatBuet14/Human-Identification-through-Matching-Dorsal-Vein/blob/master/Images/vein-02.jpg)
![vein-03.jpg](https://github.com/FarhatBuet14/Human-Identification-through-Matching-Dorsal-Vein/blob/master/Images/vein-03.jpg)
![vein-04.jpg](https://github.com/FarhatBuet14/Human-Identification-through-Matching-Dorsal-Vein/blob/master/Images/vein-04.jpg)
![vein-05.jpg](https://github.com/FarhatBuet14/Human-Identification-through-Matching-Dorsal-Vein/blob/master/Images/vein-05.jpg)
![vein-06.jpg](https://github.com/FarhatBuet14/Human-Identification-through-Matching-Dorsal-Vein/blob/master/Images/vein-06.jpg)
![vein-07.jpg](https://github.com/FarhatBuet14/Human-Identification-through-Matching-Dorsal-Vein/blob/master/Images/vein-07.jpg)
![vein-08.jpg](https://github.com/FarhatBuet14/Human-Identification-through-Matching-Dorsal-Vein/blob/master/Images/vein-08.jpg)
![vein-09.jpg](https://github.com/FarhatBuet14/Human-Identification-through-Matching-Dorsal-Vein/blob/master/Images/vein-09.jpg)
![vein-10.jpg](https://github.com/FarhatBuet14/Human-Identification-through-Matching-Dorsal-Vein/blob/master/Images/vein-10.jpg)
![vein-11.jpg](https://github.com/FarhatBuet14/Human-Identification-through-Matching-Dorsal-Vein/blob/master/Images/vein-11.jpg)


## Code Description - 
 	
### dataset_builder.py
--------------------------------------- 1) get_ID
 										2) SeedlingDataset
 										   	- get_processed
 										3) dataset_builder
 	
### utils.py
--------------------------------------- 1) Cal_loss
											- mae
											- mse
											- calculate
 										2) Vein_loss_class
 										   	- get_vein_img
 										   	- cal_vein_loss

### model.py
--------------------------------------- 1) load_model
 										2) SimpleCNN
 	
### trainer.py
--------------------------------------- 1) VeinNetTrainer
 										   	- epochTrain
 										   	- epochVal
 										   	- training
 										   	- test