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
 	
1. dataset_builder.py
	- get_ID
	- SeedlingDataset
    	- get_processed
	- dataset_builder

2. utils.py
	- Cal_loss 
		- mae
		- mse
		- calculate
	- Vein_loss_class
		- get_vein_img
		- cal_vein_loss

3. model.py
	- load_model
	- SimpleCNN
 	
4. trainer.py
	- VeinNetTrainer
		- epochTrain
		- epochVal
		- training
		- testing