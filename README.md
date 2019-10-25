# Robust Human Authentication using Dynamic ROI Extraction from Dorsal/Palm Hand Vein Images
This project is to detect the dorsal vein using an infrared illuminator and then enhancement of vein images is done by using different techniques for identifying the person. The proposed work is related to the implementation of Infrared vein detection and matching system for person identification.

## Watch the [Video][1] (Undergraduate Project)
[1]: https://youtu.be/0xPcVjBJbuc

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


## Code Description - 
 	
1. utils.py
	- get_ID
	- Train_Validation_Split
	- SeedlingDataset
    	- get_processed
	- dataset_builder
	- load_checkpoint

2. losses.py
	- Cal_loss 
		- mae
		- mse

3. Vein_loss_class
	- get_vein_img

4. model.py
	- load_model
	
5. DensenetModel.py
	- DenseNet121
	- DenseNet169
	- DenseNet201

6. User_defined_model.py
	- SimpleCNN
	- RakibNET
 	
5. trainer.py
	- VeinNetTrainer
		- epochTrain
		- epochVal
		- training

6. test.py
	- VeinVetTester
		- testing

7. Random_Image_Test.py
	- resizing_inputs
	- random_test
