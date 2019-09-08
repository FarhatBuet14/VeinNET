import numpy as np 
import os
import cv2
import glob
import torch
import utils,  model
from torchvision import transforms
from PIL import Image


def resizing_inputs(test_folder):
    resized_folder = test_folder + "Resized/"
    files = os.listdir(test_folder)

    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)
    else:
        test = resized_folder + '*'
        r = glob.glob(test)
        for i in r:
            os.remove(i)
    
    for file in files:
        if(len(file.split(".")) == 2): 
            image = cv2.imread(test_folder + file)
            image = cv2.resize(image, (300, 240))
        
            cv2.imwrite(resized_folder + file, image)
    print("Finished Resizing..")

if __name__ == "__main__":
    
    test_folder = "./Test Images/"
    file = "59_____1.7215716044108074_____3.3250184059143066_____16.050174967447916_____15.23857307434082.pth.tar"
    pathModel = "./Model_Output/Best Model/" + file
    
    resizing_inputs(test_folder)
    images = []
    images_arr = []
    names = []
    for file in os.listdir(test_folder + "Resized/"):
        names.append(file)
        images.append(Image.open(test_folder + "Resized/" + file).convert("RGB"))
        images_arr.append(np.asarray(Image.open(test_folder + "Resized/" + file).convert("RGB")))
    
    trans_pipeline = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    processor = utils.SeedlingDataset(images_arr, test_folder, trans_pipeline = trans_pipeline)

    test_model = model.load_model("resnet50", False, 3, 4)
        
    modelCheckpoint = torch.load(pathModel)
    test_model.load_state_dict(modelCheckpoint['state_dict'])


    for sample in range(0, len(images)):
        image = images[sample]
        input = processor.get_processed(image)
        input = processor.trans_pipeline(input)
        input = torch.reshape(input, (1, 3, 240, 300)).cuda()
        output = test_model(input).reshape((2, 2)).cpu().detach().numpy()
        img = np.asarray(image)
        color = [(255, 255, 255), (0, 0, 0)] # Left - White, # Right - Black
        count = 0  
        for point in output:
            point = np.array(point).astype(int)
            cv2.circle(img, (point[0], point[1]), 
                    5, color[count], -1)
            count += 1
        cv2.imwrite(test_folder + "Results/" + names[sample], img)
    
    
    print("Finished")
