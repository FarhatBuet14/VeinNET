import numpy as np 
import os
import cv2
import glob
import torch
import utils,  models
from torchvision import transforms
from PIL import Image
import argparse


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

def random_test(opt):
    
    # --------- Prepare Data
    images = []
    images_arr = []
    names = []
    for file in os.listdir(opt.pathDirData + "Resized/"):
        names.append(file)
        images.append(Image.open(opt.pathDirData + "Resized/" + file).convert("RGB"))
        images_arr.append(np.asarray(Image.open(opt.pathDirData + "Resized/" + file).convert("RGB")))
    
    trans_pipeline = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    processor = utils.SeedlingDataset(images_arr, opt.pathDirData, trans_pipeline = trans_pipeline)

    input_arr = []
    for sample in range(0, len(images)):
        
        image = images[sample]
        input = processor.get_processed(image)
        input = processor.trans_pipeline(input)
        input = torch.reshape(input, (3, 240, 300)).cuda()
        input_arr.append(input)
    
    input = torch.stack(input_arr).cuda()

    # ---------- Load Model
    test_model = models.resnet50_DANN(opt)
    modelCheckpoint = torch.load(opt.checkpoint)
    print('-' * 100)
    print('-' * 100)
    test_model.load_state_dict(modelCheckpoint['state_dict'])
    
    output, domain_output = test_model(input, alpha = 0)
    output = output.reshape((len(output), 2, 2)).cpu().detach().numpy()
    domain_loss = sum(domain_output.cpu().detach().numpy())/len(domain_output)
    print("Domain loss = " + str(domain_loss))

    for sample in range(0, len(input)):
        
        img = np.asarray(images[sample])
        
        color = [(255, 255, 255), (0, 0, 0)] # Left - White, # Right - Black
        count = 0  
        for point in output[sample]:
            point = np.array(point).astype(int)
            cv2.circle(img, (point[0], point[1]), 
                    5, color[count], -1)
            count += 1
        
        cv2.imwrite(opt.pathDirData + "Results/" + names[sample], img)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training VeinNetTrainer...')
    parser.add_argument("--pathDirData", type=str, metavar = "", help="Train/Test/Validation Data Directory", default = "./Test Images/")
    parser.add_argument("--nnArchitecture", type=str, metavar = "", help="Name of Model Architechture", default = "resnet50")
    parser.add_argument("--nnIsTrained", action = "store_true", help="Use Trained Network or not", default = True)
    parser.add_argument("--nnInChanCount", type=int, metavar = "", help="Number of Input channel", default = 3)
    parser.add_argument("--nnClassCount", type=int, metavar = "", help="Number of predicted values", default = 4)
    parser.add_argument("--gpu", action = "store_true", help="Use GPU or not", default = True)
    parser.add_argument("--checkpoint", type=str, metavar = "", help="Checkpoint File", 
    default = './Model_Output/' + '73_____2.162804449172247_____17.516155242919922_____3.787345668247768_____8.933741569519043_____-0.35563125610351565_____-0.34951141476631165.pth.tar')
    
    opt = parser.parse_args()
    print(opt)

    resizing_inputs(opt.pathDirData)
    random_test(opt)
    
    
    print("Finished")
