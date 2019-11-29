import numpy as np
import os
import re



def get_ID(img_names, dataset = None, id_info_file = None):
    IDs = []
    org = []
    if(dataset == "test_data_2"):
        id_info = pd.read_csv(id_info_file)
        id_col = id_info.iloc[:, 0].values
        name_col = id_info.iloc[:, 7].values
    
    # ---- Find the IDs of the datasets
    for name in img_names:
        if(dataset == "Vera"):
            org.append(0)
            IDs.append(int(name.split("_")[0]))
        elif(dataset == "Bosphorus"):
            org.append(1)
            temp = name.split("_")[0].split()[0]
            IDs.append(np.array(re.findall(r'\d+', temp)).astype(int)[0])
        elif(dataset == "test_data_1"):
            org.append(2)
            temp = name.split("_")[0].split()[0]
            IDs.append(int(temp))
        elif(dataset == "test_data_2"):
            org.append(3)
            if(name in name_col):
                IDs.append(int(id_col[list(name_col).index(name)]))
            else:
                print(name + " is not found in info file..")
                break
        
    return IDs, org


def Train_Validation_Split(X_train_names, y_train, ID, split_factor = float(2/3)):
    X_names = []
    X_val_names = []
    y = []
    y_val = []

    # Splitting IDs for Train-Validation Splitting
    ID_a = np.array(list(dict.fromkeys(ID)))
    randomize = np.arange(len(ID_a))
    np.random.shuffle(randomize)
    ID_a = ID_a[randomize]
    slices = int(len(ID_a) * split_factor)
    ID_train = list(ID_a[:slices])
    ID_val = list(ID_a[slices:])

    for sample in range(0, len(X_train_names)):
        if(ID[sample] in ID_train):
            X_names.append(X_train_names[sample])
            y.append(y_train[sample])
        elif(ID[sample] in ID_val):
            X_val_names.append(X_train_names[sample])
            y_val.append(y_train[sample])
        
    return np.array(X_names), np.array(X_val_names), np.array(y), np.array(y_val)



pathDirData = "./"

## ----------------------------------------------------------- Bosphorus
## Import Main Data
#data = np.load(pathDirData + 'Train_data_without_augmentation.npz') 
#X_names = data['X_train_names'].astype(str)
#y = data['y_train']
#
## Import Augmented Data
#data = np.load(pathDirData + "Augmented_Train_data.npz") 
#X_train_aug_names = data['X_train_aug_names'].astype(str)
#y_train_aug = data['y_train_aug'].reshape((-1, 4))
#
## Concatenate main data and augmented data
#X_names = np.concatenate((X_names, X_train_aug_names), axis = 0)
#y = np.concatenate((y, y_train_aug), axis = 0)
#
#ID, org = get_ID(X_names, "Bosphorus")
#split_factor = float(2/3)
#X_names, X_val_names, y, y_val = Train_Validation_Split(X_names, y, ID, split_factor)
#ID, org = get_ID(X_names, "Bosphorus")
#ID_val, org_val = get_ID(X_val_names, "Bosphorus")
#
#np.savez("Bos_train_val_data.npz", 
#        names = X_names,
#        y = y,
#        names_val = X_val_names,
#        y_val = y_val)
#
## ----------------------------------------------------------- Vera
## Import Main Data
#data = np.load(pathDirData + 'Train.npz') 
#X_names_v = data['names'].astype(str)
#y_v = np.array(data['manual_points']).reshape((-1, 4))
#
## Import Augmented Data
#data = np.load(pathDirData + "Augmented_Train_data_vera.npz") 
#X_train_aug_names = data['X_train_aug_names'].astype(str)
#y_train_aug = data['y_train_aug'].reshape((-1, 4))
#
## Concatenate main data and augmented data
#X_names_v = np.concatenate((X_names_v, X_train_aug_names), axis = 0)
#y_v = np.concatenate((y_v, y_train_aug), axis = 0)
#
## Train-val splitting
#ID_v, org_v = get_ID(X_names_v, "Vera")
#split_factor_v = float(2/3)
#X_names_v, X_val_names_v, y_v, y_val_v = Train_Validation_Split(X_names_v, y_v, ID_v, split_factor_v)
#ID_v, org_v = get_ID(X_names_v, "Vera")
#ID_val_v, org_val_v = get_ID(X_val_names_v, "Vera")
#
## ----------------------------------- Concatenate Bosphorus and Vera Data
#np.savez("Vera_train_val_data.npz", 
#        names = X_names_v,
#        y = y_v,
#        names_val = X_val_names_v,
#        y_val = y_val_v)


# ------------------------------------------------------------- Bosphorus
data = np.load(pathDirData + 'Test_Bosphorous.npz') 
X_names = data['X_test_names'].astype(str)
y = data['y_test']
ID, org = get_ID(X_names, "Bosphorus")
print('-' * 100)
print("Test on Bosphorous Dataset...")
print('-' * 100)

# ------------------------------------------------------------- Vera
data = np.load(pathDirData + 'Test_Vera.npz') 
X_names = data['names'].astype(str)
y = np.array(data['manual_points']).reshape((-1, 4))
ID, org = get_ID(X_names, "Vera")
print('-' * 100)
print("Test on Vera Dataset...")
print('-' * 100)

# ------------------------------------------------------------- Test Data - 1
data = np.load(pathDirData + 'test_data_1.npz') 
X_names = data['names'].astype(str)
y = np.array(data['manual_points']).reshape((-1, 4))
ID, org = get_ID(X_names, "test_data_1")
print('-' * 100)
print("Test on Dataset_1...")
print('-' * 100)

# ------------------------------------------------------------- Test Data - 2
data = np.load(pathDirData + 'test_data_2.npz') 
X_names = data['names'].astype(str)
y = np.array(data['manual_points']).reshape((-1, 4))
id_info_file = pathDirData + 'Test_data_2_ID_info.csv'
ID, org = get_ID(X_names, "test_data_2", id_info_file)
print('-' * 100)
print("Test on Dataset_2...")
print('-' * 100)

# Concatenate Bosphorus and Vera Data
X_names = np.concatenate((X_names, X_names_v), axis = 0)
y = np.concatenate((y, y_v), axis = 0)
ID = np.concatenate((ID, ID_v), axis = 0)
org = np.concatenate((org, org_v), axis = 0)

np.savez("Bos_test_data.npz", 
        names = X_names_v,
        y = y_v,
        names_val = X_val_names_v,
        y_val = y_val_v)













