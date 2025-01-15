import warnings
import pickle

import torch 
import numpy as np 
import pandas as pd 
from PIL import Image

import xgboost as xgb
import torch.nn.functional as F
import torchvision.transforms as transforms

import models
from timm.models import create_model, load_checkpoint

warnings.filterwarnings("ignore")


IMAGENET_MEAN = [0.2867, 0.2861, 0.2948]  # RGB
IMAGENET_STD = [0.2336, 0.2331, 0.2373]  # RGB

def normalize(img):
    img = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(img)
    return img
# preprocess the image, crop and resize, cropbox example: (x1,y1,x2,y2)
def preprocess_img_ori(cropbox,img,img_size):
    cropbox = cropbox.strip("()-")
    cropbox = [int(i) for i in cropbox.split(',')]
    img_pil = img.crop(cropbox)  
    eval_transform = transforms.Compose([transforms.Resize((img_size, img_size)),transforms.ToTensor()])
    img_pil = eval_transform(img_pil)
    img_pil = normalize(img_pil)
    return img_pil

# normalize and add noise, nosie is 0 in valid 
def normalize_and_add_noise(df, column, min_val, max_val):
    def apply_normalize_and_noise(value):
        if value is None or np.isnan(value):
            return np.nan
        else:
            normalized_value = (value - min_val) / (max_val - min_val)
        # ensure the value is between 0 and 1
        normalized_value = max(0, min(1, normalized_value))
        return normalized_value
    return df[column].apply(apply_normalize_and_noise)

# normalize the features
def TM_bias(df):
    df['CA125'] = normalize_and_add_noise(df, 'CA125', min_val=3.3, max_val=610)
    df['HE4'] = normalize_and_add_noise(df, 'HE4', min_val=15, max_val=343)
    df['CA19-9'] =normalize_and_add_noise(df, 'CA19-9', min_val=0.7, max_val=150)
    df['AFP'] = normalize_and_add_noise(df, 'AFP', min_val=0.2, max_val=6.6)
    df['CEA'] = normalize_and_add_noise(df, 'CEA', min_val=1.7, max_val=4.5)
    df['MaximumDiameter'] =normalize_and_add_noise(df, 'MaximumDiameter', min_val=1.5, max_val=16)
    return df

def phase1_pred(model, img):
    with torch.no_grad():
        x = img.cuda().unsqueeze(0)
        model = model.to('cuda')
        result, unc, feature = model(x)
        result = F.softmax(result, dim=1)
        p = result.tolist()[0][1]
        u= unc.item()
        return p,u
    
def phase2_feature(model, img):
    feature_names = ['img_feature_' + str(i) for i in range(256)]
    with torch.no_grad():
        x = img.cuda().unsqueeze(0)
        model = model.to('cuda')
        result, feature = model.result_with_features(x)
        feature = feature.tolist()
        feature = pd.Series(feature[0], index=feature_names).to_frame().T
        result = F.softmax(result, dim=1)
        feature['img_pred_result'] = result.tolist()[0][1]
        return feature

# load the model
phase1_ckpt = './checkpoint/checkpoint-phase1.pth.tar'
phase1 = create_model('van_tiny2', num_classes=2)
load_checkpoint(phase1, checkpoint_path=phase1_ckpt, use_ema=True)
phase2_ckpt = './checkpoint/checkpoint-phase2.pth.tar'
phase2 = create_model('van_tiny', num_classes=2)
load_checkpoint(phase2, checkpoint_path=phase2_ckpt, use_ema=True)
phase2_p = pickle.load(open("./checkpoint/phase2_predict.pkl", "rb"))
phase2_u = pickle.load(open("./checkpoint/phase2_uncertainty.pkl", "rb"))

# example
bounding_box = '(19 ,140 ,195 ,328)'
image_path = './data/test/1.jpg'
imagex = Image.open(image_path)
structured_data = {"MenopausalStatus": 0, "FamilyHistory": 0, "MedicalHistory": 0, 
                    "BloodFlowPresence": 1, "HormoneTherapyHistory": 0, "TendernessonPalpation": 0, 
                    "CA125": 273.0, "HE4": 87.2, "CA19-9": 1.51, 
                    "AFP": 1.82, "CEA": None, "MaximumDiameter": 5.3}

# phase1 predict
imagex = preprocess_img_ori(bounding_box,image,324)
p,u = phase1_pred(phase1,imagex)
# if phase1 predict is low-risk, end
if p < 0.65 and u< 0.62:
    print("Clearly low-risk lesions, recommend regular follow-up.")
    print("The probability of high-risk is: ", 0.01)
    print("The original confidence is: ", p)
    print("The uncertainty is: ", u)
# else, phase2 predict
else:
    # get the image features
    imagex = preprocess_img_ori(bounding_box,image,352)
    features = phase2_feature(phase2,imagex)
    # preprocess structure features
    df = pd.DataFrame([structured_data])
    normalized_dataset=TM_bias(df)
    # concatenate the image features and structure features
    features_concat=pd.concat([normalized_dataset,features],axis=1,ignore_index=False)
    # lasso
    features167 = pd.read_csv('./LASSO.csv')
    features_lasso = pd.DataFrame(features_concat,columns =features167['Unnamed: 0'])
    features_lasso= features_lasso.drop(labels=['Intercept'] ,axis=1)
    # predict 
    predprob = phase2_p.predict_proba(features_lasso)[:,1] 
    unc = phase2_u.predict(features_lasso)

    print("The probability of high-risk is: ", predprob)
    print("The uncertainty is: ", unc)
