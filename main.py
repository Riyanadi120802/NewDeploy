import torch
from inference import inference_process
from Postprocessing import visualization
from preprocessing import process_image
from init_model import Model

# Input image file in jpg, jpeg, or png
input_image = input('Enter image file name: ')

# Initialization
# device = torch.device('cpu')

PATH = 'D:\File Kuliah\MBKM\Deploy_Model\ML\models\model_Mobilenetv2Neww.pt'
model = Model()
model.load_state_dict(torch.load(PATH))
model.eval()

# Preprocessing
tensor_image, image = process_image(input_image)

# Prediction
label, confidence = inference_process(tensor_image, model)

# Post-processing
visualization(image, label, confidence)
