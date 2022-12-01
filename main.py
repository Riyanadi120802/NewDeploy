import torch
from inference import inference_process
from Postprocessing import visualization
from preprocessing import process_image
from init_model import Model

# Input image file in jpg, jpeg, or png
input_image = input('Enter image file name: ')

# Initialization
# device = torch.device('cpu')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATH = 'models\mobilenetv2_model.pt'
model = Model()
model.load_state_dict(torch.load(PATH), maping_location=torch.device('cpu'))
model.eval()

# Preprocessing
tensor_image, image = process_image(input_image)

# Prediction
label, confidence = inference_process(tensor_image, model)

# Post-processing
visualization(image, label, confidence)
