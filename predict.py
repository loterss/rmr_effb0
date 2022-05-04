import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.image as mpimg
import torch
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from model import FineTuneModel


def getPredictions(image_path):
    image = mpimg.imread(image_path)
    transform = T.Compose([
        T.ToPILImage(mode='RGB'),
        T.Resize((480, 640)),
        T.CenterCrop(448),
        T.Resize(224),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = torch.unsqueeze(transform(image),0)

    original_model = torchvision.models.efficientnet_b0(pretrained=False)
    model = FineTuneModel(original_model, "efficientnet_b0", 6)
    model.load_state_dict(torch.load('final_model.pt', map_location=torch.device('cpu')))
    model.eval()

    propability_output = F.softmax(model(input_tensor), dim=1)
    propability_output = propability_output.detach().numpy()[0]*100

    plt.rcParams.update({'font.size': 16})
    plt.rcParams["figure.figsize"] = (15, 10)
    label = ['RMR\n 20-30', 'RMR\n 30-40', 'RMR\n 40-50', 'RMR\n 50-60', 'RMR\n 60-70', 'RMR\n 70-80']
    plt.bar(label, propability_output)
    plt.ylabel('Probability (%)')

    class_type = np.argmax(propability_output)

    return propability_output, class_type


if __name__ == '__main__':
    getPredictions('E:\Works\Civil Proj\S2-IMG\S2-0002.jpg')