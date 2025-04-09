import torch 
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
%matplotlibinline
import json
import argparse

#To get argument fromm the CMD

def get_input_arg():
    parser = argparse.ArgumentParser(description='Image Classifier')

    parser.add_argument("image_dir", required=True, type=str, help="path to folder of images")
    parser.add_argument('--checkpoint',required=True, type = str, default='densenet121', help=" CNN model architecture")
    parser.add_argument('--top_k', action='store_true', type=int, default=3, help="Learning rate of the model")
    parser.add_argument('--category_name', action='store_true',type=str, default='cat_to_name.json', help="To save the directory")
    parser.add_argument('--gpu', action='store_true',type=str, default='cuda', help="Set to gpu mode")
    
    return parser.parse_args()

#To process the image for feeding the network

def process_image(path_image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(path_image)
    new_size = (256, 256)
    resized_image = image.resize(new_size)
    np_image = np.array(resized_image)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (np_image - mean) / std
    transposed_image = image.transpose()
    
    return transposed_image

#To print an image tensor
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


#To make prediction

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    images = process_image(image_path)
    image_tensor = torch.from_numpy(images).type(torch.FloatTensor)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    predicted = model.forward(image_tensor)
    ps = torch.exp(predicted)
    top_ps, top_idices = predicted.topk(topk)
    
    top_ps = top_ps.numpy().squeeze()
    top_indices = top_indices.numpy().squeeze()
    
    idx_to_class = {i: j for i, j in model.class_to_idx.item()}
    top_classes = [idx_to_class[idx] for idx in top_indices]
    
    return top_ps, top_classes

# Display category names

def display_cat(img_path, top_ps, top_class ,json_file):
    image_name = []

    for key, value in json.items():
        for name in top_class:
            if name == value:
                flower_name.append(value)
      
    print(image_name) 

#To print an image with its probalilities

def display_image(img_path, top_ps, top_class ,json_file):
    img_name = json[img_path.split('/')[1]].title()
    
    fig, axes = plt.subplot(1,2, figsize=(14, 8))
    axes[0].imshow(top_class)
    axes[0].set(title=img_name)
    axes[0].axes('off')
    
    axes[1].barh(top_class, ps, aligned='Center')
    axes[1].set(title='Prediction')
    axes[1].set_xticks('Probabilities')
    
    plt.tight_layout()
    plt.show()
    
    
#Make prediction

    
def main():
    in_args = get_input_arg()
    image_path = in_args.img_dir
    checkpoint = in_args.checkpoint
    top_k = in_args.top_k
    cat_name = in_args.category_name
    gpu_mode = in_args.gpu
    
    if top_k:
        top_ps, top_class = predict(image_dir, checkpoint, top_k)
        print(top_class)
    if cat_name:
        top_ps, top_class = predict(image_dir, checkpoint)
        display_cat(top_class,cat_name)
    top_ps, top_class =predict(image_dir, chekpoint)
    display_image(image_dir, top_ps, top_class, json_file)

if __name__ == "__main__":
    main()
