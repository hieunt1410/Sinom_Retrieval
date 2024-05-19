import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import os
import cv2
from src.utils import *
import src.models as models
import configs.config as cfg

from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

def load_img_tensor(image_path):
    image_tensor = transform(Image.open(image_path).convert('RGB'))
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def compute_similar_images(image_path, num_images, embedding, encoder, device):
    image_tensor = load_img_tensor(image_path)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()
        
    flatten_embedding = embedding.reshape((embedding.shape[0], -1))
    
    knn = NearestNeighbors(n_neighbors=num_images, metric='cosine')
    knn.fit(embedding)
    
    _, indices = knn.kneighbors(flatten_embedding)
    indices_list = indices.tolist()
    
    return indices_list


def compute_similar_features(image_path, num_images, embedding, device, nfeatures=20):
    image = cv2.imread(image_path)
    orb = cv2.ORB_create(nfeatures=nfeatures)

    # Detect features
    keypoint_features = orb.detect(image)
    # compute the descriptors with ORB
    keypoint_features, des = orb.compute(image, keypoint_features)

    des = des / 255.0
    des = np.expand_dims(des, axis=0)
    des = np.reshape(des, (des.shape[0], -1))

    pca = PCA(n_components=des.shape[-1])
    reduced_embedding = pca.fit_transform(
        embedding,
    )
    
    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(reduced_embedding)
    _, indices = knn.kneighbors(des)

    indices_list = indices.tolist()

    return indices_list


def plot_similar_images(test_img_path, indices_list):
    
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')
    
    test_img_name = test_img_path.split('/')[-1].split('.')[0]
    
    if not os.path.exists(f'./outputs/{test_img_name}'):
        os.mkdir(f'./outputs/{test_img_name}')
    
    indices = indices_list[0]
    
    for index in indices:
        img_name = str(index) + '.png'
        img_path = os.path.join(cfg.DATA_PATH, 'database_2D', img_name)
        img = Image.open(img_path).convert('RGB')
        plt.imshow(img)
        plt.show()
        img.save(os.path.join(f'./outputs/{test_img_name}', img_name))
        
                
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = models.ConvEncoder()
    load_model(encoder, 'encoder', device)
    encoder.to(device)
    encoder.eval()
    
    embedding = np.load(cfg.EMBEDDING_PATH)
    
    indices_list = compute_similar_images(cfg.TEST_IMAGE_PATH, cfg.NUM_IMAGES, embedding, encoder, device)
    plot_similar_images(cfg.TEST_IMAGE_PATH, indices_list)
    
    # indices_list = compute_similar_features(cfg.TEST_IMAGE_PATH, cfg.NUM_IMAGES, embedding, device, nfeatures=30)
    # plot_similar_images(cfg.TEST_IMAGE_PATH, indices_list)      