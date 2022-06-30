# imports python libraries
import numpy as np
import torch
#import pyvista as pv
import matplotlib.pyplot as plt
#from pyvirtualdisplay import Display
import torch.nn.functional as F
import torch.optim as optim



# import code from git
import sys
sys.path.append('C:/Users/ido lev/Desktop/repositories/UnsupervisedSaliencyPointCloud/Dataset')
sys.path.append('C:/Users/ido lev/Desktop/repositories/UnsupervisedSaliencyPointCloud/Saliency')
sys.path.append('C:/Users/ido lev/Desktop/repositories/UnsupervisedSaliencyPointCloud/Models')
sys.path.append('C:/Users/ido lev/Desktop/repositories/UnsupervisedSaliencyPointCloud/Training')
from model import PointNetCls, feature_transform_regularizer
from DataSet import ShapeNetDataset
import ModelTrainer
import AttackingForSaliency


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print(idle)
num_classes = 16
feature_transform = True

classifier = PointNetCls(k=num_classes, feature_transform=feature_transform)
model ='C:/Users/ido lev/Desktop/repositories/UnsupervisedSaliencyPointCloud/cls/cls_model_9.pth'
if model != '':
    classifier.load_state_dict(torch.load(model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()


datasetPath = 'C:/Users/ido lev/Desktop/DataSet/unsupervised saliency/shapenetcore_partanno_segmentation_benchmark_v0'
batchSize = 32
npoints = 2500
workers = 2
dataset = ShapeNetDataset(
        root=datasetPath,
        classification=True,
        npoints = npoints)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchSize,
    shuffle=True,
    num_workers = workers)

points, target = next(iter(dataloader))

test_dataset = ShapeNetDataset(
        root=datasetPath,
        classification=True,
        split='test',
        npoints = npoints,
        data_augmentation=False)
testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers = workers)



Model = ModelTrainer.ModelPointNetTrainer(dataloader, testdataloader,classifier,optimizer,scheduler,batchSize)
num_epochs = 250
Model.train(num_epochs=num_epochs)

