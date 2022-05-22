import os
import cv2
import json
import random
import numpy as np
import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt

from pycocotools.coco import COCO

# detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators

# Ignore warnings
import warnings
warnings.filterwarnings('ignore') 

# Setup logger
setup_logger()


# ## Register dataset

# In[2]:


Data_Resister_training="train_xworld";
Data_Resister_valid="val_xworld";
from detectron2.data.datasets import register_coco_instances

register_coco_instances(Data_Resister_training,{}, 'data/instances_train_xworld.json', Path("data/train_xworld"))
register_coco_instances(Data_Resister_valid,{},'data/instances_val_xworld.json', Path("data/val_xworld"))

metadata = MetadataCatalog.get(Data_Resister_training)
dataset_train = DatasetCatalog.get(Data_Resister_training)
dataset_valid = DatasetCatalog.get(Data_Resister_valid)


# In[57]:


# print(dataset_train[690])
HEIGHT = 600
WIDTH = 600


# In[88]:


import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from pycocotools.coco import maskUtils as MU
import numpy as np
from PIL import Image

torch.backends.cudnn.benchmark = True

def resize_box(bbox, orig_size, target_size):
    # get sizes
    hh, ww = target_size
    h, w = orig_size

    # get coords
    x1, y1, x2, y2 = bbox

    # normalize
    x1, x2 = float(x1/w), float(x2/w)
    y1, y2 = float(y1/h), float(y2/h)

    # rescale
    x1, x2 = ww*x1, ww*x2
    y1, y2 = hh*y1, hh*y2

    return [x1, y1, x2, y2]

def resize_seg(segs, orig_size, target_size):
    scale_y = float(target_size[0] / orig_size[0])
    scale_x = float(target_size[1] / orig_size[1])

    res = segs

    for seg in res:
        for i in range(0,len(seg)-1, 2):
            sx, sy = seg[i:i+2]
            sx, sy = scale_x*sx, scale_y*sy
            seg[i:i+2] = [sx, sy]

    return res

class AVADataset(Dataset):
    def __init__(self, ds, height=1080, width=1920, transform=None):
        super(AVADataset, self).__init__()
        self.labels = []
        self.bboxes = []
        self.segmentations = []
        self.paths = []
        self.crowd = []
        self.transform = transform
        self.ids = []

        self.height = height
        self.width = width

        scaled_size = (self.height, self.width)
        
        for sample in ds:
            for s in sample["annotations"]:
                if 'keypoints' in s.keys():
                    s.pop("keypoints", None)
        
        for data in ds:
            s = []
            b = []
            l = []
            c = []
            flag = 0
            
            assert data['height'] == 1080 and data['width'] == 1920, "incosistent height and width!"

            orig_size = (data['height'], data['width'])
            
            for item in data['annotations']:
                if item['bbox'][2] <= 0 or item['bbox'][3] <= 0:
                    flag = 1
                    break
                    
                if 'segmentation' in item.keys():
                    segs = item['segmentation']
                    rescaled = resize_seg(segs, orig_size, scaled_size)
#                     for seg in segs:
#                         for se in seg:
#                             se = int(se * scale)
                    s.append(rescaled)
                if 'bbox' in item.keys():
                    box = item['bbox'].copy()
                    box[2] = box[2] + box[0]
                    box[3] = box[3] + box[1]
                    box = box[:4]
                    assert box[2] > box[0], "x coord error! {}"
                    assert box[3] > box[1], "y coord error! {}"
                    assert len(box) == 4, "bbox wrong length! {}".format(item)
                    rescaled = resize_box(box, orig_size, scaled_size)
                    b.append(rescaled)
                if 'category_id' in item.keys():
                    l.append(item['category_id'])
                if 'iscrowd' in item.keys():
                    c.append(item['iscrowd'])
            if (flag == 1):
                flag = 0
                continue
            self.labels.append(l)
            self.bboxes.append(b)
            self.segmentations.append(s)
            self.crowd.append(c)
            self.paths.append(data['file_name'])
            self.ids.append(data['image_id'])

        assert len(self.labels) == len(self.bboxes) == len(self.segmentations) == len(self.paths), "lengths must be same"
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        h, w = self.height, self.width
                
        # img = cv2.imread(self.paths[idx])[:,:,::-1].reshape(3, h, w)
        img = cv2.imread(self.paths[idx])[:,:,::-1]
        img = cv2.resize(img, (self.height, self.width)).reshape(3, h, w)
            
        label = self.labels[idx]
        image_id = self.ids[idx]
        bbox = self.bboxes[idx]
        rles = [MU.merge(MU.frPyObjects(seg, h, w))
                   for seg in self.segmentations[idx]]
        segm = [MU.decode(rle)
                   for rle in rles]
        area = [MU.area(rle) for rle in rles]
        iscrowd = self.crowd[idx]
        
        target = {}
        
        target["boxes"] = torch.as_tensor(bbox, dtype=torch.float32)
        target["labels"] = torch.as_tensor(label, dtype=torch.int64)
        target["masks"] = torch.as_tensor(segm, dtype=torch.uint8)
        target["image_id"] = torch.tensor([image_id])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)
        
        return torch.as_tensor(img.copy(), dtype=torch.float32), target


# In[4]:


from utils import collate_fn, MetricLogger
import utils
from engine import train_one_epoch, evaluate, train_step


# In[89]:


# create datasets
train_ds = AVADataset(dataset_train, HEIGHT, WIDTH)
valid_ds = AVADataset(dataset_valid, HEIGHT, WIDTH)


# In[90]:


print("length of training set:", len(train_ds))
print("length of validation set:", len(valid_ds))


# In[91]:


# create dataloaders
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=6, shuffle=True, 
                                           num_workers=12, pin_memory=True,
                                          collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(valid_ds, batch_size=1, 
                                         shuffle=False, num_workers=12, 
                                         pin_memory=True, collate_fn=collate_fn)


# In[11]:


sample = next(iter(train_loader))


# In[19]:


first = sample[0][0]
first_feats = sample[1][0]


# In[64]:


#print(first_feats)


# In[33]:


copy = first.numpy().astype('uint8').reshape((HEIGHT,WIDTH,-1))


# In[34]:


for box in first_feats['boxes'].numpy().tolist():
    start_point = box[:2]
    start_point = [int(p) for p in start_point]
    end_point = box[2:]
    end_point = [int(p) for p in end_point]
    cv2.rectangle(copy, start_point, end_point, (0,255,0), 4)

# for mask in first_feats['masks'].numpy():
#     plt.figure()
#     plt.imshow(mask.reshape(HEIGHT, WIDTH))

# In[35]:


# import matplotlib.pyplot as plt
# plt.imshow(copy)
# plt.show()

# In[7]:


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


# In[94]:


model = get_model_instance_segmentation(8)


# In[95]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# In[ ]:


model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.00001,
                            momentum=0.9)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# let's train it for 10 epochs
num_epochs = 10

print_freq = 10
eval_freq = 5000

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    #train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(train_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for step, (images, targets) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):

        metrics = train_step(images, targets, model, optimizer, lr_scheduler, device)

        metric_logger.update(loss=metrics['losses'], **metrics['loss_dict'])
        metric_logger.update(lr=metrics["lr"])

        if ((step+1) % eval_freq == 0):
            evaluate(model, val_loader, device=device)
            model.train(True)


    # update the learning rate
    #lr_scheduler.step()
    # evaluate on the test dataset
    #evaluate(model, val_loader, device=device)
