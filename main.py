import torchvision
import torch
import pathlib
import os 
import numpy as np
import transforms

import torch.optim as optim

from engine import train_one_epoch, evaluate
import utils

import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO

class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


def main():
	PATH_TO_TEST_IMAGES_DIR = pathlib.Path('./old')
	
	TRAIN_IMAGE_PATHS = PATH_TO_TEST_IMAGES_DIR/"train"
	TRAIN_ANNOTATIONS_PATH  = PATH_TO_TEST_IMAGES_DIR/"annotations"/"train.json"

	EVAL_IMAGE_PATHS = PATH_TO_TEST_IMAGES_DIR/"val"
	EVAL_ANNOTATIONS_PATH = PATH_TO_TEST_IMAGES_DIR/"annotations"/"val.json"

#	TEST_IMAGE_PATHS = PATH_TO_TEST_IMAGES_DIR/"test"
#	TEST_ANNOTATIONS_PATH = PATH_TO_TEST_IMAGES_DIR/"annotations"/"test.json"

	trans =  transforms.ToTensor()

	train_data = myOwnDataset(TRAIN_IMAGE_PATHS, TRAIN_ANNOTATIONS_PATH, transforms =trans)
	eval_data = myOwnDataset(EVAL_IMAGE_PATHS, EVAL_ANNOTATIONS_PATH, transforms =trans)
#	test_data = myOwnDataset(TEST_IMAGE_PATHS, TEST_ANNOTATIONS_PATH)

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

	print(device)

	model.to(device)

	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4,collate_fn=utils.collate_fn)
	eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=1, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
	#test_loader = torch.utils.data.DataLoader(eval_data, batch_size=1, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
	
	NUM_EPOCHS = 10
	for epoch in range(NUM_EPOCHS):

		# train for one epoch, printing every 10 iterations
		train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=5)
		# evaluate on the test dataset
		evaluate(model, eval_loader, device=device)
		# checkpoint
		torch.save(model.state_dict(), "ckp_{}.pt".format(epoch))

	print('Finished Training')

		
if __name__ == '__main__':
	main()