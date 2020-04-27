

import cv2
import numpy as np

from PIL import Image

from matplotlib import pyplot as plt
import torchvision
from facenet_pytorch import MTCNN
from tqdm import tqdm
device = 'cpu'

mtcnn = MTCNN(image_size=256, margin=80, min_face_size=20, keep_all=True, factor=0.5, post_process=False, device=device).eval()

img = Image.open('D:/x25.jpg')
img_cropped = mtcnn(img)
# import pdb
# pdb.set_trace()
img_cropped /= 255
img_cropped = torchvision.transforms.ToPILImage()(img_cropped.squeeze(0))
img_cropped.show()