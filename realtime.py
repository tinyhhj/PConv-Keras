import cv2
import numpy as np

from Sketcher import Sketcher
from libs.util import MaskGenerator, ImageChunker
from libs.pconv_model import PConvUnet

import sys
from copy import deepcopy

print('load model...')
model = PConvUnet(vgg_weights=None, inference_only=True)
model.load('data/logs/MyDataset_phase1/weights.75-0.51.h5', train_bn=False)
# model.summary()
import os
src = 'D:\git\crawler\zigbang\\test\class\\'
dest = 'images\\'
def random_pick(path):
  return os.listdir(path)[np.random.randint(0,len(os.listdir(path)))]
img = cv2.imread(os.path.join(src,random_pick(src)), cv2.IMREAD_COLOR)

img_masked = img.copy()
mask = np.zeros(img.shape[:2], np.uint8)

sketcher = Sketcher('image', [img_masked, mask], lambda : ((255, 255, 255), 255))
chunker = ImageChunker(512, 512, 30)

while True:
  key = cv2.waitKey()
  if key == ord('s'): #save
    sketcher.save_files(dest)
  if key == ord('c'):
    new_img = cv2.imread(os.path.join(src, random_pick(src)), cv2.IMREAD_COLOR)
    new_mask = np.zeros(new_img.shape[:2],np.uint8)
    sketcher.dests[0] = new_img
    sketcher.dests[1] = new_mask
    sketcher.show()
  if key == ord('q'): # quit
    break
  if key == ord('r'): # reset
    print('reset')
    img_masked[:] = img
    mask[:] = 0
    sketcher.show()
  if key == 32: # hit spacebar to run inpainting
    input_img = img_masked.copy()
    input_img = input_img.astype(np.float32) / 255.

    input_mask = cv2.bitwise_not(mask)
    input_mask = input_mask.astype(np.float32) / 255.
    input_mask = np.repeat(np.expand_dims(input_mask, axis=-1), repeats=3, axis=-1)

    # cv2.imshow('input_img', input_img)
    # cv2.imshow('input_mask', input_mask)

    print('processing...')

    chunked_imgs = chunker.dimension_preprocess(deepcopy(input_img))
    chunked_masks = chunker.dimension_preprocess(deepcopy(input_mask))

    # for i, im in enumerate(chunked_imgs):
    #   cv2.imshow('im %s' % i, im)
    #   cv2.imshow('mk %s' % i, chunked_masks[i])

    pred_imgs = model.predict([chunked_imgs, chunked_masks])
    result_img = chunker.dimension_postprocess(pred_imgs, input_img)

    print('completed!')

    cv2.imshow('result', result_img)

cv2.destroyAllWindows()
