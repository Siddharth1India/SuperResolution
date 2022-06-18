from pyexpat import model
import tensorflow as tf
import os.path as osp
import glob
import cv2
import numpy as np
import torch

test_img_folder = './images/*'

model = tf.saved_model.load("./RealESRGAN_1/")

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    print(img.shape)
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, axis=0)
    img = tf.dtypes.cast(img, tf.float32)
    
    with torch.no_grad():
        output = model(x = img)
    
    output = output['sum'].numpy()
    output = output[0, :, :, :]
    print(output.shape)
    output = np.transpose(output, (1,2,0))
    print(output.shape)
    cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./results/{:s}_rlt.png'.format(base), output)