import random
import cv2 as cv
import numpy as np

# python test.py -i "images/image.png" -t "images/trimap.png"
if __name__ == '__main__':
    img_rows, img_cols = 320, 320
    channel = 4

from capsule_layers import ConvCapsuleLayer, DeconvCapsuleLayer, Mask, Length
from keras import layers, models
from keras import backend as K
from custom_losses import dice_hard, weighted_binary_crossentropy_loss, dice_loss, margin_loss
from tqdm import tqdm
import SimpleITK as sitk
from os.path import join, basename
from keras.optimizers import Adam,SGD,Nadam
import tensorflow as tf
from keras.layers import ZeroPadding2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
from utils import overall_loss, get_available_cpus, get_available_gpus, alpha_prediction_loss,compute_mse_loss
from data_generator import train_gen,valid_gen,generate_trimap,random_choice,safe_crop
from config import patience, batch_size, epochs, num_train_samples, num_valid_samples , img_cols,img_rows
import keras as keras
from Build_model import capsnet,build_refinement

if __name__ == '__main__':
    i=2
    final = capsnet()
    #final = build_refinement(capsnet)
    final.load_weights("checkpointt/weights-improvement-32-0.74-0.146.hdf5")
    print(final.summary())
     
    import cv2 as cv
    import numpy as np
    x_test = np.empty((1, img_rows, img_cols, 4), dtype=np.float32)
    bgr_img = cv.imread('test/a/GT13.png')
    alpha = cv.imread('test/at/GT13.png', 0)
    
    
    trimap= generate_trimap(alpha)
    different_sizes = [(320, 320), (320, 320), (480, 480)]
    crop_size = random.choice(different_sizes)
    x, y = random_choice(trimap, crop_size)

    
    bgr_img = safe_crop(bgr_img, x, y, crop_size)
    alpha = safe_crop(alpha, x, y, crop_size)
    trimap = safe_crop(trimap, x, y, crop_size)
    cv.imwrite('test/{}_image.png'.format(i), np.array(bgr_img).astype(np.uint8))
    cv.imwrite('test/{}_trimap.png'.format(i), np.array(trimap).astype(np.uint8))
    cv.imwrite('test/{}_alpha.png'.format(i), np.array(alpha).astype(np.uint8))
    
    
    x_test = np.empty((1, 320, 320, 4), dtype=np.float32)
    x_test[0, :, :, 0:3] = bgr_img / 255.
    x_test[0, :, :, 3] = trimap / 255.
    
    y_true = np.empty((1, img_rows, img_cols, 2), dtype=np.float32)
    y_true[0, :, :, 0] = alpha / 255.
    y_true[0, :, :, 1] = trimap / 255.
    
    
    out = final.predict(x_test)
    out = np.reshape(out, (img_rows, img_cols))
    
    mse_loss = compute_mse_loss(out, alpha, trimap)
    print(mse_loss)
    str_msg = ' mse_loss: %.4f' % (mse_loss)
    
    print(out.shape)
    out = out * 255.0
    out = out.astype(np.uint8)
#    draw_str(out, (10, 20), str_msg)

    cv.imshow('out', out)
    cv.imwrite('test/GT13.png', out)
    cv.waitKey(0)
    cv.destroyAllWindows()
