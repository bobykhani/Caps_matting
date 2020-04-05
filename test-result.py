import os

import cv2 as cv
import numpy as np


from capsule_layers import ConvCapsuleLayer, DeconvCapsuleLayer, Mask, Length
from keras import layers, models
from keras import backend as K
#from custom_losses import dice_hard, weighted_binary_crossentropy_loss, dice_loss, margin_loss
from tqdm import tqdm
import SimpleITK as sitk
from os.path import join, basename
from keras.optimizers import Adam,SGD,Nadam
import tensorflow as tf
from keras.layers import ZeroPadding2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
from utils import overall_loss, get_available_cpus, get_available_gpus, alpha_prediction_loss , get_final_output
from data_generator import train_gen,valid_gen
from config import patience, batch_size, epochs, num_train_samples, num_valid_samples , img_cols,img_rows
import keras as keras
from Build_model import capsnet,build_refinement

final = capsnet()
#final = build_refinement(final)
final.load_weights("checkpointt/weights-improvement-32-0.74-0.146.hdf5")
#final.load_weights("checkpointt/weights-improvement-R-02-0.73-0.002.hdf5")

print(final.summary())

images = [f for f in os.listdir('New folder (3)\merged') if f.endswith('.png')]

for image_name in images:
    filename = os.path.join('New folder (3)\merged', image_name)
    im = cv.imread(filename)
    im_h, im_w = im.shape[:2]
    
    cropx = np.empty((320, 320, 3), dtype=np.float32)
    tricropx = np.empty((320, 320), dtype=np.float32)

    from data_generator import generate_trimap
    
    trimap_name = os.path.join('New folder (3)\mask', image_name)
    trimap = cv.imread(trimap_name, 0)
    trimap = generate_trimap(trimap)
			
    finalout = np.zeros((((int(np.ceil(im_h/320))+1)*320),((int(np.ceil(im_w/320))+1)*320)),dtype=np.float32)
    #finalout = np.zeros(im_h,im_w),dtype=np.float32)
    print (trimap_name)
    a=0
    b=0
    cut=320
    
    for i in range(0, (int(np.ceil(im_h/cut)))):
        for j in range(0,int(np.ceil(im_w/cut))):
            print('hi')
            x = j * cut
            y = i * cut
            w = min(cut, im_w - x)
            h = min(cut, im_h - y)
            crop = im[y:y+h,x:x+w]
            tri_crop = trimap[y:y+h,x:x+w]

          #  crop = cv.resize(crop,(320,320))
          #  tri_crop = cv.resize(tri_crop,(320,320))
            print(tri_crop.shape)
            if(crop.shape[0] == 320, crop.shape[1] == 320, tri_crop.shape[0]==320, tri_crop.shape[1]==320):
                
                yyy=(i*cut)
                ttt=((i+1)*cut)
                yy=(j*cut)
                tt=((j+1)*cut)

                
                
                x_test = np.zeros((1, 320, 320, 4), dtype=np.float32)
                print("crop",crop.shape)
                x_test[0, 0:tri_crop.shape[0],0:tri_crop.shape[1], 0:3] = crop[:,:,0:3] / 255.0
                x_test[0, 0:tri_crop.shape[0],0:tri_crop.shape[1], 3] = tri_crop[:,:] / 255.0
                
                #address1 = "alpha/{}{}.png".format(i,j)
                #cv.imwrite(address1, tri_crop[:,:])
                
                out=final.predict(x_test)
                y_pred = np.reshape(out, (img_rows, img_cols))
                print(y_pred.shape)
                y_pred = y_pred * 255.0
                y_pred = get_final_output(y_pred,x_test[0,:,:,3]*255)
                y_pred = y_pred.astype(np.uint8)
                
                out = y_pred.copy()
                a=i
                b=j
#                    for e in range(0,im_h):
#                        for d in range(0,im_w):
#                            if(e>0 and e%320==0):
                
              #  out = cv.resize(out,(cut,cut))
                finalout[yyy:ttt,yy:tt]=out#np.max((finalout[yyy:ttt,yy:tt],out))
                print(i,a,finalout.shape,out.shape)
                
    for i in range(0, (int(np.ceil(im_h/cut)))-1):
        for j in range(0,int(np.ceil(im_w/cut))-1):
            print('hi')
            x = j * cut + 160
            y = i * cut + 160
            w = min(cut, im_w - x)
            h = min(cut, im_h - y)
            crop = im[y:y+h,x:x+w]
            tri_crop = trimap[y:y+h,x:x+w]

          #  crop = cv.resize(crop,(320,320))
          #  tri_crop = cv.resize(tri_crop,(320,320))

            if(crop.shape[0]==320,crop.shape[1]==320):
                
                yyy=(i*cut) + 160
                ttt=((i+1)*cut) + 160
                yy=(j*cut) + 160
                tt=((j+1)*cut) + 160
                
                print(x,y,w,h,yyy,ttt,yy,tt)

                
                #print("crop",crop.shape)
                x_test[0, 0:tri_crop.shape[0],0:tri_crop.shape[1], 0:3] = crop[:,:,0:3] / 255.0
                x_test[0, 0:tri_crop.shape[0],0:tri_crop.shape[1], 3] = tri_crop[:,:] / 255.0
                
                out=final.predict(x_test)
                y_pred = np.reshape(out, (img_rows, img_cols))
                print(y_pred.shape)
                y_pred = y_pred * 255.0
                y_pred = get_final_output(y_pred,x_test[0,:,:,3]*255)
                y_pred = y_pred.astype(np.uint8)
 
                address1 = "New folder (3)/bobak/{}{}.png".format(i,j)
                cv.imwrite(address1, y_pred[:,:])

                out = y_pred.copy()
                a=i
                b=j
#                    for e in range(0,im_h):
#                        for d in range(0,im_w):
#                            if(e>0 and e%320==0):
                
              #  out = cv.resize(out,(cut,cut))
                finalout[yyy:ttt,yy:tt]=(out+finalout[yyy:ttt,yy:tt])/2#np.max((finalout[yyy:ttt,yy:tt],out))
                print(i,a,finalout.shape,out.shape)
                
#                    for e in range(0,im_h):
#                        for d in range(0,im_w):
#                            if(e>0 and e%320==0):
#                                finalout[e,d]=(finalout[e-1,d+1]+finalout[e,d+1]+finalout[e+1,d+1]+finalout[e,d-1]+finalout[e,d]+finalout[e,d+1]+finalout[e-1,d-1]+finalout[e,d-1]+finalout[e+1,d-1])/9
#                            if(d>0 and d%320==0):
#                                finalout[e,d]=(finalout[e-1,d+1]+finalout[e,d+1]+finalout[e+1,d+1]+finalout[e,d-1]+finalout[e,d]+finalout[e,d+1]+finalout[e-1,d-1]+finalout[e,d-1]+finalout[e+1,d-1])/9
                
        alpha_out = np.zeros((im_h,im_w, 1), dtype=np.float32)
        alpha_out = finalout[0:im_h,0:im_w]
        
#        for i in range(0,im_h):
#            for j in range(0,im_w):
#                if (finalout[i,j]>180):
#                    finalout[i,j]=255
#                if (finalout[i,j]<120):
#                    finalout[i,j]=0
        
        address = "New folder (3)/bobak/{}".format(image_name)
        cv.imwrite(address, alpha_out)


			
			


