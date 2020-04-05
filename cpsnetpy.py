from keras.optimizers import Adam,SGD,Nadam
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
from utils import acc,overall_loss, get_available_cpus, get_available_gpus, alpha_prediction_loss,compute_sad_loss,compute_mse_loss,compositional_loss
from data_generator import train_gen,valid_gen
from config import patience, batch_size, epochs, num_train_samples, num_valid_samples,max_queue_size
import keras as keras
from Build_model import capsnet,build_refinement
from keras import losses

final = capsnet()

final.load_weights("checkpointt/weights-improvement-32-0.74-0.146.hdf5")
print("Loaded model from disk")


#sgd = SGD(lr=1e-6, decay=1e-7, momentum=0.9, nesterov=True)
tensor_board = keras.callbacks.TensorBoard(log_dir='./logs',histogram_freq=0, write_graph=True, write_images=True)
#early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
#opt = Adam(lr=4.741474821482305e-06, beta_1=0.99, beta_2=0.999, decay=0.00001)#lr=10.0003#
opt = Adam(lr=1e-6, beta_1=0.99, beta_2=0.999, decay=1e-7)#lr=10.0003#

#decoder_target = tf.placeholder(dtype='float32', shape=(None, None, None, None))
final.compile(optimizer=opt, loss=overall_loss,metrics=[acc],options = run_opts)
final.summary()

# Save the checkpoint in the /output folder
filepath="checkpointt/weights-improvement-{epoch:02d}-{val_acc:.2f}-{val_loss:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


callbacks = [tensor_board, reduce_lr,checkpoint]
final.fit_generator(train_gen(),
                    steps_per_epoch=num_train_samples // batch_size,
                    validation_data=valid_gen(),
                    validation_steps=num_valid_samples // batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks,
                  #  initial_epoch=19,
                    max_queue_size=max_queue_size
                    #use_multiprocessing=True,
                    #workers=6#int(get_available_cpus() / 2)
                    )

