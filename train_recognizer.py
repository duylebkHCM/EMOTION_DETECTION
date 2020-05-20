from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from config import emotion_config as config
from hdf5datasetgenerator import HDF5DatasetGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from emotionModel import EmotionDetectModel
from emotionModelTransfer import TranserLearningModel
from utils.imagetoarraypreprocess import ImageToArrayPreprocess
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from utils.trainmonitor import TrainingMonitor
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import os


class EarlyStopTraining(Callback):
  def on_epoch_end(self, epoch, logs = {}):
    if logs['accuracy'] >= 0.9:
      print('\nReach the desire accuracy so stop training')
      self.model.stop_training = True

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
args = vars(ap.parse_args())

model = EmotionDetectModel() #Appoach 1
model.summary()
# model = TranserLearningModel() #Approach 2

train_datagen = ImageDataGenerator(
    rescale = 1.0/255,
    rotation_range = 20,
    width_shift_range = 0.15,
    height_shift_range = 0.15,
    zoom_range = 0.1,
    shear_range = 0.2,
    horizontal_flip= True,
    fill_mode = 'nearest'
)

val_datagen = ImageDataGenerator(
    rescale = 1.0/255
)

iap = ImageToArrayPreprocess()
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=train_datagen, preprocessors=[iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, aug=val_datagen, preprocessors=[iap], classes=config.NUM_CLASSES)

EPOCHS = 100
INIT_LR = 1e-2
DECAY_RATE = 1.0
FACTOR = 0.1

lr_decay_1 = LearningRateScheduler(lambda epoch: INIT_LR*(1/(1 + DECAY_RATE*epoch)))
lr_decay_2 = LearningRateScheduler(lambda epoch: INIT_LR*FACTOR**(epoch/10))

figPath = os.path.sep.join([config.OUTPUT_PATH, "Duynet_emotion.png"])
jsonPath = os.path.sep.join([config.OUTPUT_PATH, "Duynet_emotion.json"])

monitor = TrainingMonitor(figPath, jsonPath=jsonPath, startAt=0)

checkpoint = ModelCheckpoint(
    save_best_only = True,
    monitor = 'val_loss',
    mode = 'min',
    filepath = args['checkpoints'],
    verbose = 1
)

stop_train = EarlyStopTraining()

callbacks = [monitor, checkpoint, stop_train]

adam = Adam(lr = INIT_LR, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)

model.compile(optimizer = adam, loss = tf.keras.losses.CategoricalCrossentropy(), metrics = ['accuracy'])

history = model.fit_generator(
    trainGen.generator(),
    epochs = EPOCHS,
    steps_per_epoch = trainGen.numImages // config.BATCH_SIZE,
    validation_data = valGen.generator(),
    validation_steps = valGen.numImages //  config.BATCH_SIZE,
    callbacks = callbacks,
    verbose = 1
)

trainGen.close()
valGen.close()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label = 'train_accuracy')
plt.plot(epochs, val_acc, 'b', label = 'val_accuracy')
plt.title('Train acc and Val acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.figure()


plt.plot(epochs, acc, 'r', label = 'train_loss')
plt.plot(epochs, acc, 'b', label = 'val_loss')
plt.title('Train loss and Val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.figure()


