from config import emotion_config as config
from utils.imagetoarraypreprocess import ImageToArrayPreprocess
from hdf5datasetgenerator import HDF5DatasetGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, help="path to model checkpoint to load")
args = vars(ap.parse_args())

testGen = ImageDataGenerator(
  rescale = 1.0/255
)

iap = ImageToArrayPreprocess()

testAugGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE, aug=testGen, preprocessors=[iap], classes=config.NUM_CLASSES)

model = load_model(args['model'])

loss, acc = model.evaluate_generator(
  testAugGen.generator(),
  steps = testAugGen.numImages // config.BATCH_SIZE
)

print('[Accuracy] {:.2f}'.format(acc*100))

testAugGen.close()