from config import emotion_config as config
from hdf5datasetwriter import HDF5DatasetWriter
import numpy as np

print('[INFO] loading input data....')

f = open(config.INPUT_PATH)
f.__next__()

(trainImages, trainLabels) = ([], [])
(valImages, valLabels) = ([], [])
(testImages, testLabels) = ([], [])

for row in f:
    (label, pixel, usage) = row.strip().split(',')
    label = int(label)
    if config.NUM_CLASSES == 6:
      if label == 1:
        label == 0
      if label > 0:
        label -= 1
        
    pixel = np.array(pixel.split(" "), dtype = 'uint8')
    pixel = pixel.reshape((48, 48))
    
    if usage == 'Training':
        trainImages.append(pixel)
        trainLabels.append(label)
    elif usage == 'PublicTest':
        testImages.append(pixel)
        testLabels.append(label)
    else:
        valImages.append(pixel)
        valLabels.append(label)

datasets = [(trainImages, trainLabels, config.TRAIN_HDF5),(valImages, valLabels, config.VAL_HDF5),(testImages, testLabels, config.TEST_HDF5)]

for images, labels, outputs in datasets:
    print('[INFO] building dataset...')
    writer = HDF5DatasetWriter((len(images), 48, 48), outputs)

    for (image, label) in zip(images, labels):
        writer.add([image], [label])

    writer.close()

f.close()
