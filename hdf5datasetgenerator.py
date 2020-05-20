from tensorflow.keras.utils import to_categorical
import numpy as np
import h5py

class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, preprocessors=None, aug=None, binarize=True, classes=2):
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        self.db = h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]

    def generator(self, passes = np.inf):
        epochs = 0
        while epochs < passes:
            for i in range(0, self.numImages, self.batchSize):
                images = self.db['images'][i : i + self.batchSize]
                labels = self.db['labels'][i : i + self.batchSize]
                
                if self.binarize:
                    labels = to_categorical(labels, num_classes = self.classes)

                if self.preprocessors is not None:
                    proImages = []
                    for p in self.preprocessors:
                      for image in images:
                          image = p.preprocess(image)
                          # image = np.squeeze(image, axis = -1)
                          # image = np.repeat(image[..., np.newaxis], 3, -1)
                          proImages.append(image)
                    
                    images = np.array(proImages)

                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images, labels, batch_size=self.batchSize))

                yield (images, labels)

            epochs += 1

    def close(self):
        self.db.close()