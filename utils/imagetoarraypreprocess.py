from tensorflow.keras.preprocessing.image import img_to_array

class ImageToArrayPreprocess:
    def preprocess(self, image):
        return img_to_array(image, dtype = 'float32')