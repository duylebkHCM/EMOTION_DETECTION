from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ELU
def TranserLearningModel(inputshape = (48, 48, 3), numclasses = 6):
    pretrained_model = VGG19(input_shape = inputshape, weights = 'imagenet', include_top = False)

    for layer in pretrained_model.layers:
        layer.trainable = False

    layer_output = pretrained_model.output
    x = Flatten()(layer_output)
    x = Dense(1024, kernel_initializer = "he_normal")(x)
    x = Activation(ELU())(x)
    x = BatchNormalization(axis = -1)(x)
    x = Dropout(0.25)(x)
    
    x = Dense(512, kernel_initializer = "he_normal")(x)
    x = Activation(ELU())(x)
    x = BatchNormalization(axis = -1)(x)
    x = Dropout(0.5)(x)

    x = Dense(numclasses, kernel_initializer = "he_normal")(x)
    x = Activation('softmax')(x)

    model = Model(pretrained_model.input, outputs = x)

    return model


model = TranserLearningModel()
model.summary()