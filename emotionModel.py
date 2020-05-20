from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout, Flatten, Dense, Input
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import ELU

def EmotionDetectModel(inputshape = (48, 48, 1), numclasses = 6):
    input = Input(inputshape)
    
    #Block 1
    x = Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', strides = (1,1), name = 'conv2d_11', kernel_initializer = "he_normal")(input)
    x = Activation(ELU())(x)
    x = BatchNormalization(axis = -1, name = 'bn_11')(x)
    x = Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', strides = (1,1), name = 'conv2d_12', kernel_initializer = "he_normal")(x)
    x = Activation(ELU())(x)
    x = BatchNormalization(axis = -1, name = 'bn_12')(x)
    x = MaxPooling2D(pool_size = (2,2), name = 'pool_11')(x)
    x = Dropout(0.25, name = 'drop_11')(x)
    
    #Block 2
    x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', strides = (1,1), name = 'conv2d_21', kernel_initializer = "he_normal")(x)
    x = Activation(ELU())(x)
    x = BatchNormalization(axis = -1, name = 'bn_21')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', strides = (1,1), name = 'conv2d_22', kernel_initializer = "he_normal")(x)
    x = Activation(ELU())(x)
    x = BatchNormalization(axis = -1, name = 'bn_22')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', strides = (1,1), name = 'conv2d_23', kernel_initializer = "he_normal")(x)
    x = Activation(ELU())(x)
    x = BatchNormalization(axis = -1, name = 'bn_23')(x)
    x = MaxPooling2D(pool_size = (2,2), name = 'pool_21')(x)
    x = Dropout(0.25, name = 'drop_21')(x)
    
    #Block 3
    x = Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', strides = (1,1), name = 'conv2d_31', kernel_initializer = "he_normal")(x)
    x = Activation(ELU())(x)
    x = BatchNormalization(axis = -1, name = 'bn_31')(x)
    x = Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', strides = (1,1), name = 'conv2d_32', kernel_initializer = "he_normal")(x)
    x = Activation(ELU())(x)
    x = BatchNormalization(axis = -1, name = 'bn_32')(x)
    x = Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', strides = (1,1), name = 'conv2d_33', kernel_initializer = "he_normal")(x)
    x = Activation(ELU())(x)
    x = BatchNormalization(axis = -1, name = 'bn_33')(x)
    x = Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', strides = (1,1), name = 'conv2d_34', kernel_initializer = "he_normal")(x)
    x = Activation(ELU())(x)
    x = BatchNormalization(axis = -1, name = 'bn_34')(x)
    x = MaxPooling2D(pool_size = (2,2), name = 'pool_31')(x)
    x = Dropout(0.25, name = 'drop_31')(x)
    
    #Block 4
    x = Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', strides = (1,1), name = 'conv2d_41', kernel_initializer = "he_normal")(x)
    x = Activation(ELU())(x)
    x = BatchNormalization(axis = -1, name = 'bn_41')(x)
    x = Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', strides = (1,1), name = 'conv2d_42', kernel_initializer = "he_normal")(x)
    x = Activation(ELU())(x)
    x = BatchNormalization(axis = -1, name = 'bn_42')(x)
    x = Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', strides = (1,1), name = 'conv2d_43', kernel_initializer = "he_normal")(x)
    x = Activation(ELU())(x)
    x = BatchNormalization(axis = -1, name = 'bn_43')(x)
    x = Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', strides = (1,1), name = 'conv2d_44', kernel_initializer = "he_normal")(x)
    x = Activation(ELU())(x)
    x = BatchNormalization(axis = -1, name = 'bn_44')(x)
    x = MaxPooling2D(pool_size = (2,2), name = 'pool_41')(x)
    x = Dropout(0.4, name = 'drop_41')(x)

    # #Block 5
    # x = Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', strides = (1,1), name = 'conv2d_51', kernel_initializer = glorot_uniform(seed = 0))(input)
    # x = BatchNormalization(axis = -1, name = 'bn_51')(x)
    # x = Activation('relu')(x)
    # x = Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', strides = (1,1), name = 'conv2d_52', kernel_initializer = glorot_uniform(seed = 0))(input)
    # x = BatchNormalization(axis = -1, name = 'bn_52')(x)
    # x = Activation('relu')(x)
    # x = Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', strides = (1,1), name = 'conv2d_53', kernel_initializer = glorot_uniform(seed = 0))(input)
    # x = BatchNormalization(axis = -1, name = 'bn_53')(x)
    # x = Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', strides = (1,1), name = 'conv2d_54', kernel_initializer = glorot_uniform(seed = 0))(input)
    # x = BatchNormalization(axis = -1, name = 'bn_54')(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D(pool_size = (2,2), name = 'pool_51')(x)
    # x = Dropout(0.4, name = 'drop_51')(x)
    
    x = Flatten()(x)
    x = Dense(512, kernel_initializer="he_normal", name = 'dense_1')(x)
    x = BatchNormalization(axis = -1, name = 'bn_61')(x)
    x = Activation(ELU())(x)
    x = Dropout(0.5)(x)

    x = Dense(64, kernel_initializer="he_normal", name = 'dense_2')(x)
    x = BatchNormalization(axis = -1, name = 'bn_71')(x)
    x = Activation(ELU())(x)
    x = Dropout(0.5)(x)
    
    x = Dense(numclasses, kernel_initializer="he_normal", name = 'dense_3')(x)
    x = Activation('softmax')(x)
    
    
    model = Model(inputs = input, outputs = x)
    
    return model
    