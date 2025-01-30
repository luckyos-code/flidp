import tensorflow as tf
from tensorflow.keras.layers import Input, Rescaling, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GroupNormalization, Resizing, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications import EfficientNetV2B0, MobileNetV2


def get_model(model_name, input_shape, num_classes, rescale_factor, compile=False):
    if model_name == "simple-cnn":
        return get_simple_cnn(input_shape, num_classes, rescale_factor, compile=compile)
    elif model_name == "lucasnet":
        return get_lucasnet(input_shape, num_classes, rescale_factor, compile=compile)
    elif model_name == "efficientnet-raw":
        return get_efficientnet_raw(input_shape, num_classes, compile=compile)
    elif model_name == "efficientnet-frozen":
        return get_efficientnet_frozen(input_shape, num_classes, compile=compile)
    elif model_name == "efficientnet-imagenet":
        return get_efficientnet_imagenet(input_shape, num_classes, compile=compile)
    elif model_name == "mobilenet":
        return get_mobilenet(input_shape, num_classes, compile=compile)
    else:
        raise NotImplementedError(f"There is no model implemented for {model_name}")

def get_simple_cnn(input_shape, num_classes, rescale_factor, compile=False):
    model = Sequential([
        Input(input_shape),
        Rescaling(rescale_factor),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.1),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.1),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.1),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(num_classes, activation='softmax'),
    ])
    if compile:
        model.compile(
            optimizer=Adam(learning_rate=1e-3), 
            loss=SparseCategoricalCrossentropy(), 
            metrics=['accuracy']
        )
    
    return model


def get_lucasnet(input_shape, num_classes, rescale_factor, compile=False):
    groups = 32
    model = Sequential([
        Input(shape=input_shape),
        Rescaling(rescale_factor),  # added Rescaling to scale features to [0,1]
        Resizing(64, 64),  # added resizing to resize to the default value in the original function
#        RandomFlip(
#            "horizontal",
#            seed=42,
#        ),
        Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation="relu",),
        GroupNormalization(groups=groups),
        MaxPooling2D(2, 2),
        Dropout(0.1),  # also added Dropout for regularization
        Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation="relu",),
        GroupNormalization(groups=groups),
        MaxPooling2D(2, 2),
        Dropout(0.1),
        Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", activation="relu",),
        GroupNormalization(groups=groups),
        MaxPooling2D(2, 2),
        Dropout(0.1),
        Flatten(),
        Dense(512, activation="relu"),
        GroupNormalization(groups=groups),
        Dropout(0.1),
        Dense(num_classes, activation="softmax"),
    ])
    
    if compile:
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
    
    return model

def get_efficientnet_raw(input_shape, num_classes, compile=False):
    model = EfficientNetV2B0(
            include_top=True,
            classes=num_classes,
            input_shape=input_shape,
            weights=None,
            include_preprocessing=True)
    
    model.trainable = True # make sure efficientnet is trainable
    
    if compile:
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss=SparseCategoricalCrossentropy(), 
            metrics=['accuracy']
        )
    
    return model


def get_efficientnet_imagenet(input_shape, num_classes, compile=False):
    model = Sequential([
        Input(input_shape),
        Resizing(224, 224),
        EfficientNetV2B0(
            include_top=False, # for using imagenet pretraining but on less classes
            input_shape=(224, 224,3),
            weights='imagenet',
            include_preprocessing=True
        ),
        GlobalAveragePooling2D(),  # Pool the features to reduce dimensions
        Dense(num_classes, activation='softmax')  # Add a custom classification head
    ])
    
    model.trainable = True # make sure efficientnet is trainable
    
    if compile:
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss=SparseCategoricalCrossentropy(), 
            metrics=['accuracy']
        )
    
    return model

def get_efficientnet_frozen(input_shape, num_classes, compile=False):
    effnet = EfficientNetV2B0(
            include_top=False, # for using imagenet pretraining but on less classes
            input_shape=(224, 224, 3),
            weights='imagenet',
            include_preprocessing=True)

    effnet.trainable = False # efficientnet_body
    
    model = Sequential([
        Input(input_shape),
        Resizing(224, 224),
        effnet,
        GlobalAveragePooling2D(),  # Pool features to reduce dimensions
        Dropout(0.5),  # Add dropout for regularization
        Dense(128, activation='relu'),  # Add a dense layer for feature extraction
        Dropout(0.5),  # Add another dropout layer
        Dense(num_classes, activation='softmax')  # Add a custom classification head
    ])
    
    if compile:
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss=SparseCategoricalCrossentropy(), 
            metrics=['accuracy']
        )
    
    return model


def get_mobilenet(input_shape, num_classes, compile=False):
    model = Sequential([
        Input(input_shape),
        Resizing(224, 224),
        MobileNetV2(
            include_top=False, # for using imagenet pretraining but on less classes
            input_shape=(224, 224,3),
            weights='imagenet',
            include_preprocessing=True
        ),
        GlobalAveragePooling2D(),  # Pool the features to reduce dimensions
        Dense(num_classes, activation='softmax')  # Add a custom classification head
    ])
   
    model.trainable = True # make sure efficientnet is trainable
    
    if compile:
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss=SparseCategoricalCrossentropy(), 
            metrics=['accuracy']
        )
    
    return model