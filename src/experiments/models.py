import tensorflow as tf
from tensorflow.keras.layers import Input, Rescaling, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GroupNormalization, Resizing
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications import EfficientNetV2B0, MobileNetV2


def get_model(model_name, input_shape, num_classes, rescale_factor):
    if model_name == "simple-cnn":
        return get_simple_cnn(input_shape, num_classes, rescale_factor, False)
    elif model_name == "lucasnet":
        return get_lucasnet(input_shape, num_classes, rescale_factor, False)
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


def get_efficientnet(input_shape, num_classes, compile=False):
    model = Sequential([
        Input(input_shape),
        Resizing(224, 224),
        EfficientNetV2B0(
            include_top=True,
            weights=None,
            pooling=None,
            classes=num_classes,
            classifier_activation='softmax',
            include_preprocessing=True
        )
    ])
    if compile:
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss=SparseCategoricalCrossentropy(from_logits=False), 
            metrics=['accuracy']
        )
    
    return model


def get_mobilenet(input_shape, num_classes, compile=False):
    model = Sequential([
        Input(input_shape),
        Resizing(224, 224),
        MobileNetV2(
            classes=num_classes,
            weights=None,
            include_top=True,
        )
    ])

    if compile:
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss=SparseCategoricalCrossentropy(from_logits=False), 
            metrics=['accuracy']
        )
    
    return model