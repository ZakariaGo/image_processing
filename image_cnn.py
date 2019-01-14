import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.utils import to_categorical

# Part 1 : Training a small CNN

# 1) loading CIFAR-10 data and rescaling it

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train
X_test = X_test
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2) Using an ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
)
train_datagen.fit(X_train)

# 3) creating a simple CNN model
model = Sequential()
model.add(Conv2D(
    filters=64,
    kernel_size=(3, 3),
    input_shape=(32, 32, 3),
    activation='relu'
))
model.add(Conv2D(
    filters=128,
    kernel_size=(3, 3),
    activation='relu'
))
model.add(Conv2D(
    filters=256,
    kernel_size=(3, 3),
    activation='relu'
))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

# 3) training model
model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

model.fit_generator(
    generator=train_datagen.flow(X_train, y_train, batch_size=32),
    epochs=10,
    steps_per_epoch=len(X_train)/32
)


# Part 2 : Using VGG16 to make a prediction

# 1) loading and checking model
vgg = VGG16(weights='imagenet', include_top=True)
vgg.summary()

# 2) preparing image
img = img_to_array(load_img('hamster.jpg', target_size=(224, 224)))
img = preprocess_input(img)

# 3) adding a dimension
img = np.expand_dims(img, axis=0)

# 4) making prediction
prediction = decode_predictions(vgg.predict(img))
print(prediction)


# Part 3 : Fine-tuning VGG16 on a new dataset

# 1) loading pretrained model without fully connected layers
vgg = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)
vgg.summary()

# 2) adding a new classification layer and training it on CIFAR10
model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

# 3) Freezing VGG16
model.get_layer("vgg16").trainable = False

print([(layer, layer.trainable) for layer in model.layers])

# 4) Compiling and Training
model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(lr=0.01, momentum=0.9),
    metrics=['accuracy']
)

model.fit_generator(
    generator=train_datagen.flow(X_train, y_train, batch_size=32),
    epochs=10,
    steps_per_epoch=len(X_train)/32
)
loss, accuracy = model.evaluate(X_test, y_test)

print("Final loss : %s" % loss)
print("Final accuracy : %s" % (accuracy * 100))

# 5) This network is too deep for 32x32 images

# 6) By reducing the amount of used layers from VGG16
