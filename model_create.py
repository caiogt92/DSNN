#Train a simple deep CNN on the CIFAR10 small images dataset.
from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter


#Path to the training folder containing the truth CSV and the folder with images
pathT = ""

batch_size = 16
num_classes = 4
epochs = 15
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'


# ###Dataset
traindf = pd.read_csv(pathT + '/train.truth.csv', dtype=str)

datagen = ImageDataGenerator(rescale=None, validation_split=0.15)

train_generator=datagen.flow_from_dataframe(dataframe=traindf,
                                            directory= pathT + "/train/",
                                            x_col="fn",
                                            y_col = "label",
                                            subset="training",
                                            batch_size=32,
                                            seed = 20,
                                            shuffle=True,
                                            class_mode="categorical",
                                            target_size=(64,64))

test_generator=datagen.flow_from_dataframe(dataframe=traindf,
                                            directory= pathT + "/train/",
                                            x_col="fn",
                                            y_col = "label",
                                            subset="validation",
                                            batch_size=32,
                                            seed = 20,
                                            shuffle=False,
                                            class_mode="categorical",
                                            target_size=(64,64))
#print(train_generator.class_indices)

# #### Number of images per class --- Oversample/Undersample check
# all_labels = traindf['label'].to_numpy()
# labels_count = Counter(all_labels)
# ax = sns.countplot(all_labels, order=[k for k, _ in labels_count.most_common()])
# ax.set_title('Number of images with a class label')
# ax.set_ylim(1E2, 1E4)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90);



train_generator.reset()
x_train=np.concatenate([train_generator.next()[0] for i in range(train_generator.__len__())])

test_generator.reset()
x_test=np.concatenate([test_generator.next()[0] for i in range(test_generator.__len__())])
y_test=np.concatenate([test_generator.next()[1] for i in range(test_generator.__len__())])


# ###Model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Fit the model on the batches generated
model.fit_generator(train_generator,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# ###Score trained model.
#scores = model.evaluate(x_test, y_test, verbose=1)
#print('Test loss:', scores[0])
#('Test accuracy:', scores[1])