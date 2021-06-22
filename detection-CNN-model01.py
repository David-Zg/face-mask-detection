#%%
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix, classification_report

#%% Functions
def build_model():

  model = Sequential()

  model.add(Conv2D(32, (3, 3), activation='relu',
                       input_shape=(28, 28, 3)))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))

  model.add(Flatten())

  model.add(Dropout(0.5))

  model.add(Dense(1, activation = "sigmoid", kernel_regularizer='l2'))

  return model

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="red" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./confusion.png',dpi=300)

#%% Callbacks

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', min_delta=0, patience=2, verbose=0,
    mode='auto', baseline=None, restore_best_weights=False)

#%% Variable
batch = 200
size = (28,28)

#%% Preprocessing
base_dir = "path"

train_dir       = os.path.join(base_dir, "Train")
validation_dir  = os.path.join(base_dir, "Validation")
test_dir        = os.path.join(base_dir, "Test")

train_datagen        = ImageDataGenerator(rescale = 1.0/255.0)
validation_datagen   = ImageDataGenerator(rescale = 1.0/255.0)
test_datagen         = ImageDataGenerator(rescale = 1.0/255.0)

train_generator      = train_datagen.flow_from_directory(train_dir,
                                                         batch_size = batch,
                                                         class_mode = "binary",
                                                         target_size = size)

validation_generator = train_datagen.flow_from_directory(validation_dir,
                                                         batch_size = batch,
                                                         class_mode = "binary",
                                                         target_size = size)

#%% Building a model
model = build_model()

#%% Compiling a model
model.compile(optimizer = RMSprop(lr = 1e-4),
              loss = "binary_crossentropy",
              metrics = ["accuracy"])
              
#%% Training the model
history = model.fit(train_generator,
                    steps_per_epoch = train_generator.n // batch,
                    epochs = 25,
                    validation_data = validation_generator,
                    validation_steps = validation_generator.n // batch,
                    callbacks=[callback],
                    verbose = 1)
                
#%% Results Training
acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs    = range(1,len(acc)+1,1) 

plt.plot  ( epochs,     acc, 'r--', label='Training acc'  )
plt.plot  ( epochs, val_acc,  'b', label='Validation acc')
plt.title ('Training and Validation Accuracy')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.savefig('path/train1.png',dpi=300)
plt.legend()
plt.figure()


plt.plot  ( epochs,     loss, 'r--', label='Training recall'  )
plt.plot  ( epochs, val_loss ,  'b', label='Validation recall' )
plt.title ('Training and Validation Loss'   )
plt.ylabel('loss')
plt.xlabel('epochs')
plt.savefig('path/train2.png',dpi=300)
plt.legend()
plt.figure()

#%% Confusion matrix

test_generator  = train_datagen.flow_from_directory(test_dir,
                                                    batch_size = 32,
                                                    class_mode = "binary",
                                                    target_size = size,
                                                    shuffle=False)

test_lost, test_acc= model.evaluate(test_generator)
print ("Test Accuracy:", test_acc)

predictions = model.predict(test_generator, verbose=0)

labels_test = []
labels = []
for i in range (test_generator.n // 32):
    _, test_labels = next(test_generator)
    labels.append(test_labels)

cm = confusion_matrix(np.array(labels).flatten(), predictions>0.5)

cm_plot_labels = ['WithMask','WithoutMask']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

report = classification_report(np.array(labels).flatten(), predictions>0.5)
print(report)

#%% Save model Keras
model.save('path/mascara1.h5')

#%% Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('path/model1.tflite', 'wb') as f:
  f.write(tflite_model)
