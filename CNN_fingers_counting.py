import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from skimage import color, io
import tensorflow_addons as tfa
import cv2
import glob

#%% load train test val datasets

# norm layer
normalization_layer = tf.keras.layers.Rescaling(1./255)

dataset_path = r"general path to the dataset"
batch_size = 32
img_size = (360, 640)  # original img size = 1080 x 1920 - reduce /3 dim


train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    r'train path',s
    image_size = img_size,
    batch_size = batch_size
).map(lambda x, y: (normalization_layer(x), y))

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    r'val path',
    image_size = img_size,
    batch_size = batch_size
).map(lambda x, y: (normalization_layer(x), y))

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    r'test path',
    image_size = img_size,
    batch_size = batch_size
).map(lambda x, y: (normalization_layer(x), y))


#%% create CNN model
from tensorflow.keras import layers, models

# define model
model = models.Sequential()

# conv layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(360, 640, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# fully connected layer
model.add(layers.Flatten()) 
model.add(layers.Dense(units=128, activation='relu'))
model.add(layers.Dropout(0.5))  # regularization dropout
model.add(layers.Dense(units=6, activation='linear'))  # 6 classes (0-5 fingers)      # activation linear and usng logits=trueor optimal calculus

# model compilation
model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# fit model
history = model.fit(
    train_dataset,  
    epochs=10,       
    validation_data=val_dataset,  
    batch_size=32   
)

# plot cost function
train_loss = history.history['loss']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']


plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss')
plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.legend()
plt.show()
plt.figure()
plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy'), plt.title('Validation Accuracy')
# plt.ylim(0,1)
plt.show()

model.save("CNNv1.keras")

#%% model evaluation
# from tensorflow.keras.models import load_model
# model = load_model('CNNv1.keras')


from sklearn.metrics import confusion_matrix, classification_report

y_true = []  # ground truth
y_pred = []  # predictions

# iterate through images
for images, labels in test_dataset:
    
    predictions = model.predict(images)
    predictions = predictions
    predicted_labels = predictions.argmax(axis=-1)  # result is not a probability because i used logits=true and not softmax act func
    
    # save labels
    y_true.extend(labels.numpy())  
    y_pred.extend(predicted_labels)

# confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# classification repport (precision, recall, f1-score)
report = classification_report(y_true, y_pred)
print("Classification Report:")
print(report)
