import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
# import matplotlib.pyplot as plt


CHECKPOINT_PATH = "training_1/cp.ckpt"
CHECKPOINT_PATH = "training_3/cp-{epoch:04d}.ckpt"
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0


def create_model():
	model = keras.Sequential([
	    keras.layers.Flatten(input_shape=(28, 28)),
	    keras.layers.Dense(128, activation='relu'),
	    keras.layers.Dense(10),
		keras.layers.Softmax()
	])

	model.compile(optimizer='adam',
	              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	              metrics=['accuracy'])
	return model

def run_with_checkpoint(model):
	cp_callback = keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
	                                             save_weights_only=True,
	                                             verbose=1)

	# model.save_weights(CHECKPOINT_PATH.format(epoch=0))
	model.fit(train_images, train_labels, epochs=100, validation_data=(test_images,test_labels), callbacks=[cp_callback], verbose=0)
	return model


def run(model):
	model.fit(train_images, train_labels, epochs=100)


def run_with_h5(model):

	model.fit(train_images, train_labels, epochs=5)
	model.save('my_model_1.h5')


def load_from_h5():
	return tf.keras.models.load_model('my_model_1.h5')

def load_from_checkpoint():
	model = create_model()
	model.load_weights(tf.train.latest_checkpoint(CHECKPOINT_DIR))
	# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
	# print('\nTest accuracy:', test_acc)
	return model


def save_my_model(model):

	MODEL_DIR = "my_model"
	version = 1
	export_path = os.path.join(MODEL_DIR, str(version))
	print('export_path = {}\n'.format(export_path))

	tf.keras.models.save_model(
	    model,
	    export_path,
	    overwrite=True,
	    include_optimizer=True,
	    save_format=None,
	    signatures=None,
	    options=None
	)

model = create_model()
run(model)
save_my_model(model)




# AUTOTUNE = tf.data.experimental.AUTOTUNE
# import pathlib
# data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
#                                          fname='flower_photos', untar=True)
# data_root = pathlib.Path(data_root_orig)
# print(data_root)


# probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# predictions = probability_model.predict(test_images)
# print(predictions[0])
