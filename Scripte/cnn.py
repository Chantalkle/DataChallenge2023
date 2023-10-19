import tensorflow as tf
import numpy as np
# import pickle
import os
import csv
import pandas
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
"""
TO DO: preprocessing, neue models, var für obv rev etc bfunktion für pr3edictions (nue werte ausgeben) auch on batch.
"""

class ConvNet():

    def __init__(self, train_path, test_path, model='ResNet50V2',  img_size=(224, 224, 3), mint=''):
        """
        train_path: Path to training data. Should be in a tree structure.
        test_path: Path to test data. Should be in a tree structure.
        model: which ResNet model to use. Default ResNet50V2 other options ResNet101V2 and ResNet152V2
        img_size: Input size for the CNN. Default input size of ResNet (224,224,3)
        """
        self.mint = mint
        self.model_name = model
        self.img_size = img_size
        self.num_class = len(os.listdir(train_path))
        self.load_model(name=model)
        self.train_path = train_path
        self.test_path = test_path
        # self.classes = os.listdir(self.train_path)
        self.callbacks = []
        self.optimizer = None  
        self.augmentation = False
    def load_model(self, name):
        """
        Loads a pretrained Model with a dense layer on top. Size of the dense layer equal to the number of classes.
        Other architectures can be added. See https://www.tensorflow.org/api_docs/python/tf/keras/applications/ .
        Don't forget to change the preprocessing function!! 
        """
        if name == 'ResNet50V2':
            # https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50V2
            self.base_model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet',
                                                            input_shape=self.img_size, pooling='avg'
                                                            )
            self.preprocessing = tf.keras.applications.resnet_v2.preprocess_input
            self.model = tf.keras.Sequential()
            self.model.add(self.base_model)
            self.model.add(tf.keras.layers.Dense(
                        self.num_class, activation='softmax'))
        elif name == 'ResNet101V2':
            # https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet101V2
            self.base_model = tf.keras.applications.ResNet101V2(include_top=False, weights='imagenet',
                                                            input_shape=self.img_size, pooling='avg'
                                                            )
            self.preprocessing = tf.keras.applications.resnet_v2.preprocess_input
            self.model = tf.keras.Sequential()
            self.model.add(self.base_model)
            self.model.add(tf.keras.layers.Dense(
                        self.num_class, activation='softmax'))
        elif name == 'ResNet152V2':
            # https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet152V2
            self.base_model = tf.keras.applications.ResNet152V2(include_top=False, weights='imagenet',
                                                            input_shape=self.img_size, pooling='avg'
                                                            )
            self.preprocessing = tf.keras.applications.resnet_v2.preprocess_input
            self.model = tf.keras.Sequential()
            self.model.add(self.base_model)
            self.model.add(tf.keras.layers.Dense(
                        self.num_class, activation='softmax'))
        
        elif name == 'VGG16':
            self.base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet',input_shape=self.img_size,
                    include_top=False)
            self.base_model.trainable = False 
            self.preprocessing = tf.keras.applications.vgg16.preprocess_input

            self.model = tf.keras.Sequential([ self.base_model,
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(4096, activation='relu'),
                                        tf.keras.layers.Dense(4096, activation='relu'),
                                        #layers.Dropout(0.2),
                                        tf.keras.layers.Dense(self.num_class, activation='softmax')])
        elif name == 'EfficientNetB7':
            self.base_model = tf.keras.applications.efficientnet.EfficientNetB7(weights='imagenet',input_shape=self.img_size,
                    include_top=False)
            self.base_model.trainable = False 
            self.preprocessing = tf.keras.applications.efficientnet.preprocess_input

            self.model = tf.keras.Sequential()
            self.model.add(self.base_model)
            self.model.add(tf.keras.layers.GlobalAveragePooling2D())
            self.model.add(tf.keras.layers.Dropout(0.2))
            self.model.add(tf.keras.layers.Dense(
                        self.num_class, activation='softmax'))

    def load_coin_model(self, path ): #,mint = ''):
        """
        Loads a Tensorflow/Keras Model
        Data should be properly preprocessed!! 
        """
        self.model = tf.keras.models.load_model(path)
        # self.mint = mint

    def freeze(self, layer_name='conv5'):
        """
        Freeze all layer until a layer named @name appears 
        """
        for layer in self.base_model.layers:
            if layer.name.startswith(layer_name):
                break
            layer.trainable = False
    def set_optimizer(self,optimizer = tf.keras.optimizers.Adam(),
                      optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.00001)
        ):
        """ 
        Set the optimizer for training. First trained with optimizer then wit second optimizer after 10 Epochs
        """
        self.optimizer = optimizer
        self.seconde_optimizer = optimizer2
        self.model.compile(optimizer=self.optimizer,
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                        #from_logits=True),
                    metrics=[tf.keras.metrics.CategoricalAccuracy()])
        

    def set_callbacks(self, callback_list=[], set_default=False):
        """
        Adding callbacks.
        If set_default is true early stopping and checkpoint are added to the list.
        https://www.tensorflow.org/api_docs/python/tf/keras/callbacks list of all callbacks
        """
        self.callbacks += callback_list
        if set_default:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath='best_weights.hdf5',
                monitor='val_sparse_categorical_accuracy',
                mode='auto', save_freq='epoch', save_best_only=True)
            early = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=0.001, patience=10, verbose=0, mode='auto',
                baseline=None, restore_best_weights=False)
            self.callbacks.append(early)
            self.callbacks.append(checkpoint)

    def preprocess_images(self, x, y):
        return self.preprocessing(x), y

    def prepare(self, data):
        """
        preprocessing the images
        """
        data = data.map(self.preprocess_images)
        return data


    def load_dataset(self, batch_size, val_split=0.2, seed=None):
        """
        Function to load and preprocess the datasets.
        batch_size: Batch size for training. Test batch size must be  changed manually in the code!
        val_split: Split size for training/validation split
        https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory for more information
        """

        self.train_set = tf.keras.utils.image_dataset_from_directory(
            self.train_path, labels='inferred', label_mode='categorical',
            batch_size=batch_size, image_size=self.img_size[:-1], 
            shuffle=True, seed=123,validation_split=val_split, 
            color_mode='rgb', subset="training"
        )

        self.val_set = tf.keras.utils.image_dataset_from_directory(
            self.train_path, labels='inferred', label_mode='categorical',
            batch_size=batch_size, image_size=self.img_size[:-1], 
            shuffle=True, seed=123,validation_split=val_split,
            color_mode='rgb', subset="training"
        )
        self.test_set = tf.keras.utils.image_dataset_from_directory(
            self.test_path, labels='inferred', label_mode='categorical',
            image_size=self.img_size[:-1], 
            shuffle=False, color_mode='rgb',
            batch_size=128
        )

        self.train_set = self.prepare(self.train_set)
        self.val_set = self.prepare(self.val_set)
        self.test_set = self.prepare(self.test_set)

    def load_image(self, path, preprocessing=True):
        """
        Function to load a single Image.
        Intended for testing purposes 
        path: path to image
        preprocessing: Default True. Preprocesses the image with self.preprocess
        """
        image = tf.keras.preprocessing.image.load_img(
            path=path, target_size=self.img_size[:-1])
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        if preprocessing:
            img_array = self.preprocessing(img_array).reshape(-1, 224, 224, 3)
        return img_array

    def train(self, CNN_name="ResNet"):
        """
        Function for training the network.
        5 Epochs with learning rate 0.001
        20 Epochs with learning rate 0.00001
        """
        # if (self.augmentation):
        #     self.history5 = self.model.fit(self.augmented_dataset, epochs=10,
        #                                 validation_data=self.val_set  # , verbose=2
        #                                 )
        # else:
        self.history5 = self.model.fit(self.train_set, epochs=10,
                                    validation_data=self.val_set  # , verbose=2
                                    )
        history_dataframe = pandas.DataFrame(self.history5.history)
        history_csv_file = CNN_name + '_history.csv'
        with open(history_csv_file, mode='w') as his_file:
            history_dataframe.to_csv(his_file)
        test_loss, test_acc = self.model.evaluate(self.test_set  # , verbose=2
                                                )
        print("First training finished.")
        print(CNN_name, 'loss', test_loss, 'acc', test_acc)

        

        self.model.compile(optimizer=self.seconde_optimizer,
                            loss=tf.keras.losses.CategoricalCrossentropy(),
                                #from_logits=True),
                            metrics=[tf.keras.metrics.CategoricalAccuracy()])
        if (self.augmentation):
            self.history100 = self.model.fit(self.augmented_dataset, callbacks=self.callbacks,
                                                epochs=20, validation_data=self.val_set, verbose=2
                                                )
        else:
            self.history100 = self.model.fit(self.train_set, callbacks=self.callbacks,
                                                epochs=10, validation_data=self.val_set, verbose=2
                                                )
        history_dataframe = pandas.DataFrame(self.history100.history)
        with open(history_csv_file, mode='w') as his_file:
            history_dataframe.to_csv(his_file)

        test_loss, test_acc = self.model.evaluate(self.test_set, verbose=2
                                                )

        print(CNN_name, 'loss', test_loss, 'acc', test_acc)

        save_name = CNN_name + '_' + self.mint + ".h5"
        self.model.save(save_name)

        print("sucess")

    def apply_dataaugmentation(self,flip="horizontal",rot=0.2):
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(flip),
            tf.keras.layers.RandomRotation(rot),
        ])
        self.augmented_dataset = self.train_set.map(lambda x, y: (self.data_augmentation(x, training=True), y))
        self.augmentation = True

    def get_last_layer_activations(self, path,save_name='default',set_name='default'):
        """ 
        Function to get the prediction of data and save it as an numpy array
        """
        self.batch_set = tf.keras.utils.image_dataset_from_directory(
                path, labels='inferred', label_mode='categorical',
                image_size=self.img_size[:-1],
                batch_size=64,shuffle=False, color_mode='rgb'
        )
        self.batch_set = self.prepare(self.batch_set)
        results = []
        # Iterate over the data generator and perform predictions on each batch
        for batch_images, batch_labels in self.batch_set:
            predictions = self.model.predict_on_batch(batch_images)
            
            # Store the activations and labels in the results list
            for prediction, label in zip(predictions, batch_labels):
                results.append([prediction.tolist(), label])

        data_array = np.array(results)

        # Save as a NumPy binary file
        np.save(save_name + '_' +set_name+ '_' + self.mint + '_' +'.npy', data_array)
        return results



def __main__():
    cnn = ConvNet('train/', 'test/')
    cnn.freeze()
    cnn.set_callbacks([], True)
    cnn.load_dataset(batch_size=128, seed=1)
    # cnn.train()
    path = './test/CATII-H-17544A.jpg'


