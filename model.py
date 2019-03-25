import keras
from keras.preprocessing import text, sequence
from keras import utils, regularizers, metrics, optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import math
import re
import json
from firebase_admin import db, credentials
import pandas as pd
import firebase_admin

class Model:
    def __init__(self):
            print("object")

    def sanitize_features(self, df):
        """
        This function will replace urls with ' LINK '. It
        also makes label column to collect all labels into 
        one column.
        """
        df['labels'] = [[] for _ in range(len(df))]
        for index in range(0, df['body'].size):
            df['body'][index] = re.sub(r'^https?:\/\/.*[\r\n]*', ' LINK ', df['body'][index].encode('utf-8'), flags=re.MULTILINE).lower()
            labels = []
            labels.append(df['audience'][index])
            labels.append(df['emailType'][index])
            labels.append(df['majors'][index])
            labels.append(df['language'][index])
            df['labels'][index] = labels
        return df

    def build_model(self, l1_reg, l2_reg, learning_rate, vocabulary_size, num_classes):
        """
        This function defines the architecture of the model.
        """
        model = Sequential([
            Dense(2048, input_shape = (vocabulary_size,), activation = 'relu', kernel_regularizer=regularizers.l2(l2_reg), activity_regularizer=regularizers.l1(l1_reg)),
            Dense(1024, activation = 'relu', kernel_regularizer=regularizers.l2(l2_reg), activity_regularizer=regularizers.l1(l1_reg)),
            Dense(num_classes, activation='sigmoid')
        ])
        model.compile(optimizer = optimizers.Adam(lr = learning_rate),
                    loss ='binary_crossentropy',
                    metrics = [metrics.binary_accuracy])
        return model

    def train_model_fit(self, model, tokenized_emails_train, encoded_label_training, tokenized_emails_validation, encoded__label_validation, batch_size = 150, steps = 100, epochs = 5, verbose = 1, validation_split = 0.5):
        """
        This function will train and evaluate the model this function also returns the trained model.
        """
        train = model.fit(tokenized_emails_train, encoded_label_training,
                        batch_size= batch_size, 
                        epochs= epochs, 
                        verbose= verbose,
                        validation_split= validation_split)
        evaluation = model.evaluate(tokenized_emails_validation,
                                encoded__label_validation, 
                                batch_size= batch_size, 
                                verbose= verbose)
        return train
    
    def train_model(self):
        """
        This is the function that hanldes retraining of the model.
        """
        if (not len(firebase_admin._apps)):
            cred = credentials.Certificate('./easy-cartero-firebase-adminsdk-1sp67-ffb2ac83bf.json') 
            default_app = firebase_admin.initialize_app(cred, {'databaseURL': 'https://easy-cartero.firebaseio.com/'})
        # Get a database reference to our posts
        ref = db.reference()
        ref = ref.child('Production/labeled')
        data = ref.get()

        #convert dictionary into array of items
        emails = json.dumps(list(data.values()))
        df = pd.read_json(emails)
        #We have the first 162 emails with NaN so we replace with default value
        for index in range(0,163):
            df['audience'][index] = ['Everyone']

        df = self.sanitize_features(df)
        df.drop(columns=['audience','author', 'date', 'emailType', 'key', 'language', 'majors', 'subject'])
        possible_labels = ['Internship or Job Application','Scholarships & Fellowship','Volunteer','Sports & Fitness','Workshop','Class',
                        'Competition','Conference','Social Events','Sale','Research','Health & Security','Student Associations',
                        'University Announcement', 'Other','Spanish','English','Engineering','Arts & Sciences','Business','Agriculture',
                        'All Majors','Undergraduate Student','Graduate Student','Faculty','All Students','Everyone']
        ## shuffle data
        df.reindex(np.random.permutation(df.index))
        #define training data size
        train_data_ratio = 1.0
        training_size = int(len(df) * train_data_ratio)

        #split data into training and validation data
        training_emails = df['body'][:training_size]
        training_labels = df['labels'][:training_size]

        validation_emails = df['body'][:training_size]
        validation_labels = df['labels'][:training_size]
        vectorizer = CountVectorizer()
        vectorizer.fit_transform(df['body']).todense()

        vocabulary_size = len(vectorizer.vocabulary_) 
        tokenize = text.Tokenizer(num_words=vocabulary_size, char_level=False, lower=False, split=' ')

        tokenize.fit_on_texts(training_emails) # only fit on train
        tokenized_emails_train = tokenize.texts_to_matrix(training_emails)
        tokenized_emails_validation = tokenize.texts_to_matrix(validation_emails)

        # Use sklearn utility to convert label strings to numbered index
        encoder = MultiLabelBinarizer(classes= possible_labels)
        encoder.fit(df['labels'])
        encoded_training = encoder.transform(training_labels)
        encoded_validation = encoder.transform(validation_labels)
        num_classes = len(possible_labels)

        batch_size = 100
        steps = 75
        epochs = 5
        verbose = 1
        validation_split = 0.4
        L1_regularization = 0.0000025
        L2_regularization = 0.00000025
        LEARNING_RATE = 0.001
        Positive_Label_Threshold = 0.5

        model = self.build_model(
            l1_reg = L1_regularization,
            l2_reg = L2_regularization, 
            learning_rate = LEARNING_RATE,
            vocabulary_size = len(vectorizer.vocabulary_),
            num_classes = num_classes
        )

        history = self.train_model_fit(
            model = model,
            tokenized_emails_train = tokenized_emails_train,
            encoded_label_training = encoded_training,
            tokenized_emails_validation = tokenized_emails_validation,
            encoded__label_validation = encoded_validation,
            batch_size = batch_size,
            steps = steps,
            epochs = epochs,
            verbose = verbose,
            validation_split = validation_split
        )


    def extract_labels_from_prediction(self, possible_labels, prediction, threshold):
        labels = []
        for index in range(0, len(possible_labels)):
            if(prediction[index] >= threshold):
                labels.append(possible_labels[index])
        return labels

    def predict(self):
        predictions = model.predict(tokenized_emails_train)
        for i in range(0, len(predictions)):
            predicted_label = extract_labels_from_prediction(encoder.classes_, predictions[i], Positive_Label_Threshold)
            actual_labels = extract_labels_from_prediction(encoder.classes_, encoded_training[i], Positive_Label_Threshold)
            print(training_emails[i])
            print("Predicted labels: " + str(predicted_label) + "\n")
            print("Actual labels: " + str(actual_labels) + "\n") 