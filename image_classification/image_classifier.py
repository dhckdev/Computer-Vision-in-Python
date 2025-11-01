import os
import pickle
import kagglehub
import time
import numpy as np

from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Set dir where image data is stored
data_dir = '/Users/damianhuckele/Developer/Python/ComputerVision/clf-data'
categories = ['empty', 'not_empty']

data = []
labels = []

# start stopwatch for loading and processing data
start_time = time.time()

data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(data_dir, category)):
        img_path = os.path.join(data_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

# split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, shuffle=True)

# train the classifier
classifier = SVC()
parameters = [{'gamma':[0.01, 0.001, 0.0001], 'C': [1,10,100,1000]}]
grid_search = GridSearchCV(classifier, parameters) # classify each combination of the parameters gamma + c
grid_search.fit(X_train, y_train) # fit the model with the training data

# test performance of classifier
best_estimator = grid_search.best_estimator_

# predict values from best estimator
y_prediction = best_estimator.predict(X_test)

# calculate the accuracy score
acc_score = accuracy_score(y_prediction, y_test)
print("{}% of samples were correctly classified".format(str(acc_score*100)))

## f1 score
f1 = f1_score(y_prediction, y_test)
print(f"F1: {f1}")

## confusion matrix
confusion_mtx = None

# stop stopwatch
end_time = time.time()
print(f"Time to load: {end_time-start_time}")

# save the model for later
pickle.dump(best_estimator, open('./model.p0', 'wb'))