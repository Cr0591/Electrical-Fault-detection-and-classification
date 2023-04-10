# Importing necessary packages
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier

# Importing the data
detection_train = pd.read_excel('detect_dataset.xlsx').dropna(axis=1)
features = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']

detection_data_X = detection_train[features]
detection_data_Y = detection_train['Output (S)']

detect_accuracy = list()
detect_error = list()

detection_train_X, detection_test_X, detection_train_Y, detection_test_Y = train_test_split(
    detection_data_X, detection_data_Y, test_size=0.33, random_state=1)
print('-----------Close the corresponding plot to move to the next model-----------\n\n')

print('Linear Regression\n')

# Defining different Models for different classification problems
detection_model = linear_model.Lasso(alpha=2.0)
# Fitting the data in different models
detection_model.fit(detection_train_X, detection_train_Y)
# Predicting test values and printing out Mean Squared Error
detection_preds = detection_model.predict(detection_test_X)
print('The Error of our Detection Model is: ',
      mean_squared_error(detection_test_Y, detection_preds))
# storing error values
detect_error.append(mean_squared_error(detection_test_Y, detection_preds))
# Printing out accuracy scores of our models
print('The accuracy score of our Detection Model is: ',
      (detection_model.score(detection_test_X, detection_test_Y)))
# Storing accuracy values
detect_accuracy.append(
    (detection_model.score(detection_test_X, detection_test_Y)))

fig, axs = plt.subplots(1, 2)
fig.set_figwidth(16)
fig.suptitle('Detection Results')
axs[0].set_title('Input')
axs[1].set_title('Predicted')
axs[0].plot(detection_test_X, detection_test_Y, 'o')
axs[1].plot(detection_test_X, detection_preds, 'o')


print('Logistic Regression\n')
# Defining different Models for different classification problems
detection_model = LogisticRegression()
# Fitting the data in different models
detection_model.fit(detection_train_X, detection_train_Y)
# Predicting test values and printing out Mean Squared Error
detection_preds = detection_model.predict(detection_test_X)
print('The Error of our Detection Model is: ',
      mean_squared_error(detection_test_Y, detection_preds))
# storing error values
detect_error.append(mean_squared_error(detection_test_Y, detection_preds))
# Printing out accuracy scores of our models
print('The accuracy score of our Detection Model is: ',
      (detection_model.score(detection_test_X, detection_test_Y)))
# Storing accuracy values
detect_accuracy.append(
    (detection_model.score(detection_test_X, detection_test_Y)))

fig, axs = plt.subplots(1, 2)
fig.set_figwidth(16)
fig.suptitle('Detection Results')
axs[0].set_title('Input')
axs[1].set_title('Predicted')
axs[0].plot(detection_test_X, detection_test_Y, 'o')
axs[1].plot(detection_test_X, detection_preds, 'o')

print('polynomial Regression\n')
detection_model = PolynomialFeatures(2)
detect_linear = LinearRegression()
# Fitting the data in different models
detect_linear.fit(detection_model.fit_transform(
    detection_train_X), detection_train_Y)
detection_preds = detect_linear.predict(
    detection_model.fit_transform(detection_test_X))
print('The Error of our Detection Model is: ',
      mean_squared_error(detection_test_Y, detection_preds))
# storing error values
detect_error.append(mean_squared_error(detection_test_Y, detection_preds))
print('The accuracy score of our Detection Model is: ', (detect_linear.score(
    detection_model.fit_transform(detection_test_X), detection_test_Y)))
# Storing accuracy values
detect_accuracy.append((detect_linear.score(
    detection_model.fit_transform(detection_test_X), detection_test_Y)))

# ### Graphs
fig, axs = plt.subplots(1, 2)
fig.set_figwidth(16)
fig.suptitle('Detection Results')
axs[0].set_title('Input')
axs[1].set_title('Predicted')
axs[0].plot(detection_test_X, detection_test_Y, 'o')
axs[1].plot(detection_test_X, detection_preds, 'o')

print('Multi layer pereptron\n')
# Defining different Models for different classification problems
detection_model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                hidden_layer_sizes=(5, 2), random_state=1, max_iter=1000)
# Fitting the data in different models
detection_model.fit(detection_train_X, detection_train_Y)
# Predicting test values and printing out Mean Squared Error
detection_preds = detection_model.predict(detection_test_X)
print('The Error of our Detection Model is: ',
      mean_squared_error(detection_test_Y, detection_preds))
# storing error values
detect_error.append(mean_squared_error(detection_test_Y, detection_preds))
# Printing out accuracy scores of our models
print('The accuracy score of our Detection Model is: ',
      (detection_model.score(detection_test_X, detection_test_Y)))
# Storing accuracy values
detect_accuracy.append(
    (detection_model.score(detection_test_X, detection_test_Y)))

fig, axs = plt.subplots(1, 2)
fig.set_figwidth(16)
fig.suptitle('Detection Results')
axs[0].set_title('Input')
axs[1].set_title('Predicted')
axs[0].plot(detection_test_X, detection_test_Y, 'o')
axs[1].plot(detection_test_X, detection_preds, 'o')

print('Naive Bayes\n')

# Defining different Models for different classification problems
detection_model = GaussianNB()
# Fitting the data in different models
detection_model.fit(detection_train_X, detection_train_Y)
detection_preds = detection_model.predict(detection_test_X)
print('The Error of our Detection Model is: ',
      mean_squared_error(detection_test_Y, detection_preds))
# storing error values
detect_error.append(mean_squared_error(detection_test_Y, detection_preds))
# Printing out accuracy scores of our models
print('The accuracy score of our Detection Model is: ',
      (detection_model.score(detection_test_X, detection_test_Y)))
# Storing accuracy values
detect_accuracy.append(
    (detection_model.score(detection_test_X, detection_test_Y)))

fig, axs = plt.subplots(1, 2)
fig.set_figwidth(16)
fig.suptitle('Detection Results')
axs[0].set_title('Input')
axs[1].set_title('Predicted')
axs[0].plot(detection_test_X, detection_test_Y, 'o')
axs[1].plot(detection_test_X, detection_preds, 'o')

fig, ax = plt.subplots(1, 2)
fig.set_figwidth(16)
fig.suptitle('Detection Model comparison')
x = [0, 1, 2, 3, 4]

ax[0].set_xticks(x)
ax[0].set_xticklabels(
    ['Linear', 'Logistic', 'Polynomial', 'MLPC', 'Naive Bayes'])
ax[0].set_title('Accuracy')
ax[0].plot(detect_accuracy, '*')
ax[0].plot(detect_accuracy)
for i in range(len(detect_accuracy)):
    detect_accuracy[i] = round(detect_accuracy[i], 4)
for i, j in zip(x, detect_accuracy):
    ax[0].annotate(str(j), xy=(i, j))

ax[1].set_xticks(x)
ax[1].set_xticklabels(
    ['Linear', 'Logistic', 'Polynomial', 'MLPC', 'Naive Bayes'])
ax[1].set_title('Error')
ax[1].plot(detect_error, '*')
ax[1].plot(detect_error)
for i in range(len(detect_error)):
    detect_error[i] = round(detect_error[i], 4)
for i, j in zip(x, detect_error):
    ax[1].annotate(str(j), xy=(i, j))
plt.show()
