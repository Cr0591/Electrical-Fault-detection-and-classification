import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

detection_train = pd.read_excel('detect_dataset.xlsx').dropna(axis=1)
features = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']

detection_data_X = detection_train[features]
detection_data_Y = detection_train['Output (S)']

detection_train_X, detection_test_X, detection_train_Y, detection_test_Y = train_test_split(
    detection_data_X, detection_data_Y, test_size=0.33, random_state=1)
detect_accuracy = list()
detect_error = list()
print('Linear Regression\n')

# Defining different Models for different classification problems
detection_model = linear_model.Lasso(alpha=2.0)
detection_model.fit(detection_train_X, detection_train_Y)
detection_preds = detection_model.predict(detection_test_X)
print('The Error of our Detection Model is: ',
      mean_squared_error(detection_test_Y, detection_preds))
detect_error.append(mean_squared_error(detection_test_Y, detection_preds))
print('The accuracy score of our Detection Model is: ',
      (detection_model.score(detection_test_X, detection_test_Y)))
# Storing accuracy values
detect_accuracy.append(
    (detection_model.score(detection_test_X, detection_test_Y)))
