import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import cross_val_score

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.dummy import DummyRegressor

from scipy import stats

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import validation_curve


def loadData():
    dataset = pd.read_csv("./dataset.csv")
    label_encoder = LabelEncoder()
    dataset["track_genre_encoded"] = label_encoder.fit_transform(
        dataset["track_genre"])
    dataset["explicit"] = dataset["explicit"].astype(int)
    dataset["duration_ms"] = dataset["duration_ms"].div(1000)

    # drop specific columns
    columns = ['track_id', 'album_name',
               "track_genre", "artists", "track_name"]
    dataset.drop(columns, inplace=True, axis=1)

    # drop first column
    dataset = dataset.iloc[:, 1:]
    return dataset


def randomForestReg(dataset, theFrame):
    labels = np.array(dataset['popularity'])
    feature_list = list(dataset.columns)
    features = np.array(dataset)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)

    print('Training Features Shape:', X_train.shape)
    print('Training Labels Shape:', y_train.shape)
    print('Testing Features Shape:', X_test.shape)
    print('Testing Labels Shape:', y_test.shape)

    # random forest model
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=1000, random_state=42)

    # Train the model on training data
    rf.fit(X_train, y_train)
    # Use the forest's predict method on the test data
    predictions = rf.predict(X_test)

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, linewidth=1, label="original")
    plt.plot(x_ax, predictions, linewidth=1.1, label="predicted")
    plt.title("y-test and y-predicted data")
    plt.xlabel('Row Data #')
    plt.ylabel('Popularity Level')
    plt.legend(loc='best',fancybox=True, shadow=True)
    plt.grid(True)
    plt.show() 

    theFrame.loc[len(theFrame.index)] = [
        "Random Forest Regressor", metrics.mean_absolute_error(
            y_test, predictions),
        metrics.mean_squared_error(y_test, predictions), np.sqrt(
            metrics.mean_squared_error(y_test, predictions))
    ]
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
    print('Root Mean Squared Error:', np.sqrt(
        metrics.mean_squared_error(y_test, predictions)))
    return predictions


def compareToBaseline(predictions, dataset, theFrame):
    labels = np.array(dataset['popularity'])
    feature_list = list(dataset.columns)
    features = np.array(dataset)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)
    # create a dummy regressor
    dummy_reg = DummyRegressor(strategy='mean')
    dummy_reg.fit(X_train, y_train)
    y_pred = dummy_reg.predict(X_test)

    # calculate root mean squared error
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    # print("Dummy RMSE:", rmse)
    # print("Score", dummy_reg.score(X_test, y_test))

    welch_ttest(predictions, y_pred)


def welch_ttest(x, y):
    ## Welch-Satterthwaite Degrees of Freedom ##
    dof = (x.var()/x.size + y.var()/y.size)**2 / ((x.var()/x.size)
                                                  ** 2 / (x.size-1) + (y.var()/y.size)**2 / (y.size-1))

    t, p = stats.ttest_ind(x, y, equal_var=False)

    print("\n",
          f"Welch's t-test= {t:.4f}", "\n",
          f"p-value = {p:.4f}", "\n",
          f"Welch-Satterthwaite Degrees of Freedom = {dof:.4f}")


def sequentialNN(dataset, theFrame):

    # Standardize the data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(dataset)
    dataset = pd.DataFrame(normalized_data, columns=dataset.columns)
    labels = np.array(dataset['popularity'])

    X_train, X_test, y_train, y_test = train_test_split(
        dataset, labels, test_size=0.3, random_state=42)

    model = Sequential()

    # Adding the input layer and the first hidden layer
    model.add(Dense(32, input_shape=(
        X_train.shape[1],), activation='relu', input_dim=6))

    # Adding the second hidden layer
    model.add(Dense(units=32, activation='relu'))

    # Adding the third hidden layer
    model.add(Dense(units=32, activation='relu'))

    # Adding the output layer
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(
        X_train, y_train, validation_split=0.1, batch_size=100, epochs=6)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    predictions = model.predict(X_test)

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, linewidth=1, label="original")
    plt.plot(x_ax, predictions, linewidth=1.1, label="predicted")
    plt.title("y-test and y-predicted data")
    plt.xlabel('Row Data #')
    plt.ylabel('Popularity Level')
    plt.legend(loc='best',fancybox=True, shadow=True)
    plt.grid(True)
    plt.show() 

    theFrame.loc[len(theFrame.index)] = [
        "Sequential NN", metrics.mean_absolute_error(y_test, predictions),
        metrics.mean_squared_error(y_test, predictions), np.sqrt(
            metrics.mean_squared_error(y_test, predictions))
    ]
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
    print('Root Mean Squared Error:', np.sqrt(
        metrics.mean_squared_error(y_test, predictions)))

    return predictions


if __name__ == "__main__":
    dataset = loadData()
    theFrame = pd.DataFrame()
    theFrame["Algorithm"] = []
    theFrame["Mean Absolute Error"] = []
    theFrame["Mean Squared Error"] = []
    theFrame["Root Mean Squared Error"] = []
    predictions = sequentialNN(dataset, theFrame).flatten()
    predictions = randomForestReg(dataset, theFrame)
    print(theFrame)
    compareToBaseline(predictions, dataset, theFrame)
