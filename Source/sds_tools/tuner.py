
import numpy
from keras.layers import Dense, BatchNormalization, Activation
from keras.models import Sequential
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


# Function to create model, required for KerasClassifier
def create_model(neurons=1):
    model = Sequential()
    model.add(
        Dense(
            neurons, input_shape=(797,), kernel_initializer='lecun_uniform',
            kernel_regularizer=l2(0.001)
        )
    )
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='linear'))
    model.compile(
        loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(
        loss='mean_squared_error', metrics=['mean_absolute_error'],
        optimizer='Nadam'
    )

    return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]
# create model
model = KerasClassifier(
    build_fn=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
neurons = [1, 5, 10, 15, 20, 25, 30]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)
# summarize results
print('Best: {} using {}'.format(
    grid_result.best_score_, grid_result.best_params_)
)
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{} ({}) with: {}'.format(mean, stdev, param))
