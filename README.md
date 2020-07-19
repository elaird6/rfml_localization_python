# rfml_localization
> Focus on repository is the location of uncooperative wireless emitters in an indoor environment.  The challenge is that an indoor environment results in dynamic, random propagation of wireless signals.  An approach is to create a RF fingerprint or map of the environment using different RF measurements techniques. Relevant measurements techniques generate a range of data from IQ samples to correlation curves to time-delay estimates (TDE's) for  TDOA; multiple power readings for RSS; and multiple angle of arrival readings for AoA.  The theoretical performance bounds are generated, via CRLB, of empirically measured and derived channel limitations that exists in literature.  From same literature, a simulation model is created that generates the specified random RF channel environment. From these foundations, regression models, that learn from the data in predicting the location of detected emitters, can be tested and validated; some of which are presented here.


## Install

1. Github (using notebooks)
    - git clone
    - cd rfml_localization
    - jupyter lab 
2. Github (using pure python lib)
    - git clone
    - cd rfml_localization/rfml_localization
    - python
        - import rfml_localization
3. pip install
    - **not done yet** ~~`pip install rfml_localization`~~

## How to use

The primary purpose of package is to use available optimization techniques with RF fingerprinting for indoor localization.  The package is written to enable the leveraging of [Sci-Kit Learn](https://scikit-learn.org/).  In addition, it enables the use of GLMnet - (see [Glmnet Vignette](https://glmnet-python.readthedocs.io/en/latest/glmnet_vignette.html)). This package includes a simulation mode, `RFsimulation`, to generate a set of locations and associated synthetic measurements.  For optimization, the focus is on kernilzing data, see `HFF_k_matrix`, and then pairing with regression models ([Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html), [MLPregressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html), etc) within SKLearn or the Glmnet model.  The kernelized measurement parameters are tuned along with chosen model. 

### Manual Optimization
To use, first step is create an `RFchannel instance.

```
import rfml_localization.RFsimulation as RFsim
from sklearn.model_selection import train_test_split

#generate channel scenario using default channel parameters
RFchannel_scenario1 = RFsim.RFchannel()

```

The object's channel parameters are:

```
#print out object variables
for x in RFchannel_scenario1.__dict__:
    print(x,end=', ')
```

    maxSpreadT, PoissonInvLambda, Poissoninvlambda, PoissonGamma, Poissongamma, PathLossN, Xsigma, Wavelength, AoAsigma, 

After setting up the environment, create a sensor setup (Tx and Rx locations) and make 1000 observations of a Tx at random locations.  As methods are run on the object, additional variable are added.  This allows inspection of user-specified parameters and even methods run on object.  They are listed below.  Compare to previous cell.

```
#from channel scenario, generate locations for Tx and Rx and set of measurements
RFchannel_scenario1.generate_RxTxlocations(n_rx=6, n_runs=10000, rxtx_flag=3)
#generate set of measurements
RFchannel_scenario1.generate_Xmodel()
#print out object variables
for x in RFchannel_scenario1.__dict__:
    print(x,end=', ')
```

    maxSpreadT, PoissonInvLambda, Poissoninvlambda, PoissonGamma, Poissongamma, PathLossN, Xsigma, Wavelength, AoAsigma, n_runs, n_rx, areaWL, sensor_locs, rxtx_flag, grid_flag, seed_loc, rxtx_locs, ch_delay_flag, tdoa_flag, seed_tdoa, rxx_delay, ch_gain_flag, drss_flag, seed_rss, rxx_rssi, ch_angle_flag, daoa_flag, seed_aoa, rxx_aoa, seed_Xmodel, X_model, 

Now that a set of measurements are created, split for training and testing.

```
#take object's set of measurements and assign to X,y
X=RFchannel_scenario1.X_model  #measurements/observations
y=RFchannel_scenario1.rxtx_locs[:,0,:].transpose()  #location of  Tx

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

print(" Shapes of training data:\n",X_train.shape,y_train.shape)
```

     Shapes of training data:
     (6700, 27) (6700, 2)


#### SKLearn Regressor

After generating locations, import SKLearn models for regression.  Use the SKLearn-based kernel trick function, `sklearn_kt_regressor`, which wraps a specified SKLearn model and kernelized matrix into a single interface to enable use of SKLearn hyper-tuning tools.  The `sklearn_kt_regressor` inherits all the basic functionality of standard SKLearn model API.  The following sets up model, sets parameters, fits the model, and then predicts.

```
from sklearn.linear_model import Ridge, Lasso
import numpy as np
from sklearn.metrics import mean_squared_error


#based on knowledge of measurement, can derive from class instance rather than manually entering
#variables (shape[1] of rxx_delay, rxx_rss, rxx_aoa, i.e., RFchannel_scenario1.rxx_delay.shape[1])
num_meas_array = np.array([15,6,6]) 
#tuning parameter for each kernel
kernel_s0, kernel_s1, kernel_s2 = np.array([1.13e-06, 2.07e-03, 10])

#set up the model
skl_kt_model = sklearn_kt_regressor(skl_model = Ridge(alpha=.01), skl_kernel = 'rbf', 
                                    n_kernels = 3, kernel_s0 = kernel_s0, kernel_s1 = kernel_s1, 
                                    kernel_s2 = kernel_s2, n_meas_array=num_meas_array)

#set/get model parameters - showing methods inherited from SKLearn
#assign a model parameter
skl_kt_model.set_params(skl_model__alpha = 1.83e-06)
#display model params
print(skl_kt_model.get_params())
```

    {'kernel_s0': 1.13e-06, 'kernel_s1': 0.00207, 'kernel_s2': 10.0, 'n_kernels': 3, 'n_meas_array': array([15,  6,  6]), 'skl_kernel': 'rbf', 'skl_model__alpha': 1.83e-06, 'skl_model__copy_X': True, 'skl_model__fit_intercept': True, 'skl_model__max_iter': None, 'skl_model__normalize': False, 'skl_model__random_state': None, 'skl_model__solver': 'auto', 'skl_model__tol': 0.001, 'skl_model': Ridge(alpha=1.83e-06)}


```
#fit the model
skl_kt_model.fit(X_train,y_train)
```




    sklearn_kt_regressor(kernel_s0=1.13e-06, kernel_s1=0.00207, kernel_s2=10.0,
                         n_kernels=3, n_meas_array=array([15,  6,  6]),
                         skl_kernel='rbf', skl_model=Ridge(alpha=1.83e-06))



```
#predict the model
y_pred = skl_kt_model.predict(X_test)
#error measurement
mse = mean_squared_error(y_test,y_pred)
print('Average error for (x,y) location estimation is {:5.2g} meters'.format(mse))
```

    Average error for (x,y) location estimation is   9.9 meters


#### GLMnet Regressor 
The GLMnet regressor, `glmnet_kt_regressor`, is defined in such a way to follow the SKLearn API -- advantageous in leveraging large body of tools.  Below steps through setting up a model, fitting, and predicting.  Using same data and kernel settings as `skl_kt_regressor` example.

```
#use same training and testing set
num_meas_array = np.array([15,6,6]) 
kernel_s0, kernel_s1, kernel_s2 = np.array([1.13e-06, 2.07e-03, 10])
#glmnet_args={'family': 'mgaussian', 'standardize': False}
glmnet_args=dict(family= 'mgaussian', standardize= False)

#set up the model
glm_kt_model = glmnet_kt_regressor(glm_alpha=0, lambdau=1e-3, skl_kernel='rbf', n_kernels=3,
                 kernel_s0 = kernel_s0, kernel_s1 = kernel_s1, kernel_s2 = kernel_s2,
                 n_meas_array=num_meas_array, glmnet_args=glmnet_args)

#fit the model
glm_kt_model.fit(X_train, y_train)
```




    glmnet_kt_regressor(glm_alpha=0,
                        glmnet_args={'family': 'mgaussian', 'standardize': False},
                        kernel_s0=1.13e-06, kernel_s1=0.00207, kernel_s2=10.0,
                        lambdau=array([0.001]), n_kernels=3,
                        n_meas_array=array([15,  6,  6]), skl_kernel='rbf')



```
#predict the model
y_pred = glm_kt_model.predict(X_test)
#error measurement
mse = mean_squared_error(y_test,y_pred)
print('Average error for (x,y) location estimation is {:5.2g} meters'.format(mse))
```

    Average error for (x,y) location estimation is    23 meters


### Hyperparameter Optimization

Generally speaking, model and kernel parameters need to be tuned. Building on previous example, leverage SKLearn model tools to conduct hyperparameter tuning.

#### SKLearn Regressor
Using SKLearn models.

```
from scipy.stats import loguniform, uniform
from sklearn.model_selection import RandomizedSearchCV

#use loguniform to search uniformaly across orders of magnitude
distributions = {
        'skl_model__alpha': loguniform(1e-7, 1.0e+0),
        'kernel_s0': loguniform(1e-7, 1.0e+1),
        'kernel_s1': loguniform(1e-6, 1.0e+1), 
        'kernel_s2': loguniform(1e-4, 1.0e+2),
        'skl_kernel': ['laplacian', 'rbf'],  # categorical parameter
    }
#create search model from base model
skl_kt_model_search = RandomizedSearchCV(skl_kt_model, distributions,
                                     scoring = 'neg_mean_squared_error', 
                                     cv = 5, n_jobs = 1, n_iter = 100, verbose=1)
#fit search model
search_results = skl_kt_model_search.fit(X_train, y_train)

print(search_results.best_params_)
```

    Fitting 5 folds for each of 100 candidates, totalling 500 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 500 out of 500 | elapsed: 58.1min finished


    {'kernel_s0': 0.0007094452596116463, 'kernel_s1': 0.032187896160191284, 'kernel_s2': 0.2663595286341815, 'skl_kernel': 'laplacian', 'skl_model__alpha': 0.0006763336535533695}


From search, set params for base model and validate against test data

```
#set params based on search
skl_kt_model.set_params(**search_results.best_params_)

#fit model using best params ()
skl_kt_model.fit(X_train, y_train)

#predict and show error
y_pred = skl_kt_model.predict(X_test)
mse=mean_squared_error(y_pred, y_test)
print('Average error for (x,y) location estimation is {:5.2g} meters'.format(mse))

```

    Average error for (x,y) location estimation is   5.1 meters


#### GLMnet Regressor
Using GLMnet model.

```
from scipy.stats import loguniform, uniform
from sklearn.model_selection import RandomizedSearchCV

#use loguniform to search uniformally across orders of magnitude
distributions = {
        'lambdau': loguniform(1e-7, 1.0e+0),
        'kernel_s0': loguniform(1e-7, 1.0e+1),
        'kernel_s1': loguniform(1e-6, 1.0e+1), 
        'kernel_s2': loguniform(1e-4, 1.0e+2),
        'skl_kernel': ['laplacian', 'rbf'],  # categorical parameter
    }
#create search model from base model
#glmnet_model uses single cpu, so increase number of jobs 
glm_kt_model_search = RandomizedSearchCV(glm_kt_model, distributions,
                                     scoring = 'neg_mean_squared_error', 
                                     cv = 5, n_jobs = 6, n_iter = 100, verbose=1)
#fit search model
search_results = glm_kt_model_search.fit(X_train, y_train)

print(search_results.best_params_)
```

    Fitting 5 folds for each of 100 candidates, totalling 500 fits


    [Parallel(n_jobs=6)]: Using backend LokyBackend with 6 concurrent workers.
    [Parallel(n_jobs=6)]: Done  38 tasks      | elapsed:  7.8min
    [Parallel(n_jobs=6)]: Done 188 tasks      | elapsed: 39.3min
    [Parallel(n_jobs=6)]: Done 438 tasks      | elapsed: 102.3min
    [Parallel(n_jobs=6)]: Done 500 out of 500 | elapsed: 115.8min finished


    {'kernel_s0': 0.0003017344124394731, 'kernel_s1': 0.0015381828738127007, 'kernel_s2': 0.13428923755571717, 'lambdau': 1.1024263412692438e-07, 'skl_kernel': 'rbf'}


```
#set params based on search
glm_kt_model.set_params(**search_results.best_params_)

#fit model using best params ()
glm_kt_model.fit(X_train, y_train)

#predict and show error
y_pred = glm_kt_model.predict(X_test)
mse=mean_squared_error(y_pred, y_test)
print('Average error for (x,y) location estimation is {:5.2g} meters'.format(mse))

```

    Average error for (x,y) location estimation is   5.7 meters

