# rfml_localization
> Focus of repository is the location of uncooperative wireless emitters in an indoor environment.  The challenge is that an indoor environment results in dynamic, random propagation of wireless signals.  An approach is to create a RF fingerprint or map of the environment using different RF measurements techniques. Relevant measurements techniques generate a range of data from IQ samples to correlation curves to time-delay estimates (TDE's) for  TDOA; multiple power readings for RSS; and multiple angle of arrival readings for AoA.  The theoretical performance bounds are generated, via CRLB, of empirically measured and derived channel limitations that exists in literature.  From same literature, a simulation model is created that generates the specified random RF channel environment. From these foundations, regression models, that learn from the data in predicting the location of detected emitters, can be tested and validated; some of which are presented here.


## Install

rfml_localization requires
- numpy
- sklearn
- matplotlib
- glmnet_py
    - libgfortran3

You can install using the following methods

From source - github notebooks

    git clone
    cd rfml_localization
    jupyter lab 
From source - github python lib

    git clone
    cd rfml_localization/rfml_localization
    python
    import rfml_localization.core as rfcore
    import rfml_localization.RFsimulation as rfsim
Eventually, pip

    ~~pip install rfml_localization~~

## How to use

The primary purpose of package is to use available optimization techniques with RF fingerprinting for indoor localization. The package is written to enable the leveraging of [Sci-Kit Learn](https://scikit-learn.org/), so assumes some familiarity with said package.  In addition, it enables the use of GLMnet - (see [Glmnet Vignette](https://glmnet-python.readthedocs.io/en/latest/glmnet_vignette.html)). This package includes a simulation mode, `RFsimulation`, to generate a set of locations and associated synthetic measurements.  For optimization, the focus is on kernilzing data, see `HFF_k_matrix`, and then pairing with regression models ([Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html), [MLPregressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html), etc) within SKLearn or the Glmnet model.  The kernelized measurement parameters are tuned along with chosen model. 

### Manual Optimization

#### Create Channel
To showcase use, first step is create an `RFchannel` instance from the `RFsimulation` which provides methods to generate synthetic data for testing. For background on parameters, see Saleh [^1], Rappaport [^2], and Spencer [^3]. 

```
import rfml_localization.RFsimulation as rfsim
from sklearn.model_selection import train_test_split

#generate channel using default channel parameters
RFchannel_scenario1 = rfsim.RFchannel()
#print out parameters
print("Channel parameters of created object:",vars(RFchannel_scenario1))
```

    Channel parameters of created object: {'maxSpreadT': 100, 'PoissonInvLambda': 50, 'Poissoninvlambda': 10, 'PoissonGamma': 50, 'Poissongamma': 29, 'PathLossN': 3.0, 'Xsigma': 7, 'Wavelength': 1.4615111050485465, 'AoAsigma': 0.4537856055185257}


#### Create Sensor Setup/Generate Synthetic Data
After setting up the environment, create a sensor setup (here, fixed Rx locations) and Txmtr randomly placed at 1000 locations.  For each Txmtr location, an observation is created that contains TDoA, RSS, and AoA measurements -- see `RFsimulation.generate_Xmodel`. As methods are run on the object, additional variable are added.  This allows inspection of user-specified parameters and even methods run on the object.

The generated measurements and associated locations are pulled out and then split into test, training sets.

```
#from channel scenario, generate locations for Tx and Rx and set of measurements
RFchannel_scenario1.generate_RxTxlocations(n_rx=6, n_runs=1000, rxtx_flag=3)
#generate set of measurements
RFchannel_scenario1.generate_Xmodel()

#take object's set of measurements, locations and assign to X,y
X=RFchannel_scenario1.X_model  #measurements/observations
y=RFchannel_scenario1.rxtx_locs[:,0,:].transpose()  #location of  Tx

#split for training, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
```

#### Test Models

##### Using sklearn_kt_regressor
After generating locations, import SKLearn models for regression.  Ridge and LASSO models were imported but any Sci-Kit Learn model should work. These models along with kernel parameters are passed to the Sci-Kit Learn-based kernel trick object, `sklearn_kt_regressor`.  `sklearn_kt_regressor` wraps the specified model and kernelized matrix into a single interface to enable use of SKLearn hyper-tuning tools.  It inherits all the basic functionality of standard Sci-Kit Learn API.  The following sets up model, sets parameters, fits the model, and then predicts.  Note that no tuning of the model's parameters has been conducted so final results are not optimized.

```
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

#based on knowledge of measurement, can derive from class instance rather than manually entering
#variables (shape[1] of rxx_delay, rxx_rss, rxx_aoa, i.e., RFchannel_scenario1.rxx_delay.shape[1])
num_meas_array = np.array([15,6,6]) 

#tuning parameter for each measurement kernel (TDoA, RSS, AoA)
kernel_s0, kernel_s1, kernel_s2 = np.array([1.13e-06, 2.07e-03, 10])

#set up the model
skl_kt_model = sklearn_kt_regressor(skl_model = Ridge(), skl_kernel = 'rbf', 
                                    n_kernels = 3, kernel_s0 = kernel_s0, kernel_s1 = kernel_s1, 
                                    kernel_s2 = kernel_s2, n_meas_array=num_meas_array)

#set model parameters - showing methods inherited from SKLearn
skl_kt_model.set_params(skl_model__alpha = 1.83e-06)

#fit the model
skl_kt_model.fit(X_train,y_train)

#predict the model
y_pred = skl_kt_model.predict(X_test)

#error measurements
mse = mean_squared_error(y_test,y_pred)
msec = mse_EucDistance(y_test,y_pred)
print('-----------------------------------------------------------------------------------------------')
print('Mean summed/mean physical distance error for (x,y) location estimation: {:3.1f} / {:3.1f}: meters'.format(mse,msec))
print('-----------------------------------------------------------------------------------------------')

```

    -----------------------------------------------------------------------------------------------
    Mean summed/mean physical distance error for (x,y) location estimation: 10.9 / 4.0: meters
    -----------------------------------------------------------------------------------------------


##### Using glmnet_kt_regressor 
Similarly, the GLMnet regressor, `glmnet_kt_regressor`, is defined in such a way to follow the Sci-Kit Learn API -- advantageous in leveraging large body of tools.  Below steps through setting up a model, fitting, and predicting.  Using same data and kernel settings as `skl_kt_regressor` example.

```
#use same training and testing set
num_meas_array = np.array([15,6,6]) 
kernel_s0, kernel_s1, kernel_s2 = np.array([1.13e-06, 2.07e-03, 10])

#set up the model, passing glmnet specific parameters via dictionary
glmnet_args=dict(family= 'mgaussian', standardize= False)

glm_kt_model = glmnet_kt_regressor(glm_alpha=0, lambdau=1e-3, skl_kernel='rbf', n_kernels=3,
                 kernel_s0 = kernel_s0, kernel_s1 = kernel_s1, kernel_s2 = kernel_s2,
                 n_meas_array=num_meas_array, glmnet_args=glmnet_args)

#fit the model
glm_kt_model.fit(X_train, y_train)

#predict the model
y_pred = glm_kt_model.predict(X_test)

#error measurements
mse = mean_squared_error(y_test,y_pred)
msec = mse_EucDistance(y_test,y_pred)
print('-----------------------------------------------------------------------------------------------')
print('Mean summed/mean physical distance error for (x,y) location estimation: {:3.1f} / {:3.1f}: meters'.format(mse,msec))
print('-----------------------------------------------------------------------------------------------')

```

    -----------------------------------------------------------------------------------------------
    Mean summed/mean physical distance error for (x,y) location estimation: 23.9 / 5.9: meters
    -----------------------------------------------------------------------------------------------


### Hyperparameter Optimization

Generally speaking, model and kernel parameters need to be tuned. Building on previous example, leverage SKLearn model tools to conduct hyperparameter tuning.

#### Using sklear_kt_regressor
Using sklearn models.  For this example, for processing speed, n_iter is kept low.  For better performance, increase to 5k.  Note the improved performance.  Here, [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) is used for hyperparameter though any sklearn technique is viable.

```
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
pd.options.display.float_format = '{:.2e}'.format
pd.options.display.width = 120

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
                                     cv = 5, n_jobs = 1, n_iter = 10, verbose=1)
#fit search model
skl_search_results = skl_kt_model_search.fit(X_train, y_train)

#set params based on search
skl_kt_model.set_params(**skl_search_results.best_params_)

#fit model using best params ()
skl_kt_model.fit(X_train, y_train)

#predict and show error
y_pred = skl_kt_model.predict(X_test)
msec = mse_EucDistance(y_test,y_pred)
skl_search_results_pd = pd.DataFrame(skl_search_results.cv_results_)

print('-----------------------------------------------------------------------------------------------')
print('Mean summed/mean physical distance error for (x,y) location estimation: {:3.1f} / {:3.1f}: meters'.format(mse,msec))
print("Optimized parameters in rank order (based on tested values):\n")
print(skl_search_results_pd.sort_values(by='rank_test_score').filter(items=['param_kernel_s0', 'param_kernel_s1','param_kernel_s2','param_lambdau','param_skl_kernel','mean_test_score']).head())
print('-----------------------------------------------------------------------------------------------')
```

    Fitting 5 folds for each of 10 candidates, totalling 50 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done  50 out of  50 | elapsed:    6.1s finished


    -----------------------------------------------------------------------------------------------
    Mean summed/mean physical distance error for (x,y) location estimation: 23.9 / 3.5: meters
    Optimized parameters in rank order (based on tested values):
    
      param_kernel_s0 param_kernel_s1 param_kernel_s2 param_skl_kernel  mean_test_score
    7        3.39e-06        2.71e-05        1.18e-01              rbf        -8.11e+00
    9        4.31e-06        7.34e-01        5.23e-03              rbf        -1.14e+01
    1        4.06e-07        1.71e-03        2.10e-03        laplacian        -1.91e+01
    3        1.67e+00        5.37e-03        7.96e-01              rbf        -1.94e+01
    2        1.39e-03        9.21e-04        3.03e-01              rbf        -2.11e+01
    -----------------------------------------------------------------------------------------------


#### Using glmnet_kt_regressor
Using GLMnet model. For this example, for processing speed, n_iter is kept low.  For better performance, increase to 5k.

```
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
                                     cv = 5, n_jobs = 6, n_iter = 10, verbose=1)
#fit search model
glm_search_results = glm_kt_model_search.fit(X_train, y_train)

#set params based on search
glm_kt_model.set_params(**glm_search_results.best_params_)

#fit model using best params ()
glm_kt_model.fit(X_train, y_train)

#predict and show error
y_pred = glm_kt_model.predict(X_test)
msec = mse_EucDistance(y_test,y_pred)
glm_search_results_pd = pd.DataFrame(glm_search_results.cv_results_)

print('-----------------------------------------------------------------------------------------------')
print('Mean summed/mean physical distance error for (x,y) location estimation: {:3.1f} / {:3.1f}: meters'.format(mse,msec))
print("Optimized parameters in rank order (based on tested values):")
print(glm_search_results_pd.sort_values(by='rank_test_score').filter(items=['param_kernel_s0', 'param_kernel_s1','param_kernel_s2','param_lambdau','param_skl_kernel','mean_test_score']).head())
print('-----------------------------------------------------------------------------------------------')

```

    Fitting 5 folds for each of 10 candidates, totalling 50 fits


    [Parallel(n_jobs=6)]: Using backend LokyBackend with 6 concurrent workers.
    [Parallel(n_jobs=6)]: Done  38 tasks      | elapsed:    5.3s
    [Parallel(n_jobs=6)]: Done  50 out of  50 | elapsed:    9.5s finished


    -----------------------------------------------------------------------------------------------
    Mean summed/mean physical distance error for (x,y) location estimation: 23.9 / 3.2: meters
    Optimized parameters in rank order (based on tested values):
      param_kernel_s0 param_kernel_s1 param_kernel_s2 param_lambdau param_skl_kernel  mean_test_score
    7        2.56e-03        1.24e-02        2.73e-02      3.26e-07        laplacian        -6.95e+00
    0        4.83e-04        2.50e-06        1.13e-02      5.07e-04              rbf        -1.92e+01
    1        4.94e-06        7.78e-05        2.06e+00      1.11e-03              rbf        -2.44e+01
    9        2.64e-05        3.72e-01        1.32e-03      2.87e-03              rbf        -2.46e+01
    2        1.13e-07        1.84e-02        8.37e-03      1.73e-04        laplacian        -2.78e+01
    -----------------------------------------------------------------------------------------------


## References

[^1]: A. A. M. Saleh and R. Valenzuela, "A Statistical Model for Indoor Multipath Propagation," in IEEE Journal on Selected Areas in Communications, vol. 5, no. 2, pp. 128-137, February 1987, doi: 10.1109/JSAC.1987.1146527.

[^2]: Theodore S. Rappaport's Wireless Communications: Principles and Practice by IEEE Press, Inc. Prentic Hall ISBN: 0-7803-1167-1. Chapters 3 and 4

[^3]: Q. H. Spencer, B. D. Jeffs, M. A. Jensen and A. L. Swindlehurst, "Modeling the statistical time and angle of arrival characteristics of an indoor multipath channel," in IEEE Journal on Selected Areas in Communications, vol. 18, no. 3, pp. 347-360, March 2000, doi: 10.1109/49.840194.
