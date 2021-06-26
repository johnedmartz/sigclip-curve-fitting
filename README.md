# Curve fitting with sigma clipping

This package can fit any kind of model to 2-dimensional data using sigma-clipping to remove outliers.


## Installation

### 1 - Installation using from PyPi using pip

    pip install sigmclipfit

### 2 - Cloning the repository and installing

    git clone https://github.com/johnedmartz/sigclip-curve-fitting.git
    pip install .


## Using the package

* Import the package

        from sigmclipfit.sgmfit import sgmfit

* Set up the model with data and parameters
        
        fit = sgmfit(x_data=x, y_data=y, model_function=function, ivalues=initial_values, n_sigm=n_sigma)

* Make the fit with sigma clipping

        fit.fit_model()          

                By default:                fit.fit_model(keep_i_values = False, max_iter=np.inf)

                it can be changed like:    fit.fit_model(keep_i_values = True, max_iter=50)

* Draw the plot with the original data and data used for the fit
        
        fit.fit_plot()

* Display results
        
        fit.result