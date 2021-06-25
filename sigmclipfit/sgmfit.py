import inspect
import matplotlib.pyplot as plt 
import numpy as np
from lmfit import Model



class sgmfit():
    """
    asd
    asd
    """

    def __init__(self, x_data, y_data, model_function, i_values, n_sigm):

        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)

        self.function = model_function
        self.model = Model(model_function)
        self.i_values = i_values

        self.nsigm = n_sigm
        

    def fit_model(self):
        """
        asd
        asd
        """

        x = self.x_data
        y = self.y_data

        # This part obtains the names of the coefficients
        args = inspect.getfullargspec(self.function).args
        args.remove('x')

        # Setting the initial values
        i_params = self.model.make_params()
        iv_dict = {}
        if len(args) == len(self.i_values):
            for i in range(0,len(args)):
                iv_dict[args[i]] = self.i_values[i]
            for p in i_params:
                i_params[p].value = iv_dict[p]
        else:
            raise ValueError("Length of coefficients and initial values differ")


        #Values to force the first iteration
        len_valid = -1 
        len_y = len(y)
        n_cycles = 0


        # This loop keeps going until all the points used for the fit are inside the threshold
        while len_valid < len_y:

            # Fits the data with the given model
            try:
                result = self.model.fit(y, i_params, x=x)
            except:
                raise ValueError("Can't fit curve with current parameters (Valid points = {})".format(len_valid))

            # Saves the parameters found in the fit and sets them as the new initial parameters for the next iteration
            params = self.model.make_params()     
            for p in params:
                params[p].value = result.params[p].value
                i_params[p].value = result.params[p].value

            # Subtract the original values from the new ones
            delt = y - self.model.eval(params, x=x)
            
            # The threshold is the nsigm given times the standard deviation of delt
            threshold = self.nsigm * np.std(delt) 

            # Index of the points inside the threshold
            valid = abs(delt) < threshold
            
                
            len_y = len(y)                          # Number of points in this iteration
            len_valid = len(valid[valid==True])     # Number of points inside threshold

            
            # New x and y used for the next iteration
            x = x[valid]
            y = y[valid]

            n_cycles += 1 

        self.n_cycles = n_cycles
        self.x_fit = x
        self.y_fit = y
        self.result = result
        self.params = params

    
    def fit_plot(self, n=100):
        """
        asd
        asd
        """

        x_curve = np.linspace(min(self.x_data), max(self.x_data), num=n)
        y_curve = self.model.eval(self.params, x=x_curve)


        plt.plot(self.x_data, self.y_data, '.', label='Original sample', markersize=4)
        plt.plot(self.x_fit, self.y_fit, '.', label='Final sample used for the fit')
        plt.plot(x_curve, y_curve)
        plt.legend()
        plt.show()