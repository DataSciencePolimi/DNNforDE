# DNNforDE

The notebook "general_ODE_1d_trialf.ipynb" is the first implementation to solve Ordinary differential equations, 1 dimension, using a trial function (to satisfy boundary conditions) and no labelled data.
It uses as tests the ODE implemented in the file "functions1d.py", with the real solution known.

To implement a new ODE to solve:
* write a function that takes as input the variable ( t_i ), the function ( x(t_i) ) and its derivatives ( d^k x / d t^k ) in an array (e.g the element [2][0] will be the 3rd derivative with respect to the first variable) and returns the differential equation in a way that the result must be 0 (e.g. if the equation is x'(t) = x(t), the function must return x'(t_i)-x(t_i) )
* write the degree: the number of derivatives that has to be evaluated for the differential equation (e.g. before, degree=1, just the 1st derivative is needed)
* write a function that takes as input the neural network ( N(t_i) ) and the variable ( t_i ) and returns the trial function
* write the real solution of the ODE
* write the interval in which it needs to be evaluated
* Add everything to the dictionaries
