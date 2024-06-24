import math
import random

# Excercise 1: This function calculate f1 score value from three input tp, fp & fn 
def compute_f1_score(tp, fp, fn):
    if type(tp) != int or type(fp) != int or type(fn) != int:
      if type(tp) != int:
          print('tp must be int')
      if type(fp) != int:
          print('fp must be int')
      if type(fn) != int:
          print('fn must be int')
    else: 
      if tp <= 0 or fp <= 0 or fn <= 0:
        print('tp & fp and fn must be greater than zero')
      else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        print(f'Precision is {precision}\nRecall is {recall}\nF1_score is {f1_score}')
   
print(compute_f1_score(2, 3, 4))


# Exercise 2: Write function that simulate according to three activation function

# User input the value of x & activation funciton variables
x = input('Input x =')
activation_function = input('Input activation Function (sigmoid|relu|elu)')

# Define all the functions 

def is_number(x):
    try:
        float(x)    # Type - casting the string to ‘float ‘.
                    # If string is not a valid ‘float 
                    # it ’ll raise ‘ValueError ‘ exception    
    except ValueError:
        return False
    return True  

def cal_sigmoid_function(x):
    return 1 /(1 + math.exp(-x))

def cal_relu(x):
    if x <= 0:
        return 0
    else:
        return x

def cal_elu(x):
    alpha = 0.01
    if  x <= 0:
        return alpha * (math.exp(x) - 1)
    else:
        return x

def simulate_activation_func(x, activation_function):
    if not is_number(x):
      print('x must be a number')
      return
    x = float(x)
    
    if activation_function == 'sigmoid':
        return cal_sigmoid_function(x)
    elif activation_function == 'relu':
        return cal_relu(x)
    elif activation_function == 'elu':
        return cal_elu(x)
    elif  activation_function == activation_function: 
        print(f'{activation_function} not support')
 

simulate_activation_func(x, activation_function)


# Excersice 3: Write a function to calculate loss regression

# Function to calculate MAE (Mean Absolute Error)
def cal_mae(predict,target):
    return abs(predict - target)
   

# Function to calculate SE (Square Error)
def cal_se(predict,target):
    return (predict - target) ** 2
    
# Function to calculate different regression loss functions 
def calculate_loss(num_samples, loss_name):
    if not num_samples.isnumeric():
        print('numer of samples must be an integer number')
        return

    num_samples = int(num_samples)

    if loss_name not in ['MAE', 'MSE', 'RMSE']:
        print(f'{loss_name} not support')
        return

    total_loss = 0

    for i in range(num_samples):
        predict = random.uniform(0,10)
        target = random.uniform(0,10)
    
        if loss_name == 'MAE':
            total_loss += cal_mae(predict, target)
        elif loss_name == 'MSE' or loss_name == 'RMSE':
            total_loss += cal_se(predict, target)

        print(f'loss name:{loss_name}, sample:{num_samples}, pred:{predict}, target:{target}, loss: {total_loss}')

    final_loss = total_loss / num_samples

    if loss_name == 'RMSE':
        final_loss = math.sqrt(final_loss)

    print(f'final {loss_name} : {final_loss}')

#Test Case
num_samples = input('Input number of samples ')
loss_name = input('Input loss name (MAE|MSE|RMSE): ')
calculate_loss(num_samples, loss_name)


# Exercise 4: Write 4 functions to estimate value of sin, cos, sinh & cosh functions

def factorial(n):
    if n == 0:
        return 1
    else:
        result = 1
        for i in range(1,n+1):
            result = result * i
        return result

def approx_sin(x, n):
    result = 0
    for i in range (n):
        result += (-1)**i * x**(2*i+1) / factorial(2*i+1)
    return result

def approx_cos(x, n):
    result = 0
    for i in range (n):
        result += (-1)**i * (x**(2*i) / factorial(2*i))
    return result

def approx_sinh(x, n):
    result = 0
    for i in range (n):
        result += x**(2*i+1) / factorial(2*i+1)
    return result

def approx_cosh(x, n):
    result = 0
    for i in range (n):
        result += x**(2*i) / factorial(2*i)
    return result

#Test Scenario
x, n = 3.14, 10
print(f'approx_sin: {approx_sin(x, n)}')
print(f'approx_cos: {approx_cos(x, n)}')
print(f'approx_sinh: {approx_sinh(x, n)}')
print(f'approx_cosh: {approx_cosh(x, n)}')


# Exercise 5: Write a function to calculate MD_nRE (Mean Difference of nth Root Error) 
def cal_MD_nRE(y, y_hat, n, p):
    result = (y** (1/n) - y_hat** (1/n))**p
    rounded_result = round(result, 3)
    print(f'Md_nRE value= {rounded_result}')
    return

# Test Cases
cal_MD_nRE(100,99.5,2,1)
cal_MD_nRE(50,49.5,2,1)
cal_MD_nRE(20,19.5,2,1)
cal_MD_nRE(0.6,0.1,2,1)