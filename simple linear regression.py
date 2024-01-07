import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

class Linear_Regression:
    def __init__(self, x, y, w_0, b_0, alpha, num_iters):
        self.x = x
        self.y = y
        self.w_0 = w_0
        self.b_0 = b_0
        self.alpha = alpha
        self.num_iters = num_iters
        
    def plot(self):
        fig, ax = plt.subplots(2,2)
        ax[0,0].bar(self.x, self.y, width=1, edgecolor="white", linewidth=0.7, color = 'green')
        ax[0,0].set_title('Bar plot')
    
        ax[0,1].scatter(self.x, self.y, marker = 'x', color = 'red')
        ax[0,1].set_title('Scatter plot')
    
        ax[1,0].stem(self.x, self.y)
        ax[1,0].set_title('Stem plot')
        plt.show()
    
    # J = (1/2m)*sum(total_cost)
    # cost = f - y
    def compute_cost(self, x, y, w, b):
        m = len(x)
        total_cost = 0
    
        for i in range(m):
            f = w*x[i] + b
            cost = (f - y[i])**2
            total_cost += cost
        total_cost = total_cost/(2*m)
        return total_cost

    def compute_gradient(self, x, y, w, b):
        m = len(x)
        dj_db = 0
        dj_dw = 0
        
        for i in range(m):
            f = w*x[i] + b
            dj_dw += (f - y[i])*x[i]
            dj_db += (f - y[i])
        dj_dw = dj_dw/m
        dj_db = dj_db/m
        
        return dj_dw, dj_db

    def gradient_descent(self, x, y, w_in , b_in, alpha, num_iters):
        m = len(x)
        J_history = []
        w_history = []
        
        w = copy.deepcopy(w_in)
        b = b_in
        
        for i in range(num_iters):
            dj_dw, dj_db = self.compute_gradient(x, y, w, b)
            
            w = w - alpha*dj_dw
            b = b - alpha*dj_db
            
            cost = self.compute_cost(x, y, w, b)
            J_history.append(cost)
            
            if i%100  == 0:
                w_history.append(w)
                print(f"Interation {i}: Cost {float(J_history[-1]):.2f}")
                
        return w, b, J_history, w_history

    def prediction(self):
        w, b,_,_ = self.gradient_descent(self.x, self.y, self.w_0 , self.b_0, self.alpha, self.num_iters)
        print("w,b found by gradient descent:", w, b)
        m = len(self.x)
        predicted = np.zeros(m)
        for i in range(m):
            predicted[i] = w * x_train[i] + b
        plt.plot(x_train, predicted, c = "b")

        plt.scatter(x_train, y_train, marker='x', c='r')
        plt.title("Experience vs Salary")
        plt.ylabel('Salary in thousands')
        plt.xlabel('Experience in months')
        new_x = [60, 65, 70, 75, 80, 85]
        new_y = []
        for i in range(len(new_x)):
            new_y.append(w*new_x[i]+b)
        plt.scatter(new_x, new_y, marker = 'o', c = 'green')
        plt.show()

df = pd.read_csv('Experience-Salary.csv')
x_train = df['exp(in months)']
y_train = df['salary(in thousands)']
linear = Linear_Regression(x_train, y_train, 0, 0, 0.0005, 500)
linear.plot()
linear.prediction()




# w_0 = 0.0
# b_0 = 0.0

# for i in range(m):
#     predicted[i] = w * x_train[i] + b

# plt.plot(x_train, predicted, c = "b")

# # Create a scatter plot of the data. 
# plt.scatter(x_train, y_train, marker='x', c='r') 

# # Set the title
# plt.title("Experience vs Salary")
# # Set the y-axis label
# plt.ylabel('Salary in thousands')
# # Set the x-axis label
# plt.xlabel('Experience in months')
# new_x = [60, 65, 70, 75, 80, 85]
# new_y = []
# for i in range(len(new_x)):
#     new_y.append(w*new_x[i]+b)
# plt.scatter(new_x, new_y, marker = 'o', c = 'green')
# plt.show()




# plot(x_train, y_train)


