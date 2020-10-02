import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle

def main():
    train_model()
    predictUsingModel()
    
def train_model():
    x = np.array([4.3,1.3,0.5,1,1.5,2,2.5,3,2.8,3.5, 4, 4.5 ,5 ,1.2 ,3.3, 5.3 ,0.8, 2.3]).reshape((-1, 1))
    y = np.array([85,40,20,30,45,45,65,73,52,80,75,82, 90 ,25 , 50 , 95 ,35,45])

    model = LinearRegression()
    model.fit(x,y)

    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)
    print("Linear Regression Equation: Y = ",model.intercept_," + ",model.coef_[0], "X",sep='')

    y_pred = model.predict(x)
    print('predicted response:', y_pred, sep='\n')
    
    filename = 'lrm_model.sav'
    pickle.dump(model, open(filename, 'wb'))

def predictUsingModel():
    x = np.array([4.3 , 1.3 , 0.5, 1, 1.5 , 2 , 2.5 , 3 , 2.8,  3.5 , 4 , 4.5 ,5 ,1.2 ,3.3 , 5.3 , 0.8, 2.3]).reshape((-1, 1))
    y = np.array([85 , 40  , 20 , 30 , 60 , 45 , 65, 73 , 52,  80 , 75 , 82 , 90 ,25 , 50 , 95 ,35,45])

    model = pickle.load(open("lrm_model.sav", 'rb'))
    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)
    print("Linear Regression Equation: Y = ",model.intercept_," + ",model.coef_[0], "X",sep='')

    y_pred = model.predict(x)
    print('predicted response:', y_pred, sep='\n')
    
    plt.scatter(x,y,color='black')
    plt.plot(x,y_pred,color='red',linewidth=3)
    plt.title('Hasil Latihan')
    plt.xlabel('Jumlah Jam Latihan Ngoding')
    plt.ylabel('Presentase Nilai')
    plt.show()

main()