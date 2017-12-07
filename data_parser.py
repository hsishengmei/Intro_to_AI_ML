from xgboost import XGBClassifier
import numpy as np

def parse_data():
    col_name = []
    X = []
    with open('Train.csv', 'r') as f:
        col_name = f.readline()
        col_name = col_name.strip('\n').split(',')
        for line in f:
            line = line.strip('\n').split(',')
            line = list(map(int,line))
            X.append(line)
    X = np.array(X)
    Y = X[:, -1]
    X = np.delete(X, -1, 1)
    X = np.delete(X, 0, 1)

    testX = []
    with open('Test_Public.csv', 'r') as f:
        f.readline()
        for line in f:
            line = line.strip('\n').split(',')
            line = list(map(int,line))
            testX.append(line)
    testX = np.array(testX)
    testX = np.delete(testX, 0, 1)

    return X, Y, testX
if __name__ == '__main__':
    X, Y, testX = parse_data()
    Y = Y.astype('int32')
    Y = np.ravel(Y)

    print ('sizeX:', X.shape)
    print ('sizeY:', Y.shape)
    print ('sizeTestX:', testX.shape)

    model = XGBClassifier()
    model.fit(X, Y)
    # y_ = model.predict(testX)
    
    # print('id,label')
    # for i,v in enumerate(y_):
    #     print('%d,%d'%(i+1,v))
