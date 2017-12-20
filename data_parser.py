import numpy as np
from sklearn import preprocessing

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
    X = trans_education_marriage(X)

    testX = []
    with open('Test_Public.csv', 'r') as f:
        f.readline()
        for line in f:
            line = line.strip('\n').split(',')
            line = list(map(int,line))
            testX.append(line)
    testX = np.array(testX)
    testX = np.delete(testX, 0, 1)
    testX = trans_education_marriage(testX)
    return X, Y, testX

def trans_education_marriage(X):    
    # transfer EDUCATION, MARRIAGE to binary
    X = X.transpose()
    X = list(X)
    # print X[2]
    # print X[3]
    edu_1 = []
    edu_2 = []
    edu_3 = []
    for edu in X[2]:
        if edu == 1: edu_1.append(1)
        else: edu_1.append(0)

        if edu == 2: edu_2.append(1)
        else: edu_2.append(0)

        if edu == 3: edu_3.append(1)
        else: edu_3.append(0)

    mar_1 = []
    mar_2 = []
    mar_3 = []
    for mar in X[3]:
        if mar == 1: mar_1.append(1)
        else: mar_1.append(0)

        if mar == 2: mar_2.append(1)
        else: mar_2.append(0)

        if mar == 3: mar_3.append(1)
        else: mar_3.append(0)

    X.append(edu_1)
    X.append(edu_2)
    X.append(edu_3)
    X.append(mar_1)
    X.append(mar_2)
    X.append(mar_3)
    X.pop(3)
    X.pop(2)

    X = np.array(X).transpose()
    # print(X[0])
    return X

if __name__ == '__main__':
    X, Y, testX = parse_data()
    Y = Y.astype('int32')
    Y = np.ravel(Y)

    print ('sizeX:', X.shape)
    print ('sizeY:', Y.shape)
    print ('sizeTestX:', testX.shape)


    from lightfm import LightFM
    from lightfm.evaluation import precision_at_k

    # Load the MovieLens 100k dataset. Only five
    # star ratings are treated as positive.
    data = fetch_movielens(min_rating=5.0)

    # Instantiate and train the model
    model = LightFM(loss='warp')
    model.fit(data['train'], epochs=30, num_threads=2)

    # Evaluate the trained model
    test_precision = precision_at_k(model, data['test'], k=5).mean()


    # from xgboost import XGBClassifier
    # model = XGBClassifier()
    # model.fit(X, Y)
    # y_ = model.predict(X[:20])
    # print y_
    
    # print('id,label')
    # for i,v in enumerate(y_):
    #     print('%d,%d'%(i+1,v))
