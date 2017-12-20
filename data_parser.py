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
    X = trans_pay(X)

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
    X = X.tolist()
    print X[2]
    print X[3]
    edu_0 = []
    edu_1 = []
    edu_2 = []
    edu_3 = []
    edu_4 = []
    edu_5 = []
    edu_6 = []
    for edu in X[2]:
        if edu == 0: 
            edu_0.append(1)
            edu_1.append(0)
            edu_2.append(0)
            edu_3.append(0)
            edu_4.append(0)
            edu_5.append(0)
            edu_6.append(0)
        elif edu == 1: 
            edu_0.append(0)
            edu_1.append(1)
            edu_2.append(0)
            edu_3.append(0)
            edu_4.append(0)
            edu_5.append(0)
            edu_6.append(0)
        elif edu == 2: 
            edu_0.append(0)
            edu_1.append(0)
            edu_2.append(1)
            edu_3.append(0)
            edu_4.append(0)
            edu_5.append(0)
            edu_6.append(0)
        elif edu == 3: 
            edu_0.append(0)
            edu_1.append(0)
            edu_2.append(0)
            edu_3.append(1)
            edu_4.append(0)
            edu_5.append(0)
            edu_6.append(0)        
        elif edu == 4: 
            edu_0.append(0)
            edu_1.append(0)
            edu_2.append(0)
            edu_3.append(0)
            edu_4.append(1)
            edu_5.append(0)
            edu_6.append(0)
        elif edu == 5: 
            edu_0.append(0)
            edu_1.append(0)
            edu_2.append(0)
            edu_3.append(0)
            edu_4.append(0)
            edu_5.append(1)
            edu_6.append(0)
        else: 
            edu_0.append(0)
            edu_1.append(0)
            edu_2.append(0)
            edu_3.append(0)
            edu_4.append(0)
            edu_5.append(0)
            edu_6.append(1)

    mar_0 = []
    mar_1 = []
    mar_2 = []
    mar_3 = []
    for mar in X[3]:
        if mar == 0: 
            mar_0.append(1)
            mar_1.append(0)
            mar_2.append(0)
            mar_3.append(0)
        if mar == 1: 
            mar_0.append(0)
            mar_1.append(1)
            mar_2.append(0)
            mar_3.append(0)
        if mar == 2: 
            mar_0.append(0)
            mar_1.append(0)
            mar_2.append(1)
            mar_3.append(0)
        else: 
            mar_0.append(0)
            mar_1.append(0)
            mar_2.append(0)
            mar_3.append(1)

    X.append(edu_1)
    X.append(edu_2)
    X.append(edu_3)
    X.append(edu_4)
    X.append(edu_5)
    X.append(edu_6)
    X.append(mar_0)
    X.append(mar_1)
    X.append(mar_2)
    X.append(mar_3)
    X.pop(3)
    X.pop(2)

    # print np.array(X).shape
    # print np.array(X).transpose()[0]
    # print X[-10:]
    # print len(edu_6)
    # print len(edu_3)
    # for i in X:
    #     print type(i)
    #     print i[:10]
    # print np.array(X).shape
    # raw_input()
    X = np.array(X).transpose()
    print(X[0])
    return X


def trans_pay(X):    
    X = X.transpose()
    X = X.tolist()
    for pay_x in X[5:11]:
        pay_2 = []
        pay_1 = []
        pay_0 = []
        pay_continuous = []

        for pay in pay_x:
            if pay == -2:
                pay_2.append(1)
                pay_1.append(0)
                pay_0.append(0)
                pay_continuous.append(0)
            elif pay == -1:
                pay_2.append(0)
                pay_1.append(1)
                pay_0.append(0)
                pay_continuous.append(0)
            elif pay == 0:
                pay_2.append(0)
                pay_1.append(0)
                pay_0.append(1)
                pay_continuous.append(0)
            else:
                pay_2.append(0)
                pay_1.append(0)
                pay_0.append(0)
                pay_continuous.append(pay)

            X.append(list(pay_2))
            X.append(list(pay_1))
            X.append(list(pay_0))
            X.append(list(pay_continuous))

    X.pop(10)
    X.pop(9)
    X.pop(8)
    X.pop(7)
    X.pop(6)
    X.pop(5)

    X = np.array(X).transpose()
    print(X[0])
    return X

if __name__ == '__main__':
    X, Y, testX = parse_data()
    # Y = Y.astype('int32')
    # Y = np.ravel(Y)
    print X
    print ' '
    print Y

    print ('sizeX:', X.shape)
    print ('sizeY:', Y.shape)
    print ('sizeTestX:', testX.shape)



    # from xgboost import XGBClassifier
    # model = XGBClassifier()
    # model.fit(X, Y)
    # y_ = model.predict(X[:20])
    # print y_


    import xgboost as xgb
    xgdmat=xgb.DMatrix(X,Y)
    # our_params={'eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'objective':'reg:linear','max_depth':3,'min_child_weight':1}
    our_params = {"objective": "reg:linear", "booster":"gblinear"}
    final_gb=xgb.train(our_params,xgdmat)
    tesdmat=xgb.DMatrix(X[:1000])
    y_pred=final_gb.predict(tesdmat)
    # print(y_pred)


    rank_list = [] 
    for i,j in enumerate(y_pred):
        rank_list.append((j,i))

    rank_list.sort()
    rank_list.reverse()
    # print len(rank_list)

    # evaluate result
    actual = Y[:1000]
    print actual
    avg_precision = 0
    a = []
    for i in rank_list[:100]:
        print y_pred[i[1]], actual[i[1]]
        a.append(actual[i[1]])
    # print a
    for i in range(100):
        avg_precision += 1.0*sum(a[:(i+1)])/(i+1)
    avg_precision = avg_precision/100
    print 'avg_precision:', avg_precision


    # print('id,label')
    # for i,v in enumerate(y_):
    #     print('%d,%d'%(i+1,v))
