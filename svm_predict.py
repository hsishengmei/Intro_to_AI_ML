from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from data_parser import parse_data
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn import linear_model

if __name__ == '__main__':
    X, Y, testX = parse_data()
    print 'normalize data'
    scaler = StandardScaler()
    scaler.fit(X)
    # print(scaler.mean_)
    X = scaler.transform(X)    
    testX = scaler.transform(testX)

    kernel_list = ['rbf']
    gamma_list = [0.01]
    print 'calling svm.SVR'
    for kernel in kernel_list:
        if kernel == 'linear': gamma_list = [0.01]
        for gamma in gamma_list:
            print 'kernel:', kernel, 'gamma:', gamma 
            model = svm.SVR(kernel=kernel, C=1, gamma = gamma)
            # model = svm.SVR(kernel='linear', C=1)
            # model = linear_model.LoisticRegression(C=1e5)

            anova_filter = SelectKBest(f_regression, k=10)
            # # anova

            anova_svm = make_pipeline(anova_filter, model)
            anova_svm = model

            print 'cross validation'
            sum_pre, sum_tp = 0, 0
            n = 1000
            n_fold = len(X)/n
            for i in range(1): # n_fold
                print 'model.fit'
                XX = list(X)
                YY = list(Y)
                # print weight
                # anova_svm.fit(np.array(XX[:i*n]+XX[(i+1)*n:]),np.array(YY[:i*n]+YY[(i+1)*n:]))

                anova_svm.fit(np.array(X),np.array(Y))

                # predict_ = anova_svm.predict(X[i*n:(i+1)*n])
                predict_ = anova_svm.predict(testX)

                rank_list = [] 
                for i,j in enumerate(predict_):
                    rank_list.append((j,i))

                rank_list.sort()
                rank_list.reverse()
                print len(rank_list)

                # # evaluate result
                # actual = Y[:1000]
                # print actual
                # # avg_precision = 0
                # # a = []
                # for i in rank_list[:100]:
                #     print predict_[i[1]], actual[i[1]]
                #     a.append(actual[i[1]])
                # # print a
                # for i in range(100):
                #     avg_precision += 1.0*sum(a[:(i+1)])/(i+1)
                # avg_precision = avg_precision/100
                # print 'avg_precision:', avg_precision

                # write result to file
                with open('svm_result.csv', 'w') as f:
                    f.write('Rank_ID\n')
                    for i in rank_list:
                        f.write(str(i[1]+1)+'\n')
                print ' '
