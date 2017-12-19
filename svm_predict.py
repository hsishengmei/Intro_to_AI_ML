from sklearn import svm
from data_parser import parse_data


if __name__ == '__main__':
    X, Y, testX = parse_data()
    '''
	
    '''
    model = svm.SVC(kernel='linear', C=1, gamma=1)
    model.fit(X[50:],Y[50:])
    model.score(X[50:],Y[50:])
    predict_ = model.predict(X[:50])
    actual = Y[:50]
    print 'predict:', predict_
    print 'actual: ', actual