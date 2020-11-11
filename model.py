from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from mlxtend.classifier import StackingCVClassifier


'''
The method is used to build a stacking model.
'''
def build_model():
    knn = KNeighborsClassifier(n_neighbors=15)
    gnb = GaussianNB()
    lgb = LGBMClassifier()

    svm_1 = svm.SVC(C=2.0, gamma=0.00048828125, probability=True)
    svm_2 = svm.SVC(C=8.0, gamma=0.00048828125, probability=True)
    svm_3 = svm.SVC(probability=True)

    lr_1 = LogisticRegression()

    scvc = StackingCVClassifier(classifiers=[knn, gnb, lgb, svm_1, svm_2, svm_3,lr_1],
                                  use_probas=True,
                                  drop_last_proba=True,
                                  meta_classifier=lr_1,
                                  cv=5,
                                  random_state=42)

    return scvc





