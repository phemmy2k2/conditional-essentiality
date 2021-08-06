import pandas as pd
import numpy as np
from collections import Counter
from copy import deepcopy
# from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_curve, auc, precision_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import math

def preprocess(X):
    # X = np.nan_to_num(X)
    X[X == np.inf] = 0
    numpyMatrix = X.astype(np.float32)

    # replace missing values with mean
    imputer = SimpleImputer(strategy='median')
    numpyMatrix = imputer.fit_transform(numpyMatrix)
    # Create the Scaler object
    scaler = preprocessing.StandardScaler()  # with_mean=False
    # Fit your data on the scaler object
    X = scaler.fit_transform(numpyMatrix)
    return X


# def feature_selector(X, y, num_feats= 9): #400 #9(PPI) #21(topology), num_feats= 400,, weight
#     from sklearn.feature_selection import SelectFromModel
#     # from lightgbm import LGBMClassifier
#
#     lgbc = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
#                           reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40, n_jobs=6) #, scale_pos_weight=weight
#
#     embeded_lgb_selector = SelectFromModel(lgbc, threshold=1) #, max_features=num_feats
#     embeded_lgb_selector.fit(X, y)
#
#     embeded_lgb_support = embeded_lgb_selector.get_support()
#     return embeded_lgb_support

def execmodel(unlab_dat, lab_dat, cuttoff, outdir, endloop, res_store, acc, result, i):
    print("Shape of labelled data %s" % str(lab_dat.shape))
    print("Shape of unlabelled data %s" % str(unlab_dat.shape))
    # initialize result dict object for the current iteration
    res_name = 'Iteration ' + str(i)
    result[res_name] = {}
    # result[res_name]['line0'] = res_name
    y = lab_dat['label'].values
    X_temp = lab_dat.drop(columns=['label'])
    # X = X_temp.values
    X = preprocess(X_temp.values)

    counter = Counter(y)
    print('Class distribution in Unlabelled data is %s' % str(counter))
    result[res_name]['line1'] = 'Class distribution in Unlabelled data is %s' % str(counter)
    # train model
    # clf = LGBMClassifier(n_estimators=800, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
    #                      reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40, scale_pos_weight=estimate)
    clf = RandomForestClassifier(n_estimators=800, max_depth=70, min_samples_leaf=4, min_samples_split=10,
                                 class_weight='balanced')
    res = {'roc_auc': [], 'pr': [], 'sensitivity': [], 'specificity': [], 'accuracy': [], 'precision': []}
    cv = StratifiedKFold(n_splits=5)
    print('Performing 5 fold CV...')
    # ind = None
    auc_best = 0.0
    clf_best = None
    # ind_best = None
    for train_index, test_index in cv.split(X, y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # perform data resampling to have balanced label class
        X_resampled, y_resampled = SMOTE().fit_resample(x_train, y_train)
        # ind = feature_selector(X_resampled, y_resampled) #, estimate

        clf.fit(X_resampled, y_resampled)  # [:, ind]
        ### Performance Evaluation Metrics #############
        y_pred = clf.predict(x_test)
        probs = clf.predict_proba(x_test)[::, 1]  # [:, ind]
        fpr, tpr, thresholds = roc_curve(y_test, probs, pos_label=1, drop_intermediate=False)
        precision, recall, _ = precision_recall_curve(y_test, probs, pos_label=1)
        pr_auc = auc(recall, precision)
        roc_auc = auc(fpr, tpr)  # average='macro'(default) or 'micro'

        cm1 = confusion_matrix(y_test, y_pred)
        # total1 = sum(sum(cm1))
        a, b, c, d = cm1[0, 0], cm1[0, 1], cm1[1, 0], cm1[1, 1]
        specificity = a / (a + b)
        sensitivity = d / (c + d)
        accuracy = (sensitivity + specificity) / 2
        preci_score = precision_score(y_test, y_pred, pos_label=1)

        res['roc_auc'].append(roc_auc)
        res['pr'].append(pr_auc)
        res['sensitivity'].append(sensitivity)
        res['specificity'].append(specificity)
        res['accuracy'].append(accuracy)
        res['precision'].append(preci_score)
        if pr_auc > auc_best:
            auc_best = pr_auc
            clf_best = deepcopy(clf)
            # ind_best = ind
            # print(auc_best)

    res_test = {k: np.mean(v) for k, v in res.items()}
    print("ROC-AUC Score: %.3f" % res_test['roc_auc'])
    print("PR-AUC Score: %.3f" % res_test['pr'])
    print("Precision Score: %.3f" % res_test['precision'])
    print("Recall Score: %.3f" % res_test['sensitivity'])
    print("Specificity Score: %.3f" % res_test['specificity'])
    print("Accuracy Score: %.3f" % res_test['accuracy'])
    acc.append(res_test['accuracy'])
    result[res_name]['line2'] = "Accuracy Score: %.3f" % res_test['accuracy']

    # classify pool data
    # unlab = unlab_dat.values
    unlab = preprocess(unlab_dat.values)
    probs = clf_best.predict_proba(unlab)[::, 1]  # [:, ind_best]
    # probs = clf_best.predict_proba(unlab[:, ind_best])[::, 1]

    # cols = np.array(X_temp.columns)[ind_best]
    # print('No of features selected is %d' %len(cols))
    # feat_imp = pd.Series(clf_best.feature_importances_, cols)
    # feat_imp.to_csv(''.join(('data/cml/cancer/featImp/pred_round_', str(i - 1), '.csv')))

    # separate data based on predictions in the specified range
    res = pd.DataFrame([list(unlab_dat.index), list(probs)])
    res = res.transpose()
    res.columns = ['sampleId', 'probs']
    res['prediction'] = res.probs.astype(float).round().astype(int)
    print('Ratio of +ve and -ve prediction in unlabelled data %s ' % str(Counter(res['prediction'])))
    result[res_name]['line3'] = 'Ratio of +ve and -ve prediction in unlabelled data %s ' % str(Counter(res['prediction']))

    # get lower percentile of neg samples and upper percentile of pos samples
    lp = res[res['probs'].lt(0.5)]['probs'].quantile(q=0.25, interpolation='midpoint')
    up = res[res['probs'].gt(0.5)]['probs'].quantile(q=0.75, interpolation='midpoint')
    print("Estimated lower percentile %.5f and Estimated upper percentile is %.3f" % (lp, up))
    result[res_name]['line4'] = "Estimated lower percentile %.5f and Estimated upper percentile is %.3f" % (lp, up)

    if math.isnan(lp) or math.isnan(up):
        res_store = pd.concat([res_store, res], axis=0, sort=False)
        res_store.to_csv(outdir, index=False)

    else:
        # set upper limit to threshold if auto score is less than manual score
        if up < cuttoff:
            up = cuttoff

        # check if the sample size is large enough for another iteration

        if unlab_dat.shape[0] > endloop:  # 1500, 300
            res = res[~res['probs'].between(lp, up)]  # selects samples above 0.9 and below 0.02
            # combine prediction into a dataframe
            if res_store.empty:
                print('res_store is empty')
                res_store = res_store.append(res)
            else:
                print('res_store is populates, shape is %s' % str(res_store.shape))
                res_store = pd.concat([res_store, res], axis=0, sort=False)
                # res.to_csv(''.join(('data/cml/cancer/predictions/pred_round_', str(i - 1), '.csv')), index=False)
                # print('Ratio of +ve and -ve prediction to be added to labelled data %s ' %str(Counter(res['prediction'])))
                # print('Number of new samples to be added to the labelled data for round %d is %d' % (i, res.shape[0]))
                result[res_name]['line5'] = 'Number of new samples to be added to the labelled data for iteration %d is %d' % (i, res.shape[0])
        else:
            res_store = pd.concat([res_store, res], axis=0, sort=False)
            res_store.to_csv(outdir, index=False)

    # add specified range to labeled data and del from pool data
    inc_index = res['sampleId'].tolist()
    inc = unlab_dat[unlab_dat.index.isin(inc_index)]

    out = inc.copy()
    out['label'] = res.prediction.values
    # print(out['label'].value_counts())
    # res.to_csv('dm/al_test/pred_select.csv', index=None)

    # filter classifier output and update label and unlabel data
    # lab_dat = pd.concat([lab_dat, out], axis=0, sort=False)
    # print(Counter(lab_dat['label']))

    # set unused variables to None
    del inc
    del res

    unlab_dat.drop(index=inc_index, inplace=True)  # remove accepted samples from pool of unlabeled data
    return out, res_store, acc

result = {}
def perform_CML(userId, lab_path, unlab_path, outdir, threshold):
    # load labeled data
    i = 1
    my_type = ['int64', 'float64']
    res_store = pd.DataFrame()
    acc = []

    lab_dat = pd.read_csv(lab_path, index_col=0, na_values='?')
    unlab_dat = pd.read_csv(unlab_path, index_col=0, na_values='?')

    # Ensure all columns in label and unlabel set are same except for label
    cols_lab = lab_dat.columns.tolist()
    cols_unlab = unlab_dat.columns.tolist()
    label = set(cols_unlab).symmetric_difference(set(cols_lab))

    if len(label) == 1 and list(label)[0] == 'label':
        # Ensure class label is 0 and 1 values only
        # Validate if it is a binary classification by using Counter
        count_lab = Counter(lab_dat['label'])
        assert (0 in count_lab and 1 in count_lab), ' The label values are not 0 and 1'
        # print(count_lab)
        # print(len(count_lab))
        # print(count_lab[0], count_lab[1])
    else:
        # failure
        result['error'] = "There is inconsistency in the variable names in the input data"
        return result

    # Ensure all columns are numeric type
    # print(lab_dat.dtypes)

    col_dtypes = lab_dat.dtypes.to_dict()
    for item, typ in col_dtypes.items():
        if (typ not in my_type):
            result['error'] = "Only numeric variables are allowed in the input data"
            return result

    endloop = unlab_dat.shape[0] // 10
    # label_class_dist = Counter(lab_dat['label'])
    # lab_dat_size = lab_dat.shape[0]
    # unlab_dat_size = unlab_dat.shape[0]
    # lab_ratio = round(unlab_dat_size/lab_dat_size, 2)
    # pos_ratio = round(label_class_dist[0]/label_class_dist[1], 2)

    while (unlab_dat.shape[0] > 1):
        print('\n' + 'Loop %d\n' % i)
        out, res_store, acc = execmodel(unlab_dat, lab_dat, threshold, outdir, endloop, res_store, acc, result, i)
        res_store = res_store
        acc = acc
        lab_dat = pd.concat([lab_dat, out], axis=0, sort=False)

        i += 1
    # compute and store basic stats of the data
    # result['label_class_dist'] = Counter(lab_dat['label'])

    # result['Number of labeled samples'] = lab_dat_size
    # result['Number of unlabeled samples'] = unlab_dat_size
    # result['Ratio of labeled to unlabeled dataset '] = str(1) + " : " + str(lab_ratio)
    # result['Ratio of Pos/Neg in labeled dataset '] = str(1) + " : " + str(pos_ratio)
    # result['Number of features '] = unlab_dat.shape[1]
    # result['Base model accuracy '] = str(round(acc[0] * 100, 2)) + "%"
    result['download_file'] = 'pred_' + str(userId) + '.csv'
    return result


def main(userId, lab_file, unlab_file, threshold=0.9):
    # check if infiles are csv
    if type(threshold) == str:
        threshold = 0.9
    if lab_file.name[-4:] == '.csv' and unlab_file.name[-4:] == '.csv':
        # check file size of input
        if (lab_file.size < 1000000) or (unlab_file.size < 1000000):
            inp = 'heal/media/indir/'
            outdir = ''.join(('heal/static/outdir/', 'pred_', str(userId), '.csv'))
            lab_path = ''.join((inp, lab_file.name))
            unlab_path = ''.join((inp, unlab_file.name))
            res = perform_CML(userId, lab_path, unlab_path, outdir, float(threshold))
            return res
        else:
            result['error'] = "Input file is too large. Max size is 1Mb!"
            return result
    else:
        result['error'] = "Input files must be csv files!"
        return result

# if __name__ == '__main__':
#     # get_data()
#     userId = 8991
#     threshold = 0.9
#     lab_path = 'data/cml/cancer/cancer_labdata_sample.csv'
#     unlab_path = 'data/cml/cancer/cancer_unlabdata_sample.csv'
#
#     print(main(userId, lab_path, unlab_path, threshold))