import pandas as pd
import numpy as np
from collections import Counter

result ={}
def perform_desc(lab_path, unlab_path):
    # load labeled data
    my_type = ['int64', 'float64']

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

    label_class_dist = Counter(lab_dat['label'])
    lab_dat_size = lab_dat.shape[0]
    unlab_dat_size = unlab_dat.shape[0]
    lab_ratio = round(unlab_dat_size / lab_dat_size, 2)
    pos_ratio = round(label_class_dist[0] / label_class_dist[1], 2)

    result['Number of labeled samples'] = lab_dat_size
    result['Number of unlabeled samples'] = unlab_dat_size
    result['Ratio of labeled to unlabeled dataset '] = str(1) + " : " + str(lab_ratio)
    result['Ratio of Pos/Neg in labeled dataset '] = str(1) + " : " + str(pos_ratio)
    result['Number of features '] = unlab_dat.shape[1]

    return result


def main(lab_file, unlab_file):
    # check if infiles are csv
    if lab_file.name[-4:] == '.csv' and unlab_file.name[-4:] == '.csv':
        # check file size of input
        if (lab_file.size < 1000000) or (unlab_file.size < 1000000):
            inp = 'heal/media/indir/'
            lab_path = ''.join((inp, lab_file.name))
            unlab_path = ''.join((inp, unlab_file.name))
            res = perform_desc(lab_path, unlab_path)
            return res
        else:
            result['error'] = "Input file is too large. Max size is 1Mb!"
            return result
    else:
        result['error'] = "Input files must be csv files!"
        return result