import os
import yaml
import stat
import pickle
import numpy as np
import pandas as pd
from os import listdir
from ast import literal_eval
from os.path import isfile, join
from scipy.sparse import save_npz, load_npz
from scipy.sparse import lil_matrix, csr_matrix


def load_pandas(path, name, sep, df_name, value_name, shape=None):
    df = pd.read_csv(path + name, sep=sep)
    rows = df[df_name[0]]
    cols = df[df_name[1]]
    if value_name is not None:
        values = df[value_name]
    else:
        values = [1]*len(rows)

    if shape:
        return csr_matrix((values, (rows, cols)), shape=shape)
    else:
        return csr_matrix((values, (rows, cols)), shape=(rows.max() + 1, cols.max() + 1))


def load_pandas_without_names(path, name, sep, df_name, value_name, shape=None):
    df = pd.read_csv(path + name, sep=sep, header=None, names=df_name)
    rows = df[df_name[0]]
    cols = df[df_name[1]]
    if value_name is not None:
        values = df[value_name]
    else:
        values = [1]*len(rows)

    if shape:
        return csr_matrix((values, (rows, cols)), shape=shape)
    else:
        return csr_matrix((values, (rows, cols)), shape=(rows.max() + 1, cols.max() + 1))


def save_numpy(matrix, path, model):
    save_npz('{0}{1}'.format(path, model), matrix)


def save_array(array, path, model):
    np.save('{0}{1}'.format(path, model), array)


def load_numpy(path, name):
    return load_npz(path+name).tocsr()


def load_dataframe_csv(path, name):
    return pd.read_csv(path+name)


def save_dataframe_csv(df, path, name):
    df.to_csv(path+name, index=False)


def find_best_hyperparameters(folder_path, meatric, scene='r'):
    csv_files = [join(folder_path, f) for f in listdir(folder_path)
                 if isfile(join(folder_path, f)) and f.endswith('tuning_'+scene+'.csv') and not f.startswith('final')]
    best_settings = []
    for record in csv_files:
        df = pd.read_csv(record)
        df[meatric+'_Score'] = df[meatric].map(lambda x: literal_eval(x)[0])
        best_settings.append(df.loc[df[meatric+'_Score'].idxmax()].to_dict())

    df = pd.DataFrame(best_settings).drop(meatric+'_Score', axis=1)

    return df


def find_single_best_hyperparameters(folder_path, meatric):
    df = pd.read_csv(folder_path)
    df[meatric + '_Score'] = df[meatric].map(lambda x: literal_eval(x)[0])
    best_settings = df.loc[df[meatric + '_Score'].idxmax()].to_dict()

    return best_settings


def save_dataframe_latex(df, path, model):
    with open('{0}{1}_parameter_tuning.tex'.format(path, model), 'w') as handle:
        handle.write(df.to_latex(index=False))


def load_csv(path, name, shape=(1010000, 2262292)):
    data = np.genfromtxt(path + name, delimiter=',')
    matrix = csr_matrix((data[:, 2], (data[:, 0], data[:, 1])), shape=shape)
    save_npz(path + "rating.npz", matrix)
    return matrix


def save_pickle(path, name, data):
    with open('{0}/{1}.pickle'.format(path, name), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path, name):
    with open('{0}/{1}.pickle'.format(path, name), 'rb') as handle:
        data = pickle.load(handle)

    return data


def load_yaml(path, key='parameters'):
    with open(path, 'r') as stream:
        try:
            return yaml.load(stream)[key]
        except yaml.YAMLError as exc:
            print(exc)


def get_file_names(folder_path, extension='.yml'):
    return [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith(extension)]


def write_file(folder_path, file_name, content, exe=False):
    full_path = folder_path+'/'+file_name
    with open(full_path, 'w') as the_file:
        the_file.write(content)

    if exe:
        st = os.stat(full_path)
        os.chmod(full_path, st.st_mode | stat.S_IEXEC)


def sample_from_sc_with_mode(matrix_train, matrix_utrain, sample_size, pos_ratio, neg_ratio, mode='random'):
    m, n = matrix_train.shape

    user_item_matrix = lil_matrix(matrix_train)
    user_item_pairs = np.asarray(user_item_matrix.nonzero()).T
    label = np.asarray(matrix_train[user_item_pairs[:, 0], user_item_pairs[:, 1]])[0]

    # mode: random, in, out, head, tail
    if mode == 'random':
        pos_index = np.where(label == 1)[0]
        neg_index = np.where(label == -1)[0]

        sample_pos_idx = np.random.choice(np.arange(int(len(pos_index)), dtype=int),
                                          size=int(sample_size * pos_ratio), replace=False)
        pos_sample_pair = user_item_pairs[pos_index[sample_pos_idx]]
        pos_sample_label = np.asarray(matrix_train[pos_sample_pair[:, 0], pos_sample_pair[:, 1]]).T
        sample_neg_idx = np.random.choice(np.arange(int(len(neg_index)), dtype=int),
                                          size=int(sample_size * neg_ratio), replace=False)
        neg_sample_pair = user_item_pairs[neg_index[sample_neg_idx]]
        neg_sample_label = np.asarray(matrix_train[neg_sample_pair[:, 0], neg_sample_pair[:, 1]]).T

        sample_ui_pairs = np.vstack((pos_sample_pair, neg_sample_pair))
        sample_labels = np.vstack((pos_sample_label, neg_sample_label))
        matrix_sample = csr_matrix((sample_labels[:, 0], (sample_ui_pairs[:, 0], sample_ui_pairs[:, 1])), shape=(m, n))

        matrix_train -= matrix_sample
        matrix_utrain += matrix_sample

    elif mode == 'in':
        marks = csr_matrix(matrix_utrain.shape)
        marks[(matrix_utrain != 0).nonzero()] = 1

        unif_user_set = np.where((np.sum(marks, axis=1)) > 0)[0]
        unif_user_mask = np.isin(user_item_pairs[:, 0], unif_user_set)
        pos_index = np.where(((label == 1) & unif_user_mask))[0]
        neg_index = np.where(((label == -1) & unif_user_mask))[0]

        sample_pos_idx = np.random.choice(np.arange(int(len(pos_index)), dtype=int),
                                          size=int(sample_size * pos_ratio), replace=False)
        pos_sample_pair = user_item_pairs[pos_index[sample_pos_idx]]
        pos_sample_label = np.asarray(matrix_train[pos_sample_pair[:, 0], pos_sample_pair[:, 1]]).T
        sample_neg_idx = np.random.choice(np.arange(int(len(neg_index)), dtype=int),
                                          size=int(sample_size * neg_ratio), replace=False)
        neg_sample_pair = user_item_pairs[neg_index[sample_neg_idx]]
        neg_sample_label = np.asarray(matrix_train[neg_sample_pair[:, 0], neg_sample_pair[:, 1]]).T

        sample_ui_pairs = np.vstack((pos_sample_pair, neg_sample_pair))
        sample_labels = np.vstack((pos_sample_label, neg_sample_label))
        matrix_sample = csr_matrix((sample_labels[:, 0], (sample_ui_pairs[:, 0], sample_ui_pairs[:, 1])), shape=(m, n))

        matrix_train -= matrix_sample
        matrix_utrain += matrix_sample

    elif mode == 'out':
        marks = csr_matrix(matrix_utrain.shape)
        marks[(matrix_utrain != 0).nonzero()] = 1

        unif_user_set = np.where((np.sum(marks, axis=1)) > 0)[0]
        non_unif_user_mask = np.isin(user_item_pairs[:, 0], unif_user_set, invert=True)
        pos_index = np.where(((label == 1) & non_unif_user_mask))[0]
        neg_index = np.where(((label == -1) & non_unif_user_mask))[0]

        sample_pos_idx = np.random.choice(np.arange(int(len(pos_index)), dtype=int),
                                          size=int(sample_size * pos_ratio), replace=False)
        pos_sample_pair = user_item_pairs[pos_index[sample_pos_idx]]
        pos_sample_label = np.asarray(matrix_train[pos_sample_pair[:, 0], pos_sample_pair[:, 1]]).T
        sample_neg_idx = np.random.choice(np.arange(int(len(neg_index)), dtype=int),
                                          size=int(sample_size * neg_ratio), replace=False)
        neg_sample_pair = user_item_pairs[neg_index[sample_neg_idx]]
        neg_sample_label = np.asarray(matrix_train[neg_sample_pair[:, 0], neg_sample_pair[:, 1]]).T

        sample_ui_pairs = np.vstack((pos_sample_pair, neg_sample_pair))
        sample_labels = np.vstack((pos_sample_label, neg_sample_label))
        matrix_sample = csr_matrix((sample_labels[:, 0], (sample_ui_pairs[:, 0], sample_ui_pairs[:, 1])), shape=(m, n))

        matrix_train -= matrix_sample
        matrix_utrain += matrix_sample

    elif mode == 'head':
        marks = csr_matrix(matrix_train.shape)
        marks[(matrix_train != 0).nonzero()] = 1

        item_pop_sort_index = np.argsort(-np.sum(marks, axis=0)).A1
        pop_item_set = item_pop_sort_index[0:int(0.3 * len(item_pop_sort_index))]
        pop_item_mask = np.isin(user_item_pairs[:, 1], pop_item_set)
        pos_index = np.where(((label == 1) & pop_item_mask))[0]
        neg_index = np.where(((label == -1) & pop_item_mask))[0]

        sample_pos_idx = np.random.choice(np.arange(int(len(pos_index)), dtype=int),
                                          size=int(sample_size * pos_ratio), replace=False)
        pos_sample_pair = user_item_pairs[pos_index[sample_pos_idx]]
        pos_sample_label = np.asarray(matrix_train[pos_sample_pair[:, 0], pos_sample_pair[:, 1]]).T
        sample_neg_idx = np.random.choice(np.arange(int(len(neg_index)), dtype=int),
                                          size=int(sample_size * neg_ratio), replace=False)
        neg_sample_pair = user_item_pairs[neg_index[sample_neg_idx]]
        neg_sample_label = np.asarray(matrix_train[neg_sample_pair[:, 0], neg_sample_pair[:, 1]]).T

        sample_ui_pairs = np.vstack((pos_sample_pair, neg_sample_pair))
        sample_labels = np.vstack((pos_sample_label, neg_sample_label))
        matrix_sample = csr_matrix((sample_labels[:, 0], (sample_ui_pairs[:, 0], sample_ui_pairs[:, 1])), shape=(m, n))

        matrix_train -= matrix_sample
        matrix_utrain += matrix_sample

    elif mode == 'tail':
        marks = csr_matrix(matrix_train.shape)
        marks[(matrix_train != 0).nonzero()] = 1

        item_pop_sort_index = np.argsort(-np.sum(marks, axis=0)).A1
        non_pop_item_set = item_pop_sort_index[int(0.3 * len(item_pop_sort_index)):]
        non_pop_item_mask = np.isin(user_item_pairs[:, 1], non_pop_item_set)
        pos_index = np.where(((label == 1) & non_pop_item_mask))[0]
        neg_index = np.where(((label == -1) & non_pop_item_mask))[0]

        sample_pos_idx = np.random.choice(np.arange(int(len(pos_index)), dtype=int),
                                          size=int(sample_size * pos_ratio), replace=False)
        pos_sample_pair = user_item_pairs[pos_index[sample_pos_idx]]
        pos_sample_label = np.asarray(matrix_train[pos_sample_pair[:, 0], pos_sample_pair[:, 1]]).T
        sample_neg_idx = np.random.choice(np.arange(int(len(neg_index)), dtype=int),
                                          size=int(sample_size * neg_ratio), replace=False)
        neg_sample_pair = user_item_pairs[neg_index[sample_neg_idx]]
        neg_sample_label = np.asarray(matrix_train[neg_sample_pair[:, 0], neg_sample_pair[:, 1]]).T

        sample_ui_pairs = np.vstack((pos_sample_pair, neg_sample_pair))
        sample_labels = np.vstack((pos_sample_label, neg_sample_label))
        matrix_sample = csr_matrix((sample_labels[:, 0], (sample_ui_pairs[:, 0], sample_ui_pairs[:, 1])), shape=(m, n))

        matrix_train -= matrix_sample
        matrix_utrain += matrix_sample

    return matrix_train, matrix_utrain




