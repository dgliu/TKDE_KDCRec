import os
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from utils.modelnames import models
from utils.progress import WorkSplitter
from scipy.sparse import lil_matrix
from models.predictor import predict
from evaluation.metrics import evaluate
from utils.io import load_numpy, save_numpy, load_yaml, find_single_best_hyperparameters, save_dataframe_csv, load_dataframe_csv
from utils.argcheck import check_int_positive


def main(args):
    progress = WorkSplitter()

    progress.section("Set the random seed")
    np.random.seed(args.seed)

    progress.section("Data generation")
    if args.generation:
        R_rtrain = load_numpy(path=args.path, name=args.problem+args.unif_train)
        user_item_matrix = lil_matrix(R_rtrain)
        user_item_pairs = np.asarray(user_item_matrix.nonzero()).T

        save_dir = args.path + args.problem + 'sub/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        ratio = args.ratio.split(',')
        for r in ratio:
            index = np.random.choice(len(user_item_pairs), int(len(user_item_pairs) * float(r)), replace=False)
            rows, cols, rating = user_item_pairs[index, 0], user_item_pairs[index, 1], np.asarray(
                R_rtrain[user_item_pairs[index, 0], user_item_pairs[index, 1]]).flatten()
            R_rtrain_sub = sparse.csr_matrix((rating, (rows, cols)), shape=R_rtrain.shape, dtype=np.float32)

            save_numpy(R_rtrain_sub, save_dir, "St_" + str(r))

    progress.section("Get the results of the models")
    model_list = ['bridge_var1', 'bridge_var2', 'refine', 'local_weightc', 'global_weightc', 'delay', 'alter_featuree',
                  'concat_featuree']

    table_path = load_yaml('config/global.yml', key='path')['tables']
    topK = [5, 10, 15, 20, 50]
    metric = ['NLL', 'AUC']

    ratio = args.ratio.split(',')
    for r in ratio:
        # load data
        R_train = load_numpy(path=args.path, name=args.problem + args.train)
        R_valid = load_numpy(path=args.path, name=args.problem + args.valid)
        R_test = load_numpy(path=args.path, name=args.problem + args.test)
        sub_R_rtrain = load_numpy(path=args.path, name=args.problem + 'sub/St_' + str(r) + '.npz')

        # pretrin St
        if args.pretrain:
            df = find_single_best_hyperparameters(table_path + args.problem + 'unif_ae_tuning_r.csv', args.main_metric)
            RQ, X, xBias, Yt, yBias = models[df['model']](R_train, R_valid, embeded_matrix=np.empty(0),
                                                          matrix_utrain=sub_R_rtrain,
                                                          iteration=df['iter'], rank=df['rank'],
                                                          corruption=df['confidence'],
                                                          gpu_on=args.gpu, lam=df['lambda'], lam2=df['lambda2'],
                                                          alpha=df['alpha'], seed=args.seed,
                                                          batch_size=df['batch_size'],
                                                          step=df['step'], source=df['source'], metric=args.main_metric,
                                                          topK=topK[-1])

            np.save('latent/pretrain/{2}/{3}_U_{0}_{1}_{4}'.format(df['model'], df['rank'], args.problem, 'unif',
                                                                   args.scene), RQ)
            np.save('latent/pretrain/{2}/{3}_Y_{0}_{1}_{4}'.format(df['model'], df['rank'], args.problem, 'unif',
                                                                   args.scene), Yt)
            if yBias is not None:
                np.save(
                    'latent/pretrain/{2}/{3}_yB_{0}_{1}_{4}'.format(df['model'], df['rank'], args.problem, 'unif',
                                                                    args.scene), yBias)
            if X is not None:
                np.save(
                    'latent/pretrain/{2}/{3}_X_{0}_{1}_{4}'.format(df['model'], df['rank'], args.problem, 'unif',
                                                                   args.scene), X)
            if xBias is not None:
                np.save(
                    'latent/pretrain/{2}/{3}_xB_{0}_{1}_{4}'.format(df['model'], df['rank'], args.problem, 'unif',
                                                                    args.scene), xBias)
        save_path = args.problem + 'scale_' + str(r) + '.csv'
        if not os.path.exists(table_path + save_path):
            result_df = pd.DataFrame(columns=['model'])
        else:
            result_df = load_dataframe_csv(table_path, save_path)

        # results
        for algorithm in model_list:
            if (result_df['model'] == algorithm).any():
                continue

            table_name = algorithm + '_ae_tuning_r.csv'
            df = find_single_best_hyperparameters(table_path + args.problem + table_name, args.main_metric)

            RQ, X, xBias, Yt, yBias = models[df['model']](R_train, R_valid, embeded_matrix=np.empty(0),
                                                          matrix_utrain=sub_R_rtrain, problem=args.problem,
                                                          iteration=df['iter'], rank=df['rank'],
                                                          corruption=df['confidence'], scene=args.scene,
                                                          gpu_on=args.gpu, lam=df['lambda'], lam2=df['lambda2'],
                                                          alpha=df['alpha'], seed=args.seed,
                                                          batch_size=df['batch_size'], mode=df['mode'],
                                                          step=df['step'], source=df['source'], metric=args.main_metric,
                                                          topK=topK[-1])
            Y = Yt.T

            rating_prediction, topk_prediction = predict(matrix_U=RQ,
                                                         matrix_V=Y,
                                                         bias_V=yBias,
                                                         topK=topK[-1],
                                                         matrix_Train=R_train,
                                                         matrix_Test=R_test,
                                                         gpu=args.gpu,
                                                         is_topK=False)

            result = evaluate(rating_prediction, topk_prediction, R_test, metric, topK)

            result_dict = {'model': algorithm}
            for name in result.keys():
                result_dict[name] = [round(result[name][0], 5), round(result[name][1], 5)]
            result_df = result_df.append(result_dict, ignore_index=True)
            save_dataframe_csv(result_df, table_path, save_path)
        print('ratio %f completed\n' % (float(r)))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Scale Analysis")
    parser.add_argument('-p', dest='path', default='datax/')
    parser.add_argument('-d', dest='problem', default='yahooR3/')
    parser.add_argument('-se', dest='scene', default='t')
    parser.add_argument('-t', dest='train', default='S_c.npz')
    parser.add_argument('-v', dest='valid', default='S_va.npz')
    parser.add_argument('-e', dest='test', default='S_te.npz')
    parser.add_argument('-ut', dest='unif_train', default='S_t.npz')
    parser.add_argument('-m', dest='main_metric', default='AUC')
    parser.add_argument('-r', dest='ratio', default='0.2,0.4,0.6,0.8')
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=0)
    parser.add_argument('-g', dest='generation', action='store_false', default=True)
    parser.add_argument('-pre', dest='pretrain', action='store_false', default=True)
    parser.add_argument('-gpu', dest='gpu', action='store_false', default=True)

    args = parser.parse_args()
    main(args)
