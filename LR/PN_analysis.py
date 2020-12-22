import os
import argparse
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from utils.modelnames import models
from utils.progress import WorkSplitter
from models.predictor import predict
from evaluation.metrics import evaluate
from utils.argcheck import check_int_positive
from utils.io import load_numpy, find_single_best_hyperparameters, load_yaml, save_numpy, save_dataframe_csv, load_dataframe_csv


def main(args):
    progress = WorkSplitter()

    progress.section("Set the random seed")
    np.random.seed(args.seed)

    progress.section("Data generation")
    if args.generation:
        R_rtrain = load_numpy(path=args.path, name=args.problem + args.unif_train)
        print('St pos num: {0}'.format(len((R_rtrain == 1).nonzero()[0])))
        print('St neg num: {0}'.format(len((R_rtrain == -1).nonzero()[0])))

        R_train = load_numpy(path=args.path, name=args.problem + args.train)
        print('Sc pos num: {0}'.format(len((R_train == 1).nonzero()[0])))
        print('Sc neg num: {0}'.format(len((R_train == -1).nonzero()[0])))
        pos_rows, pos_cols = (R_train == 1).nonzero()[0], (R_train == 1).nonzero()[1]
        neg_rows, neg_cols = (R_train == -1).nonzero()[0], (R_train == -1).nonzero()[1]

        total_num = 100000

        save_dir = args.path + args.problem + 'sub/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        ratio = args.pos_ratio.split(',')
        for r in ratio:
            pos_index = np.random.choice(len(pos_rows), int(total_num * float(r)), replace=False)
            neg_index = np.random.choice(len(neg_rows), int(total_num * (1 - float(r))), replace=False)
            rows = np.hstack((pos_rows[pos_index], neg_rows[neg_index]))
            cols = np.hstack((pos_cols[pos_index], neg_cols[neg_index]))

            rating = np.asarray(R_train[rows, cols]).flatten()
            R_train_sub = sparse.csr_matrix((rating, (rows, cols)), shape=R_train.shape, dtype=np.float32)

            save_numpy(R_train_sub, save_dir, "Sc_" + str(r))

    progress.section("Get the results of the models")
    model_list = ['ips', 'bridge_var1', 'bridge_var2', 'refine', 'cause', 'local_weightc', 'global_weightc',
                  'delay', 'alter_featuree', 'concat_featuree']

    table_path = load_yaml('config/global.yml', key='path')['tables']
    topK = [5, 10, 15, 20, 50]
    metric = ['NLL', 'AUC']

    ratio = args.pos_ratio.split(',')
    for r in ratio:
        # load data
        sub_R_train = load_numpy(path=args.path, name=args.problem + 'sub/Sc_' + str(r) + '.npz')
        R_valid = load_numpy(path=args.path, name=args.problem + args.valid)
        R_test = load_numpy(path=args.path, name=args.problem + args.test)
        R_rtrain = load_numpy(path=args.path, name=args.problem + args.unif_train)

        # pretrin Sc and St
        if args.pretrain:
            df = find_single_best_hyperparameters(table_path + args.problem + 'mf_tuning_r.csv', args.main_metric)
            RQ, X, xBias, Yt, yBias = models[df['model']](sub_R_train, R_valid, embeded_matrix=np.empty(0),
                                                          matrix_utrain=R_rtrain,
                                                          iteration=df['iter'], rank=df['rank'],
                                                          corruption=df['confidence'],
                                                          gpu_on=args.gpu, lam=df['lambda'], lam2=df['lambda2'],
                                                          alpha=df['alpha'], seed=args.seed,
                                                          batch_size=df['batch_size'],
                                                          step=df['step'], source=df['source'], metric=args.main_metric,
                                                          topK=topK[-1])
            Y = Yt.T
            np.save('latent/pretrain/{2}/U_{0}_{1}_{3}'.format(df['model'], df['rank'], args.problem, args.scene), RQ)
            np.save('latent/pretrain/{2}/V_{0}_{1}_{3}'.format(df['model'], df['rank'], args.problem, args.scene), Y)
            if yBias is not None:
                np.save('latent/pretrain/{2}/vB_{0}_{1}_{3}'.format(df['model'], df['rank'], args.problem, args.scene),
                        yBias)
            if X is not None:
                np.save('latent/pretrain/{2}/X_{0}_{1}_{3}'.format(df['model'], df['rank'], args.problem, args.scene),
                        X)
            if xBias is not None:
                np.save('latent/pretrain/{2}/uB_{0}_{1}_{3}'.format(df['model'], df['rank'], args.problem, args.scene),
                        xBias)

        save_path = args.problem + 'PN_' + str(r) + '.csv'
        if not os.path.exists(table_path + save_path):
            result_df = pd.DataFrame(columns=['model'])
        else:
            result_df = load_dataframe_csv(table_path, save_path)

        # results
        for algorithm in model_list:
            if (result_df['model'] == algorithm).any():
                continue
            
            table_name = algorithm + '_mf_tuning_r.csv'
            df = find_single_best_hyperparameters(table_path + args.problem + table_name, args.main_metric)

            RQ, X, xBias, Yt, yBias = models[df['model']](sub_R_train, R_valid, embeded_matrix=np.empty(0),
                                                          matrix_utrain=R_rtrain, problem=args.problem,
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
                                                         bias_U=xBias,
                                                         bias_V=yBias,
                                                         topK=topK[-1],
                                                         matrix_Train=sub_R_train,
                                                         matrix_Test=R_test,
                                                         gpu=args.gpu,
                                                         is_topK=False)

            result = evaluate(rating_prediction, topk_prediction, R_test, metric, topK)

            result_dict = {'model': algorithm}
            for name in result.keys():
                result_dict[name] = [round(result[name][0], 4), round(result[name][1], 4)]
            result_df = result_df.append(result_dict, ignore_index=True)
            save_dataframe_csv(result_df, table_path, save_path)
        print('ratio %f completed\n', float(r))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="PN Analysis")
    parser.add_argument('-p', dest='path', default='datax/')
    parser.add_argument('-d', dest='problem', default='yahooR3/')
    parser.add_argument('-se', dest='scene', default='c')
    parser.add_argument('-t', dest='train', default='S_c.npz')
    parser.add_argument('-v', dest='valid', default='S_va.npz')
    parser.add_argument('-e', dest='test', default='S_te.npz')
    parser.add_argument('-ut', dest='unif_train', default='S_t.npz')
    parser.add_argument('-m', dest='main_metric', default='AUC')
    parser.add_argument('-r', dest='pos_ratio', default='0.1,0.3,0.5,0.7,0.9')
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=0)
    parser.add_argument('-g', dest='generation', action='store_false', default=True)
    parser.add_argument('-pre', dest='pretrain', action='store_false', default=True)
    parser.add_argument('-gpu', dest='gpu', action='store_false', default=True)

    args = parser.parse_args()
    main(args)
