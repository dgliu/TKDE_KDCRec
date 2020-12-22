import os
import argparse
import numpy as np
import pandas as pd
from utils.modelnames import models
from models.predictor import predict
from evaluation.metrics import evaluate
from utils.argcheck import check_int_positive
from utils.io import load_numpy, save_numpy, load_yaml, find_single_best_hyperparameters, save_dataframe_csv, load_dataframe_csv


def main(args):
    table_path = load_yaml('config/global.yml', key='path')['tables']

    df = find_single_best_hyperparameters(table_path+args.problem+'refine_ae_tuning_r.csv', args.main_metric)

    R_train = load_numpy(path=args.path, name=args.problem+args.train)
    R_valid = load_numpy(path=args.path, name=args.problem+args.valid)
    R_test = load_numpy(path=args.path, name=args.problem + args.test)
    R_rtrain = load_numpy(path=args.path, name=args.problem + args.unif_train)

    if args.type == 'sample':
        save_path = args.problem + 'refine_sample.csv'
        if not os.path.exists(table_path + save_path):
            result_df = pd.DataFrame(columns=['model'])
        else:
            result_df = load_dataframe_csv(table_path, save_path)
        sample_num = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600]
        for s in sample_num:
            RQ, X, xBias, Yt, yBias = models[df['model']](R_train, R_valid, embeded_matrix=np.empty(0), matrix_utrain=R_rtrain,
                                                          iteration=df['iter'], rank=df['rank'], corruption=df['confidence'],
                                                          gpu_on=args.gpu, lam=df['lambda'], lam2=df['lambda2'],
                                                          alpha=df['alpha'], seed=args.seed, batch_size=df['batch_size'],
                                                          step=df['step'], source=args.source, metric=args.main_metric,
                                                          topK=args.topk, sample_num=s, problem=args.problem)
            Y = Yt.T

            rating_prediction, topk_prediction = predict(matrix_U=RQ,
                                                         matrix_V=Y,
                                                         bias_V=yBias,
                                                         topK=args.topk,
                                                         matrix_Train=R_train,
                                                         matrix_Test=R_test,
                                                         gpu=args.gpu,
                                                         is_topK=False)

            result = evaluate(rating_prediction, topk_prediction, R_test, [args.main_metric], [args.topk])

            result_dict = {'model': 'refine'}
            for name in result.keys():
                result_dict[name] = [round(result[name][0], 4), round(result[name][1], 4)]
            result_df = result_df.append(result_dict, ignore_index=True)
            save_dataframe_csv(result_df, table_path, save_path)
    else:
        save_path = args.problem + 'refine_mode.csv'
        if not os.path.exists(table_path + save_path):
            result_df = pd.DataFrame(columns=['model'])
        else:
            result_df = load_dataframe_csv(table_path, save_path)
        mode = ['in', 'out', 'head', 'tail']
        for m in mode:
            RQ, X, xBias, Yt, yBias = models[df['model']](R_train, R_valid, embeded_matrix=np.empty(0),
                                                          matrix_utrain=R_rtrain,
                                                          iteration=df['iter'], rank=df['rank'],
                                                          corruption=df['confidence'],
                                                          gpu_on=args.gpu, lam=df['lambda'], lam2=df['lambda2'],
                                                          alpha=df['alpha'], seed=args.seed,
                                                          batch_size=df['batch_size'],
                                                          step=df['step'], source=args.source, metric=args.main_metric,
                                                          topK=args.topk, mode=m, problem=args.problem)
            Y = Yt.T

            rating_prediction, topk_prediction = predict(matrix_U=RQ,
                                                         matrix_V=Y,
                                                         bias_V=yBias,
                                                         topK=args.topk,
                                                         matrix_Train=R_train,
                                                         matrix_Test=R_test,
                                                         gpu=args.gpu,
                                                         is_topK=False)

            result = evaluate(rating_prediction, topk_prediction, R_test, ['NLL', 'AUC'], [args.topk])

            result_dict = {'model': 'refine'}
            for name in result.keys():
                result_dict[name] = [round(result[name][0], 4), round(result[name][1], 4)]
            result_df = result_df.append(result_dict, ignore_index=True)
            save_dataframe_csv(result_df, table_path, save_path)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Analysis")
    parser.add_argument('-p', dest='path', default="datax/")
    parser.add_argument('-d', dest='problem', default="yahooR3/")
    parser.add_argument('-se', dest='scene', default='r')  # or 'u' (r: random, u: user)
    parser.add_argument('-t', dest='train', default='S_c.npz')
    parser.add_argument('-v', dest='valid', default='S_va.npz')
    parser.add_argument('-e', dest='test', default='S_te.npz')
    parser.add_argument('-ut', dest='unif_train', default='S_t.npz')
    parser.add_argument('-sr', dest='source', default=None)  # or 'unif'
    parser.add_argument('-m', dest='main_metric', default='AUC')
    parser.add_argument('-ty', dest='type', default='sample')
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=0)
    parser.add_argument('-k', dest='topk', type=check_int_positive, default=50)
    parser.add_argument('-gpu', dest='gpu', action='store_false', default=True)
    args = parser.parse_args()

    main(args)