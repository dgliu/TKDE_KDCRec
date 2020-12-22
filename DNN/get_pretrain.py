import os
import argparse
import numpy as np
from utils.modelnames import models
from models.predictor import predict
from evaluation.metrics import evaluate
from utils.argcheck import check_int_positive
from utils.io import load_numpy, find_single_best_hyperparameters, load_yaml


def main(args):
    table_path = load_yaml('config/global.yml', key='path')['tables']

    df = find_single_best_hyperparameters(table_path+args.problem+args.table_name, args.main_metric)

    R_train = load_numpy(path=args.path, name=args.problem+args.train)
    R_valid = load_numpy(path=args.path, name=args.problem+args.valid)
    R_rtrain = load_numpy(path=args.path, name=args.problem + args.unif_train)

    RQ, X, xBias, Yt, yBias = models[df['model']](R_train, R_valid, embeded_matrix=np.empty(0), matrix_utrain=R_rtrain,
                                                  iteration=df['iter'], rank=df['rank'], corruption=df['confidence'],
                                                  gpu_on=args.gpu, lam=df['lambda'], lam2=df['lambda2'],
                                                  alpha=df['alpha'], seed=args.seed, batch_size=df['batch_size'],
                                                  step=df['step'], source=args.source, metric=args.main_metric,
                                                  topK=args.topk)

    if not os.path.exists('latent/pretrain/'+args.problem):
        os.makedirs('latent/pretrain/'+args.problem)

    if args.source:
        np.save('latent/pretrain/{2}/{3}_U_{0}_{1}_{4}'.format(df['model'], df['rank'], args.problem, args.source, args.scene), RQ)
        np.save('latent/pretrain/{2}/{3}_Y_{0}_{1}_{4}'.format(df['model'], df['rank'], args.problem, args.source, args.scene), Yt)
        if yBias is not None:
            np.save('latent/pretrain/{2}/{3}_yB_{0}_{1}_{4}'.format(df['model'], df['rank'], args.problem, args.source, args.scene), yBias)
        if X is not None:
            np.save('latent/pretrain/{2}/{3}_X_{0}_{1}_{4}'.format(df['model'], df['rank'], args.problem, args.source, args.scene), X)
        if xBias is not None:
            np.save('latent/pretrain/{2}/{3}_xB_{0}_{1}_{4}'.format(df['model'], df['rank'], args.problem, args.source, args.scene), xBias)
    else:
        np.save('latent/pretrain/{2}/U_{0}_{1}_{3}'.format(df['model'], df['rank'], args.problem, args.scene), RQ)
        np.save('latent/pretrain/{2}/Y_{0}_{1}_{3}'.format(df['model'], df['rank'], args.problem, args.scene), Yt)
        if yBias is not None:
            np.save('latent/pretrain/{2}/yB_{0}_{1}_{3}'.format(df['model'], df['rank'], args.problem, args.scene), yBias)
        if X is not None:
            np.save('latent/pretrain/{2}/X_{0}_{1}_{3}'.format(df['model'], df['rank'], args.problem, args.scene), X)
        if xBias is not None:
            np.save('latent/pretrain/{2}/xB_{0}_{1}_{3}'.format(df['model'], df['rank'], args.problem, args.scene), xBias)

    rating_prediction, topk_prediction = predict(matrix_U=RQ,
                                                 matrix_V=Yt.T,
                                                 bias_V=yBias,
                                                 topK=args.topk,
                                                 matrix_Train=R_train,
                                                 matrix_Test=R_valid,
                                                 gpu=args.gpu,
                                                 is_topK=False)

    result = evaluate(rating_prediction, topk_prediction, R_valid, [args.main_metric], [args.topk])
    print("-")


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Pretrain")
    parser.add_argument('-tb', dest='table_name', default="ae_tuning_r.csv")
    parser.add_argument('-p', dest='path', default="datax/")
    parser.add_argument('-d', dest='problem', default="yahooR3/")
    parser.add_argument('-se', dest='scene', default='r')  # or 'u' (r: random, u: user)
    parser.add_argument('-t', dest='train', default='S_c.npz')
    parser.add_argument('-v', dest='valid', default='S_va.npz')
    parser.add_argument('-ut', dest='unif_train', default='S_t.npz')
    parser.add_argument('-sr', dest='source', default=None)  # or 'unif'
    parser.add_argument('-m', dest='main_metric', default='AUC')
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=0)
    parser.add_argument('-k', dest='topk', type=check_int_positive, default=50)
    parser.add_argument('-gpu', dest='gpu', action='store_false', default=True)
    args = parser.parse_args()

    main(args)
