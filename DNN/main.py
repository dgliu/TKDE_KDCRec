import os
import time
import argparse
import numpy as np
from utils.io import load_numpy
from utils.modelnames import models
from models.predictor import predict
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter, inhour
from utils.argcheck import check_float_positive, check_int_positive


def main(args):
    # Progress bar
    progress = WorkSplitter()

    # Show hyper parameter settings
    progress.section("Parameter Setting")
    print("Data Path: {0}".format(args.path+args.problem))
    print("Train File Name: {0}".format(args.train))
    print("Valid File Name: {0}".format(args.valid))
    print("Algorithm: {0}".format(args.model))
    print("Alpha: {0}".format(args.alpha))
    print("Rank: {0}".format(args.rank))
    print("Lambda: {0}".format(args.lamb))
    print("SVD/Alter Iteration: {0}".format(args.iter))
    print("Evaluation Ranking Topk: {0}".format(args.topk))

    # Load Data
    progress.section("Loading Data")
    start_time = time.time()

    R_train = load_numpy(path=args.path, name=args.problem+args.train)
    R_valid = load_numpy(path=args.path, name=args.problem + args.valid)
    R_rtrain = load_numpy(path=args.path, name=args.problem + args.unif_train)
    print("Elapsed: {0}".format(inhour(time.time() - start_time)))

    print("Train U-I Dimensions: {0}".format(R_train.shape))

    RQ, X, xBias, Yt, yBias = models[args.model](R_train, R_valid, embeded_matrix=np.empty(0), matrix_utrain=R_rtrain,
                                                 iteration=args.iter, rank=args.rank, corruption=args.corruption,
                                                 gpu_on=args.gpu, lam=args.lamb, lam2=args.lamb2, beta=args.beta,
                                                 alpha=args.alpha, seed=args.seed, batch_size=args.batch_size,
                                                 learning_rate=args.learning_rate, root=args.root,
                                                 optimizer=args.optimizer, source=args.source, problem=args.problem,
                                                 scene=args.scene, metric=args.main_metric, topK=args.topk)
    Y = Yt.T

    if not os.path.exists('latent/pretrain/'+args.problem):
        os.makedirs('latent/pretrain/'+args.problem)

    if args.source:
        np.save('latent/pretrain/{2}/{3}_U_{0}_{1}_{4}'.format(args.model, args.rank, args.problem, args.source, args.scene), RQ)
        np.save('latent/pretrain/{2}/{3}_V_{0}_{1}_{4}'.format(args.model, args.rank, args.problem, args.source, args.scene), Y)
        if yBias is not None:
            np.save('latent/pretrain/{2}/{3}_B_{0}_{1}_{4}'.format(args.model, args.rank, args.problem, args.source, args.scene), yBias)
        if X is not None:
            np.save('latent/pretrain/{2}/{3}_X_{0}_{1}_{4}'.format(args.model, args.rank, args.problem, args.source, args.scene), X)
        if xBias is not None:
            np.save('latent/pretrain/{2}/{3}_xB_{0}_{1}_{4}'.format(args.model, args.rank, args.problem, args.source, args.scene), xBias)
    else:
        np.save('latent/pretrain/{2}/U_{0}_{1}_{3}'.format(args.model, args.rank, args.problem, args.scene), RQ)
        np.save('latent/pretrain/{2}/V_{0}_{1}_{3}'.format(args.model, args.rank, args.problem, args.scene), Y)
        if yBias is not None:
            np.save('latent/pretrain/{2}/B_{0}_{1}_{3}'.format(args.model, args.rank, args.problem, args.scene), yBias)
        if X is not None:
            np.save('latent/pretrain/{2}/X_{0}_{1}_{3}'.format(args.model, args.rank, args.problem, args.scene), X)
        if xBias is not None:
            np.save('latent/pretrain/{2}/xB_{0}_{1}_{3}'.format(args.model, args.rank, args.problem, args.scene), xBias)

    progress.section("Predict")
    rating_prediction, topk_prediction = predict(matrix_U=RQ,
                                                 matrix_V=Y,
                                                 bias=yBias,
                                                 topK=args.topk,
                                                 matrix_Train=R_train,
                                                 matrix_Test=R_valid,
                                                 measure=args.sim_measure,
                                                 gpu=args.gpu,
                                                 is_topK=True)
    progress.section("Create Metrics")
    start_time = time.time()

    metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Precision', 'Recall', 'MAP', 'NLL', 'AUC']
    result = evaluate(rating_prediction, topk_prediction, R_valid, metric_names, [args.topk])
    print("-")
    for metric in result.keys():
        print("{0}:{1}".format(metric, result[metric]))
        print("Elapsed: {0}".format(inhour(time.time() - start_time)))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="CRec")

    parser.add_argument('-i', dest='iter', type=check_int_positive, default=100)
    parser.add_argument('-a', dest='alpha', type=check_float_positive, default=1.0)
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=1e-4)
    parser.add_argument('-l2', dest='lamb2', type=check_float_positive, default=0.1)
    parser.add_argument('-b', dest='beta', type=check_float_positive, default=0.2)
    parser.add_argument('-r', dest='rank', type=check_int_positive, default=200)
    parser.add_argument('-f', dest='root', type=check_float_positive, default=1)
    parser.add_argument('-c', dest='corruption', type=check_float_positive, default=0.5)
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=0)
    parser.add_argument('-bs', dest='batch_size', type=check_int_positive, default=512)
    parser.add_argument('-lr', dest='learning_rate', type=check_float_positive, default=1e-3)
    parser.add_argument('-m', dest='model', default="BoundMF2")
    parser.add_argument('-p', dest='path', default="datax/")
    parser.add_argument('-d', dest='problem', default="yahooR3/")
    parser.add_argument('-se', dest='scene', default='r')  # or 'u' (r: random, u: user)
    parser.add_argument('-t', dest='train', default='S_c.npz')
    parser.add_argument('-v', dest='valid', default='S_va.npz')
    parser.add_argument('-ut', dest='unif_train', default='S_t.npz')
    parser.add_argument('-mm', dest='main_metric', default='AUC')
    parser.add_argument('-sr', dest='source', default=None)  # or 'unif' or 'combine'
    parser.add_argument('-k', dest='topk', type=check_int_positive, default=50)
    parser.add_argument('-gpu', dest='gpu', action='store_false', default=True)
    parser.add_argument('--similarity', dest='sim_measure', default='Cosine')
    parser.add_argument('-o', dest='optimizer', default='Adam')
    args = parser.parse_args()

    main(args)
