import timeit
import argparse
import pandas as pd
from utils.modelnames import models
from experiment.execute import execute
from utils.io import load_numpy, save_dataframe_csv, find_best_hyperparameters, load_yaml


def main(args):
    table_path = load_yaml('config/global.yml', key='path')['tables']

    df = find_best_hyperparameters(table_path+args.problem, args.main_metric, args.scene)

    R_train = load_numpy(path=args.path, name=args.problem+args.train)
    R_valid = load_numpy(path=args.path, name=args.problem + args.valid)
    R_test = load_numpy(path=args.path, name=args.problem+args.test)
    R_rtrain = load_numpy(path=args.path, name=args.problem + args.unif_train)

    topK = [5, 10, 15, 20, 50]

    frame = []
    for idx, row in df.iterrows():
        start = timeit.default_timer()
        row = row.to_dict()
        row['metric'] = ['NLL', 'AUC']
        row['topK'] = topK
        result = execute(R_train, R_valid, R_test, R_rtrain, row, models[row['model']], source=row['source'],
                         mode=row['mode'], problem=args.problem, gpu_on=args.gpu, folder=args.model_folder+args.problem,
                         scene=args.scene)
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        frame.append(result)

    results = pd.concat(frame)
    save_dataframe_csv(results, table_path, args.problem+args.table_name)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Reproduce")
    parser.add_argument('-tb', dest='table_name', default="final_result_r.csv")
    parser.add_argument('-p', dest='path', default="datax/")
    parser.add_argument('-d', dest='problem', default="yahooR3/")
    parser.add_argument('-s', dest='scene', default='r')  # or 'u' (r: random, u: user)
    parser.add_argument('-t', dest='train', default='S_c.npz')
    parser.add_argument('-v', dest='valid', default='S_va.npz')
    parser.add_argument('-e', dest='test', default='S_te.npz')
    parser.add_argument('-ut', dest='unif_train', default='S_t.npz')
    parser.add_argument('-m', dest='main_metric', default='AUC')
    parser.add_argument('-mf', dest='model_folder', default='latent/')  # Model saving folder
    parser.add_argument('-gpu', dest='gpu', action='store_false', default=True)
    args = parser.parse_args()

    main(args)
