import argparse
from utils.modelnames import models
from utils.io import load_numpy, load_yaml
from utils.argcheck import check_int_positive
from experiment.tuning import hyper_parameter_tuning


def main(args):
    params = load_yaml(args.grid)
    params['models'] = {params['models']: models[params['models']]}
    R_train = load_numpy(path=args.path, name=args.problem+args.train)
    R_valid = load_numpy(path=args.path, name=args.problem+args.valid)
    R_rtrain = load_numpy(path=args.path, name=args.problem+args.unif_train)
    hyper_parameter_tuning(R_train, R_valid, R_rtrain, params, dataset=args.problem, source=args.source,
                           save_path=args.problem+args.table_name, seed=args.seed, problem=args.problem,
                           gpu_on=args.gpu, scene=args.scene, metric=args.main_metric)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="ParameterTuning")
    parser.add_argument('-tb', dest='table_name', default="mf_tuning_r.csv")
    parser.add_argument('-p', dest='path', default="datax/")
    parser.add_argument('-d', dest='problem', default="yahooR3/")
    parser.add_argument('-se', dest='scene', default='r')  # or 'u' (r: random, u: user)
    parser.add_argument('-t', dest='train', default='S_c.npz')
    parser.add_argument('-v', dest='valid', default='S_va.npz')
    parser.add_argument('-ut', dest='unif_train', default='S_t.npz')
    parser.add_argument('-m', dest='main_metric', default='AUC')
    parser.add_argument('-sr', dest='source', default=None)  # or 'unif' or 'combine'
    parser.add_argument('-y', dest='grid', default='config/mf.yml')
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=0)
    parser.add_argument('-gpu', dest='gpu', action='store_false', default=True)
    args = parser.parse_args()

    main(args)
