import os
import argparse
import numpy as np

from utils.argcheck import ratio
from utils.progress import WorkSplitter
from utils.io import save_numpy, load_pandas_without_names
from utils.split import split_seed_randomly, split_seed_randomly_by_user


def main(args):
    progress = WorkSplitter()
    progress.section("Yahoo R3: Load Raw Data")
    user_rating_matrix = load_pandas_without_names(path=args.path, name=args.problem + args.user, sep=args.sep,
                                                   df_name=args.df_name, value_name='rating')
    random_rating_matrix = load_pandas_without_names(path=args.path, name=args.problem + args.random, sep=args.sep,
                                                     df_name=args.df_name, value_name='rating',
                                                     shape=user_rating_matrix.shape)

    progress.section("Yahoo R3: Split CSR Matrices")
    # For general testing, we use traditional data set splitting
    user_rating_matrix, utrain, uvalid, utest = split_seed_randomly_by_user(rating_matrix=user_rating_matrix,
                                                                            ratio=args.user_ratio,
                                                                            threshold=args.threshold,
                                                                            implicit=args.implicit,
                                                                            remove_empty=args.remove_empty,
                                                                            split_seed=args.seed,
                                                                            sampling=args.sampling,
                                                                            percentage=args.percentage)
    # For unbiased testing, we use random splitting of the full set
    _, rtrain, rvalid, rtest = split_seed_randomly(rating_matrix=random_rating_matrix, ratio=args.random_ratio,
                                                   threshold=args.threshold, implicit=args.implicit,
                                                   split_seed=args.seed)

    progress.section("Yahoo R3: Save NPZ")
    save_dir = args.path + args.problem
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_numpy(utrain, save_dir, "Utrain")
    save_numpy(uvalid, save_dir, "Uvalid")
    save_numpy(utest, save_dir, "Utest")

    save_numpy(user_rating_matrix, save_dir, "S_c")
    save_numpy(rtrain, save_dir, "S_t")
    save_numpy(rvalid, save_dir, "S_va")
    save_numpy(rtest, save_dir, "S_te")

    progress.section("Yahoo R3: Statistics of Data Sets")
    print('* S_c  #num: %6d, pos: %.6f, neg: %.6f' % (
        user_rating_matrix.count_nonzero(), np.sum(user_rating_matrix == 1) / user_rating_matrix.count_nonzero(),
        1 - np.sum(user_rating_matrix == 1) / user_rating_matrix.count_nonzero()))
    print('* S_t  #num: %6d, pos: %.6f, neg: %.6f' % (
        rtrain.count_nonzero(), np.sum(rtrain == 1) / rtrain.count_nonzero(),
        1 - np.sum(rtrain == 1) / rtrain.count_nonzero()))
    print('* S_va #num: %6d, pos: %.6f, neg: %.6f' % (
        rvalid.count_nonzero(), np.sum(rvalid == 1) / rvalid.count_nonzero(),
        1 - np.sum(rvalid == 1) / rvalid.count_nonzero()))
    print('* S_te #num: %6d, pos: %.6f, neg: %.6f' % (
        rtest.count_nonzero(), np.sum(rtest == 1) / rtest.count_nonzero(),
        1 - np.sum(rtest == 1) / rtest.count_nonzero()))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Data Preprocess")
    parser.add_argument('-p', dest='path', default='datax/')
    parser.add_argument('-d', dest='problem', default='yahooR3/')
    parser.add_argument('-user', dest='user', help='user subset', default='user.txt')
    parser.add_argument('-random', dest='random', help='random subset', default='random.txt')
    parser.add_argument('-sep', dest='sep', help='separate', default=',')
    parser.add_argument('-dn', dest='df_name', help='column names of dataframe',
                        default=['uid', 'iid', 'rating'])
    parser.add_argument('-s', dest='seed', help='random seed', type=int, default=0)
    parser.add_argument('-ur', dest='user_ratio', type=ratio, default='0.5,0.2,0.3')
    parser.add_argument('-rr', dest='random_ratio', type=ratio, default='0.05,0.05,0.9')
    parser.add_argument('-threshold', dest='threshold', default=4)
    parser.add_argument('--implicit', dest='implicit', action='store_false', default=True)
    parser.add_argument('--remove_empty', dest='remove_empty', action='store_false', default=True)
    parser.add_argument('--sampling', dest='sampling', action='store_true', default=False)
    parser.add_argument('--percentage', dest='percentage', type=float, default=0.1)
    args = parser.parse_args()
    main(args)
