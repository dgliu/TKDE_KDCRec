import os
import numpy as np
import pandas as pd
from models.predictor import predict
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter
from utils.io import load_dataframe_csv, save_dataframe_csv, load_yaml


def hyper_parameter_tuning(train, validation, rtrain, params, dataset, save_path, source=None, problem=None,
                           measure='Cosine', seed=0, gpu_on=False, scene='r', metric='AUC'):
    progress = WorkSplitter()
    table_path = load_yaml('config/global.yml', key='path')['tables']

    if not os.path.exists(table_path+save_path):
        if not os.path.exists(table_path+dataset):
            os.makedirs(table_path+dataset)
        df = pd.DataFrame(columns=['model', 'rank', 'alpha', 'lambda', 'lambda2', 'iter', 'mode', 'confidence',
                                   'batch_size', 'step', 'source'])
    else:
        df = load_dataframe_csv(table_path, save_path)

    for algorithm in params['models']:

        for rank in params['rank']:

            for alpha in params['alpha']:

                for lam in params['lambda']:

                    for lam2 in params['lambda2']:

                        for confidence in params['confidence']:

                            for batch_size in params['batch_size']:

                                for step in params['step']:

                                    if ((df['model'] == algorithm) &
                                        (df['rank'] == rank) &
                                        (df['alpha'] == alpha) &
                                        (df['lambda'] == lam) &
                                        (df['lambda2'] == lam2) &
                                        (df['confidence'] == confidence) &
                                        (df['batch_size'] == batch_size) &
                                       (df['step'] == step)).any():
                                        continue

                                    format = "model: {0}, rank: {1}, alpha: {2}, lambda: {3}, lambda2: {4}, " \
                                             "confidence: {5}, batch_size: {6}, step: {7}"
                                    progress.section(format.format(algorithm, rank, alpha, lam, lam2, confidence,
                                                                   batch_size, step))
                                    RQ, X, xBias, Yt, yBias = params['models'][algorithm](train,
                                                                                          validation,
                                                                                          embeded_matrix=np.empty(0),
                                                                                          matrix_utrain=rtrain,
                                                                                          iteration=params['iter'],
                                                                                          rank=rank,
                                                                                          batch_size=batch_size,
                                                                                          lam=lam,
                                                                                          lam2=lam2,
                                                                                          alpha=alpha,
                                                                                          confidence=confidence,
                                                                                          step=step,
                                                                                          mode=params['mode'],
                                                                                          seed=seed,
                                                                                          source=source,
                                                                                          problem=problem,
                                                                                          gpu_on=gpu_on,
                                                                                          scene=scene,
                                                                                          metric=metric,
                                                                                          topK=params['topK'][-1])
                                    Y = Yt.T

                                    progress.subsection("Prediction")

                                    rating_prediction, topk_prediction = predict(matrix_U=RQ, matrix_V=Y,
                                                                                 measure=measure,
                                                                                 bias_U=xBias, bias_V=yBias,
                                                                                 topK=params['topK'][-1],
                                                                                 matrix_Train=train,
                                                                                 matrix_Test=validation)

                                    progress.subsection("Evaluation")

                                    result = evaluate(rating_prediction, topk_prediction, validation, params['metric'],
                                                      params['topK'])

                                    result_dict = {'model': algorithm, 'rank': rank, 'alpha': alpha, 'lambda': lam,
                                                   'lambda2': lam2, 'iter': params['iter'], 'mode': params['mode'],
                                                   'confidence': confidence, 'batch_size': batch_size, 'step': step,
                                                   'source': source}

                                    for name in result.keys():
                                        result_dict[name] = [round(result[name][0], 5), round(result[name][1], 5)]

                                    df = df.append(result_dict, ignore_index=True)

                                    save_dataframe_csv(df, table_path, save_path)
