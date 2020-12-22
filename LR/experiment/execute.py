import os
import math
import numpy as np
import pandas as pd
from models.predictor import predict
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter


def execute(train, validation, test, rtrain, params, model, source=None, mode='none', problem=None, measure='Cosine',
            gpu_on=True, analytical=False, folder='latent', scene='r'):
    progress = WorkSplitter()

    columns = ['model', 'rank', 'alpha', 'lambda', 'lambda2', 'iter', 'mode', 'confidence',
               'batch_size', 'step', 'source']

    progress.section("\n".join([":".join((str(k), str(params[k]))) for k in columns]))

    df = pd.DataFrame(columns=columns)

    if not os.path.exists(folder):
        os.makedirs(folder)

    if isinstance(source, str):
        if os.path.isfile('{2}/{3}_U_{0}_{1}_{4}.npy'.format(params['model'], params['rank'], folder, source, scene)):

            RQ = np.load('{2}/{3}_U_{0}_{1}_{4}.npy'.format(params['model'], params['rank'], folder, source, scene))
            Y = np.load('{2}/{3}_V_{0}_{1}_{4}.npy'.format(params['model'], params['rank'], folder, source, scene))

            if os.path.isfile(
                    '{2}/{3}_vB_{0}_{1}_{4}.npy'.format(params['model'], params['rank'], folder, source, scene)):
                yBias = np.load(
                    '{2}/{3}_vB_{0}_{1}_{4}.npy'.format(params['model'], params['rank'], folder, source, scene))
            else:
                yBias = None

            if os.path.isfile(
                    '{2}/{3}_uB_{0}_{1}_{4}.npy'.format(params['model'], params['rank'], folder, source, scene)):
                xBias = np.load(
                    '{2}/{3}_uB_{0}_{1}_{4}.npy'.format(params['model'], params['rank'], folder, source, scene))
            else:
                xBias = None
        else:
            RQ, X, xBias, Yt, yBias = model(train,
                                            validation,
                                            embeded_matrix=np.empty(0),
                                            matrix_utrain=rtrain,
                                            iteration=params['iter'],
                                            rank=params['rank'],
                                            batch_size=params['batch_size'],
                                            lam=params['lambda'],
                                            lam2=params['lambda2'],
                                            alpha=params['alpha'],
                                            confidence=params['confidence'],
                                            mode=params['mode'],
                                            step=params['step'],
                                            source=source,
                                            problem=problem,
                                            gpu_on=gpu_on,
                                            scene=scene)
            Y = Yt.T
            np.save('{2}/{3}_U_{0}_{1}_{4}'.format(params['model'], params['rank'], folder, source, scene), RQ)
            np.save('{2}/{3}_V_{0}_{1}_{4}'.format(params['model'], params['rank'], folder, source, scene), Y)
            if yBias is not None:
                np.save('{2}/{3}_vB_{0}_{1}_{4}'.format(params['model'], params['rank'], folder, source, scene), yBias)
            if X is not None:
                np.save('{2}/{3}_X_{0}_{1}_{4}'.format(params['model'], params['rank'], folder, source, scene), X)
            if xBias is not None:
                np.save('{2}/{3}_uB_{0}_{1}_{4}'.format(params['model'], params['rank'], folder, source, scene), xBias)

    else:
        if mode == 'none':
            if os.path.isfile('{2}/U_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene)):

                RQ = np.load('{2}/U_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene))
                Y = np.load('{2}/V_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene))

                if os.path.isfile('{2}/vB_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene)):
                    yBias = np.load('{2}/vB_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene))
                else:
                    yBias = None

                if os.path.isfile('{2}/uB_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene)):
                    xBias = np.load('{2}/uB_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene))
                else:
                    xBias = None
            else:
                RQ, X, xBias, Yt, yBias = model(train,
                                                validation,
                                                embeded_matrix=np.empty(0),
                                                matrix_utrain=rtrain,
                                                iteration=params['iter'],
                                                rank=params['rank'],
                                                batch_size=params['batch_size'],
                                                lam=params['lambda'],
                                                lam2=params['lambda2'],
                                                alpha=params['alpha'],
                                                confidence=params['confidence'],
                                                mode=params['mode'],
                                                step=params['step'],
                                                source=source,
                                                problem=problem,
                                                gpu_on=gpu_on,
                                                scene=scene)
                Y = Yt.T

                np.save('{2}/U_{0}_{1}_{3}'.format(params['model'], params['rank'], folder, scene), RQ)
                np.save('{2}/V_{0}_{1}_{3}'.format(params['model'], params['rank'], folder, scene), Y)
                if yBias is not None:
                    np.save('{2}/vB_{0}_{1}_{3}'.format(params['model'], params['rank'], folder, scene), yBias)
                if X is not None:
                    np.save('{2}/X_{0}_{1}_{3}'.format(params['model'], params['rank'], folder, scene), X)
                if xBias is not None:
                    np.save('{2}/uB_{0}_{1}_{3}'.format(params['model'], params['rank'], folder, scene), xBias)
        else:
            if os.path.isfile('{2}/{4}_U_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene, mode)):

                RQ = np.load('{2}/{4}_U_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene, mode))
                Y = np.load('{2}/{4}_V_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene, mode))

                if os.path.isfile('{2}/{4}_vB_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene, mode)):
                    yBias = np.load('{2}/{4}_vB_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene, mode))
                else:
                    yBias = None

                if os.path.isfile('{2}/{4}_uB_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene, mode)):
                    xBias = np.load('{2}/{4}_uB_{0}_{1}_{3}.npy'.format(params['model'], params['rank'], folder, scene, mode))
                else:
                    xBias = None
            else:
                RQ, X, xBias, Yt, yBias = model(train,
                                                validation,
                                                embeded_matrix=np.empty(0),
                                                matrix_utrain=rtrain,
                                                iteration=params['iter'],
                                                rank=params['rank'],
                                                batch_size=params['batch_size'],
                                                lam=params['lambda'],
                                                lam2=params['lambda2'],
                                                alpha=params['alpha'],
                                                confidence=params['confidence'],
                                                mode=params['mode'],
                                                step=params['step'],
                                                source=source,
                                                problem=problem,
                                                gpu_on=gpu_on,
                                                scene=scene)
                Y = Yt.T

                np.save('{2}/{4}_U_{0}_{1}_{3}'.format(params['model'], params['rank'], folder, scene, mode), RQ)
                np.save('{2}/{4}_V_{0}_{1}_{3}'.format(params['model'], params['rank'], folder, scene, mode), Y)
                if yBias is not None:
                    np.save('{2}/{4}_vB_{0}_{1}_{3}'.format(params['model'], params['rank'], folder, scene, mode), yBias)
                if X is not None:
                    np.save('{2}/{4}_X_{0}_{1}_{3}'.format(params['model'], params['rank'], folder, scene, mode), X)
                if xBias is not None:
                    np.save('{2}/{4}_uB_{0}_{1}_{3}'.format(params['model'], params['rank'], folder, scene, mode), xBias)

    progress.subsection("Prediction")

    rating_prediction, topk_prediction = predict(matrix_U=RQ, matrix_V=Y, measure=measure, bias_U=xBias, bias_V=yBias,
                                                 topK=params['topK'][-1], matrix_Train=train, matrix_Test=test,
                                                 gpu=gpu_on, is_topK=False)

    progress.subsection("Evaluation")

    result = evaluate(rating_prediction, topk_prediction, test, params['metric'], params['topK'], analytical=analytical,
                      is_topK=False)

    if analytical:
        return result
    else:
        result_dict = params

        for name in result.keys():
            result_dict[name] = [round(result[name][0], 4), round(result[name][1], 4)]
        df = df.append(result_dict, ignore_index=True)

        return df
