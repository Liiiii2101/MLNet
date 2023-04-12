import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.metrics as metrics
import tqdm


def sample_size(y_true, groups=None):
    # create one group per sample if no group provided
    if groups is None:
        groups = np.arange(y_true.size)
    # count groups with posive and negative samples
    g_pos = np.sum([np.any(y_true[groups == g] == 1) for g in np.unique(groups)])
    return g_pos, np.unique(groups).size - g_pos


def safe_rocauc(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred) if y_true.std() > 0 else np.nan


def safe_f1score(y_true, y_pred):
    if y_true.std() == 0: return np.nan
    return metrics.f1_score(y_true, y_pred)


def safe_prc(y_true, y_pred):
    if y_true.std() == 0: return np.nan
    return metrics.average_precision_score(y_true, y_pred)


def safe_pval(y_true, y_pred):
    return stats.mannwhitneyu(y_pred[y_true == 1], y_pred[y_true < 1]).pvalue if y_true.std() > 0 else np.nan


def perf_measure(y_actual, y_hat):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            tp += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            fp += 1
        if y_actual[i] == y_hat[i] == 0:
            tn += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            fn += 1
    return tp, fp, tn, fn


def classification_analysis(y_true, y_pred, groups=None, repeats=10000, CI=95, th=0.5):
    # create one group per sample if no group provided
    g = np.arange(y_true.shape[0])#np.arange(y_true.size) if groups is None else groups
    # arrange in dataframe
    df = pd.DataFrame(np.array([y_true, y_pred, g]).T, columns=['y_true', 'y_pred', 'g'])
    df = df.astype({'y_true': 'int', 'y_pred': 'float'})
    pos, neg = sample_size(y_true, None)
    r = pos / (pos + neg)
    # bootstrap
    auc, acc, prc, sens, spec, f1, ppv, npv = ([] for i in range(8))
    eps = 0.000001
    for random_state in tqdm.tqdm(range(repeats)):
        # sampling
        if groups is not None:  # seep up
            df_ = df.groupby('g').apply(lambda x: x.sample(n=1, random_state=random_state))
        else:
            df_ = df.sample(frac=1., replace=True, random_state=random_state)
        # testing
        auc.append(safe_rocauc(df_['y_true'].values, df_['y_pred'].values))
        prc.append(safe_prc(df_['y_true'].values, df_['y_pred'].values))
        f1.append(safe_f1score(df_['y_true'].values, np.asarray([(x > th) * 1 for x in df_['y_pred'].values])))
        tp, fp, tn, fn = perf_measure(df_['y_true'].values, np.asarray([(x > th) * 1 for x in df_['y_pred'].values]))
        acc.append(np.round((tp + tn) / (tp + tn + fp + fn), 2))
        sens.append(np.round(tp / (tp + fn + eps), 2))
        spec.append(np.round(tn / (tn + fp + eps), 2))
        ppv.append(np.round(tp / (tp + fp + eps), 2))
        npv.append(np.round(tn / (tn + fn + eps), 2))

    d = {
        'rocauc': "{:1.2f}".format(np.nanpercentile(auc, 50)) + ' (' +
                  "{:1.2f}".format(np.nanpercentile(auc, 50 - (CI / 2))) + ' - ' +
                  "{:1.2f}".format(np.nanpercentile(auc, 50 + (CI / 2))) + ')',
        'prcauc': "{:1.2f}".format(np.nanpercentile(prc, 50)) + ' (' +
                  "{:1.2f}".format(np.nanpercentile(prc, 50 - (CI / 2))) + ' - ' +
                  "{:1.2f}".format(np.nanpercentile(prc, 50 + (CI / 2))) + '; ' +
                  "{:1.2f}".format(r) + ')',
        'f1_score': "{:1.2f}".format(np.nanpercentile(f1, 50)) + '(' +
                    "{:1.2f}".format(np.nanpercentile(f1, 50 - (CI / 2))) + ' - ' +
                    "{:1.2f}".format(np.nanpercentile(f1, 50 + (CI / 2))) + ')',
        'accuracy': "{:1.2f}".format(np.nanpercentile(acc, 50)) + ' (' +
                    "{:1.2f}".format(np.nanpercentile(acc, 50 - (CI / 2))) + ' - ' +
                    "{:1.2f}".format(np.nanpercentile(acc, 50 + (CI / 2))) + ')',
        'sensitivity': "{:1.2f}".format(np.nanpercentile(sens, 50)) + ' (' +
                       "{:1.2f}".format(np.nanpercentile(sens, 50 - (CI / 2))) + ' - ' +
                       "{:1.2f}".format(np.nanpercentile(sens, 50 + (CI / 2))) + ')',
        'specificity': "{:1.2f}".format(np.nanpercentile(spec, 50)) + ' (' +
                       "{:1.2f}".format(np.nanpercentile(spec, 50 - (CI / 2))) + ' - ' +
                       "{:1.2f}".format(np.nanpercentile(spec, 50 + (CI / 2))) + ')',
        'ppv': "{:1.2f}".format(np.nanpercentile(ppv, 50)) + ' (' +
               "{:1.2f}".format(np.nanpercentile(ppv, 50 - (CI / 2))) + ' - ' +
               "{:1.2f}".format(np.nanpercentile(ppv, 50 + (CI / 2))) + ')',
        'npv': "{:1.2f}".format(np.nanpercentile(npv, 50)) + ' (' +
               "{:1.2f}".format(np.nanpercentile(npv, 50 - (CI / 2))) + ' - ' +
               "{:1.2f}".format(np.nanpercentile(npv, 50 + (CI / 2))) + ')',
        'pval': safe_pval(y_true, y_pred),
        'n_pos': pos,
        'n_neg': neg
    }
    return d


def segment_analysis(paths_true: list, paths_pred: list, repeats=10000, CI=95):
    import SimpleITK as sitk
    import os
    df = pd.DataFrame(columns=['ID', 'TP', 'FP', 'FN', 'SENS', 'DSC', 'TP_vol', 'FP_vol', 'FN_vol', 'AE_vol'])
    for i, (p_true, p_pred) in enumerate(zip(tqdm.tqdm(paths_true), paths_pred)):
        img_label = sitk.ReadImage(p_true)
        arr_label = sitk.GetArrayFromImage(img_label)
        arr_pred = sitk.GetArrayFromImage(sitk.ReadImage(p_pred))

        voxel_sixe = np.prod(img_label.GetSpacing()) / 1000  # voxel size in cm³ (spacing in mm³)
        TP = np.sum(arr_pred * arr_label)
        FP = np.count_nonzero(arr_pred * 1 - arr_label * 1 == 1)
        FN = np.count_nonzero(arr_label * 1 - arr_pred * 1 == 1)
        TP_vol, FP_vol, FN_vol = (item * voxel_sixe for item in [TP, FP, FN])

        if (TP + FN) != 0:
            SENS = TP / (TP + FN)
            DSC = (2 * TP) / (2 * TP + FP + FN)
        else:
            SENS = np.nan
            DSC = np.nan

        df.at[i, 'ID'] = p_true.split(os.sep)[-1].split('.')[0]
        df.at[i, 'TP'] = TP
        df.at[i, 'FP'] = FP
        df.at[i, 'FN'] = FN
        df.at[i, 'DSC'] = np.round(DSC, 3)
        df.at[i, 'SENS'] = np.round(SENS, 3)

        df.at[i, 'TP_vol'] = np.round(TP_vol, 2)
        df.at[i, 'FP_vol'] = np.round(FP_vol, 2)
        df.at[i, 'FN_vol'] = np.round(FN_vol, 2)

        df.at[i, 'AE_vol'] = np.round(np.abs(((TP_vol + FP_vol) - (TP_vol + FN_vol))), 2)
        df.at[i, 'Label_vol'] = np.sum(arr_label.flatten()) * voxel_sixe
        df.at[i, 'Pred_vol'] = np.sum(arr_pred.flatten()) * voxel_sixe

    # bootstrap
    # sens and dice is calculated over all individual scans, m(icro)dice and m(icro)sens is calculated by
    # concatenating all scans to one volume. Difference between these are likely due to substantial differences in
    # ground-truth volumes.
    sens, msens, dice, mdice, labelvol, predvol, mae, corr = ([] for i in range(8))
    for random_state in tqdm.tqdm(range(repeats)):
        # sampling
        df_ = df.sample(frac=1., replace=True, random_state=random_state)
        # testing
        dice.append(df_['DSC'].median())
        sens.append(df_['SENS'].median())
        mae.append(df_['AE_vol'].median())
        predvol.append(np.median(df_['TP_vol'] + df_['FP_vol']))
        labelvol.append(np.median(df_['TP_vol'] + df_['FN_vol']))
        corr.append(stats.spearmanr(df_['TP_vol'] + df_['FP_vol'], df_['TP_vol'] + df_['FN_vol']))
        tp, fp, fn = df_['TP_vol'].sum(), df_['FP_vol'].sum(), df_['FN_vol'].sum(),
        if (tp + fn) != 0:  # no ground truth, dice and sens not defined
            msens.append(np.round(tp / (tp + fn), 2))
            mdice.append(np.round(2 * tp / (2 * tp + fp + fn), 2))
        else:
            msens.append(np.nan)
            mdice.append(np.nan)
    d = {
        'DICE': "{:1.2f}".format(np.nanpercentile(dice, 50)) + '(' +
                "{:1.2f}".format(np.nanpercentile(dice, 50 - (CI / 2))) + ' - ' +
                "{:1.2f}".format(np.nanpercentile(dice, 50 + (CI / 2))) + ')',
        'mDICE': "{:1.2f}".format(np.nanpercentile(mdice, 50)) + '(' +
                 "{:1.2f}".format(np.nanpercentile(mdice, 50 - (CI / 2))) + ' - ' +
                 "{:1.2f}".format(np.nanpercentile(mdice, 50 + (CI / 2))) + ')',
        'SENS': "{:1.2f}".format(np.nanpercentile(sens, 50)) + ' (' +
                "{:1.2f}".format(np.nanpercentile(sens, 50 - (CI / 2))) + ' - ' +
                "{:1.2f}".format(np.nanpercentile(sens, 50 + (CI / 2))) + ')',
        'mSENS': "{:1.2f}".format(np.nanpercentile(msens, 50)) + ' (' +
                 "{:1.2f}".format(np.nanpercentile(msens, 50 - (CI / 2))) + ' - ' +
                 "{:1.2f}".format(np.nanpercentile(msens, 50 + (CI / 2))) + ')',
        'Label Volume': "{:1.2f}".format(np.nanpercentile(labelvol, 50)) + ' (' +
                        "{:1.2f}".format(np.nanpercentile(labelvol, 50 - (CI / 2))) + ' - ' +
                        "{:1.2f}".format(np.nanpercentile(labelvol, 50 + (CI / 2))) + ')',
        'Pred Volume': "{:1.2f}".format(np.nanpercentile(predvol, 50)) + ' (' +
                       "{:1.2f}".format(np.nanpercentile(predvol, 50 - (CI / 2))) + ' - ' +
                       "{:1.2f}".format(np.nanpercentile(predvol, 50 + (CI / 2))) + ')',
        'Absolute Error Volume': "{:1.2f}".format(np.nanpercentile(mae, 50)) + ' (' +
                                 "{:1.2f}".format(np.nanpercentile(mae, 50 - (CI / 2))) + ' - ' +
                                 "{:1.2f}".format(np.nanpercentile(mae, 50 + (CI / 2))) + ')',
    }
    return df, d