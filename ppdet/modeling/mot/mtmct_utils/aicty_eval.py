#!/usr/bin/python3
"""
Evaluate submissions for the AI City Challenge.
"""
import numpy as np
import pandas as pd
import motmetrics as mm
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

names = ['CameraId','Id', 'FrameId', 'X', 'Y', 'Width', 'Height', 'Xworld', 'Yworld']
mh = mm.metrics.create()
def getData(fpath, names=None, sep='\s+|\t+|,'):
    """ Get the necessary track data from a file handle.
    Params
    ------
    fpath : str
        Original path of file reading from.
    names : list<str>
        List of column names for the data.
    sep : str
        Allowed separators regular expression string.
    Returns
    -------
    df : pandas.DataFrame
        Data frame containing the data loaded from the stream with optionally assigned column names.
        No index is set on the data.
    """
    try:
        df = pd.read_csv(
            fpath, 
            sep=sep, 
            index_col=None, 
            skipinitialspace=True, 
            header=None,
            names=names,
            engine='python'
        )
        
        return df
    
    except Exception as e:
        raise ValueError("Could not read input from %s. Error: %s" % (fpath, repr(e)))


def print_results(summary, mread=False):
    """Print a summary dataframe in a human- or machine-readable format.
    
    Params
    ------
    summary : pandas.DataFrame
        Data frame of evaluation results in motmetrics format.
    mread : bool
        Whether to print results in machine-readable format (JSON).
    Returns
    -------
    None
        Prints results to screen.
    """
    if mread:
        print('{"results":%s}' % summary.iloc[-1].to_json())
        return
    
    formatters = {'idf1': '{:2.2f}'.format,
                  'idp': '{:2.2f}'.format,
                  'idr': '{:2.2f}'.format}
    
    summary = summary[['idf1','idp','idr']]
    # summary['idp'] *= 100
    # summary['idr'] *= 100
    # summary['idf1'] *= 100

    summary.loc[:,'idp'] *= 100
    summary.loc[:,'idr'] *= 100
    summary.loc[:,'idf1'] *= 100


    print(mm.io.render_summary(summary, formatters=formatters, namemap=mm.io.motchallenge_metric_names))
    return

def compare_dataframes_mtmc(gts, ts):
    """Compute ID-based evaluation metrics for multi-camera multi-object tracking.
    
    Params
    ------
    gts : pandas.DataFrame
        Ground truth data.
    ts : pandas.DataFrame
        Prediction/test data.
    Returns
    -------
    df : pandas.DataFrame
        Results of the evaluations in a df with only the 'idf1', 'idp', and 'idr' columns.
    """
    gtds = []
    tsds = []
    gtcams = gts['CameraId'].drop_duplicates().tolist()
    tscams = ts['CameraId'].drop_duplicates().tolist()
    maxFrameId = 0;

    for k in sorted(gtcams):
        gtd = gts.query('CameraId == %d' % k)
        gtd = gtd[['FrameId', 'Id', 'X', 'Y', 'Width', 'Height']]
        # max FrameId in gtd only
        mfid = gtd['FrameId'].max()
        gtd['FrameId'] += maxFrameId
        gtd = gtd.set_index(['FrameId', 'Id'])
        gtds.append(gtd)

        if k in tscams:
            tsd = ts.query('CameraId == %d' % k)
            tsd = tsd[['FrameId', 'Id', 'X', 'Y', 'Width', 'Height']]
            # max FrameId among both gtd and tsd
            mfid = max(mfid, tsd['FrameId'].max())
            tsd['FrameId'] += maxFrameId
            tsd = tsd.set_index(['FrameId', 'Id'])
            tsds.append(tsd)

        maxFrameId += mfid

    # compute multi-camera tracking evaluation stats
    multiCamAcc = mm.utils.compare_to_groundtruth(pd.concat(gtds), pd.concat(tsds), 'iou')
    metrics=list(mm.metrics.motchallenge_metrics)
    metrics.extend(['num_frames','idfp','idfn','idtp'])
    summary = mh.compute(multiCamAcc, metrics=metrics, name='MultiCam')
    return summary

def print_mtmct_result(gt_file, pred_file):
    gt = getData(gt_file, names=names)
    pred = getData(pred_file, names=names)
    summary = compare_dataframes_mtmc(gt, pred)
    print_results(summary)

if __name__ == "__main__":
    gt = getData('/paddle/VscodeWorkspace/Experiment/MCMT/dataset/mtmct/S01/gt/gt.txt', names=names)
    pred = getData('/paddle/VscodeWorkspace/Experiment/MCMT/output/mtmct_result.txt', names=names)
    summary = compare_dataframes_mtmc(gt, pred)
    # print_results(summary)
    # print(summary)





