import numpy as np
import json
import pandas as pd

def compute_data_kappa(filename, labelname):
    jsondata = readjson(filename)
    datas = readdata(jsondata, labelname)
    if len(datas.values()) != 2:
        print("Not enough! Skipped... " + labelname)
        return None
    a, b = datas.values()
    if ( len(a) >= len(b)):
        piner, amy = b, a
    else:
        piner, amy = a, b

    amy_midpoints = get_midpoint(amy)
    # piner_labels = align_midpoint_to_piner(amy_midpoints)
    piner_f = piner_fac(piner)
    piner_labels = {t: piner_f(t) for t in amy_midpoints}

    matrix = transfer_to_matrix([amy_midpoints, piner_labels])
    """         a, b, c
    [a, b] ==> [1, 1, 0]
    [a, c] ==> [1, 0, 1]
    """
    kappa = kappa_function(matrix)
    return kappa


# amy_midpoints = get_midpoint(amy)
def get_midpoint(amy):
    """input list of triples, output midpoint to label."""
    result = {}
    for start, end, label in amy:
        midpoint = (start + end) / 2
        result[midpoint] = label
    return result





def readdata(jsondata, labelname):
    return {i:
        jsondata[i]["raw_entrylist"]
        for i in jsondata
        if i.startswith(labelname)
    }
    # 高虹安_教育文化_speech_constant_piner.TextGrid": {


def piner_fac(piner):
    def align_midpoint_to_piner_time(time):
        for start, end, label in piner:
            if start <= time <= end:
                return label
        else:
            return "-"

    return align_midpoint_to_piner_time

def transfer_to_matrix(labels):
    piner_labels, amy_midpoints = labels
    df_labels = pd.DataFrame([piner_labels, amy_midpoints]).T
    df_labels.columns = ["piner_labels", "amy_midpoints"]
    cats = sorted(set(df_labels.values.reshape(-1)))
    df = pd.DataFrame([({
        "timestamp": row.Index,
        **{cat: [row.piner_labels, row.amy_midpoints].count(cat) for cat in cats}
    })
    for row in df_labels.itertuples()
    ]    )
    
    df =  df.set_index('timestamp')
    return df.values

def compute_fleiss_kappa(
    rate_list: np.array,
    n: int = None,  # annotators
) -> float:
    verbose = False
    rate_list = np.array(rate_list)
    guessed_annotators = np.unique(rate_list.sum(axis=1)).item()

    if n is not None:
        assert n == guessed_annotators, "Weird number of annotators!"
    else:
        if verbose:
            print(f"(No `n` provided! Inferred from data...)")
        n = guessed_annotators


    assert rate_list.ndim == 2, "Rating table should be 2-dim..."
    (N, k) = (subjects, categories) = rate_list.shape

    if verbose:
        # 入力された情報の確認
        print(f'Number of annotators = {BLUE(n)}')
        print(f'Number of subjects   = {BLUE(N)}')
        print(f'Number of categories = {BLUE(k)}')
        print(f"{'=' * 40} 計算はじめ！ {'=' * 40}" )

    # Piの値を求めて，P_barを求める
    P_i = (((rate_list ** 2).sum(1) - n) / (n * (n - 1)))
    if verbose: print( f"P_i    = \n{GREEN(repr(P_i))}")
    P_bar = P_i.mean()
    if verbose: print(f"P_bar  = {GREEN(P_bar)}")

    # pjの値を求めて，Pe_barを求める
    pj = (rate_list.sum(0) / (N * n))
    if verbose: print(f"pj     = {GREEN(repr(pj))}")
    Pe_bar = (pj ** 2).sum()
    if verbose: print(f"Pe_bar = {GREEN(Pe_bar)}")
    
    if verbose: print(f"{'=' * 44} 結果 {'=' * 44}" )
    # fleiss kappa値の計算
    kappa = 1. if np.allclose(Pe_bar, 1) else (
        (P_bar - Pe_bar) / (1 - Pe_bar)
    )
    if verbose: print(f"kappa  = {RED(kappa)}")
    return kappa

# ========== つまんないこと（てへぺろ） ========== #
def RED(text):   return f"\033[01;31m{text}\033[0m"
def BLUE(text):  return f"\033[01;34m{text}\033[0m"
def GREEN(text): return f"\033[01;32m{text}\033[0m"

kappa_function = compute_fleiss_kappa


def readjson(jsonfile):
    with open(jsonfile) as f:
        jsondata = json.load(f)
        return jsondata
    

DATAFILE = "/Users/chenjiancheng/Desktop/MAIKA/entry_metadata.json"
# kappa = compute_data_kappa(DATAFILE, "高虹安_教育文化_speech_constant")

KAPPAS = {
"張其祿_交通_face": compute_data_kappa(DATAFILE, "張其祿_交通_face"),
"張其祿_交通_gesture_function": compute_data_kappa(DATAFILE, "張其祿_交通_gesture_function"),
"張其祿_交通_hand": compute_data_kappa(DATAFILE, "張其祿_交通_hand"),
"張其祿_交通_head": compute_data_kappa(DATAFILE, "張其祿_交通_head"),
"張其祿_交通_speech_constant": compute_data_kappa(DATAFILE, "張其祿_交通_speech_constant"),
"張其祿_財政_face": compute_data_kappa(DATAFILE, "張其祿_財政_face"),
"張其祿_財政_gesture_function": compute_data_kappa(DATAFILE, "張其祿_財政_gesture_function"),
"張其祿_財政_hand": compute_data_kappa(DATAFILE, "張其祿_財政_hand"),
"張其祿_財政_head": compute_data_kappa(DATAFILE, "張其祿_財政_head"),
"張其祿_財政_speech_constant": compute_data_kappa(DATAFILE, "張其祿_財政_speech_constant"),
"楊瓊瓔_教育文化_face": compute_data_kappa(DATAFILE, "楊瓊瓔_教育文化_face"),
"楊瓊瓔_教育文化_gesture_function": compute_data_kappa(DATAFILE, "楊瓊瓔_教育文化_gesture_function"),
"楊瓊瓔_教育文化_hand": compute_data_kappa(DATAFILE, "楊瓊瓔_教育文化_hand"),
"楊瓊瓔_教育文化_head": compute_data_kappa(DATAFILE, "楊瓊瓔_教育文化_head"),
"楊瓊瓔_教育文化_speech_constant": compute_data_kappa(DATAFILE, "楊瓊瓔_教育文化_speech_constant"),
"楊瓊瓔_社會福利_face": compute_data_kappa(DATAFILE, "楊瓊瓔_社會福利_face"),
"楊瓊瓔_社會福利_gesture_function": compute_data_kappa(DATAFILE, "楊瓊瓔_社會福利_gesture_function"),
"楊瓊瓔_社會福利_hand": compute_data_kappa(DATAFILE, "楊瓊瓔_社會福利_hand"),
"楊瓊瓔_社會福利_head": compute_data_kappa(DATAFILE, "楊瓊瓔_社會福利_head"),
"楊瓊瓔_社會福利_speech_constant": compute_data_kappa(DATAFILE, "楊瓊瓔_社會福利_speech_constant"),
"費鴻泰_司法法制_face": compute_data_kappa(DATAFILE, "費鴻泰_司法法制_face"),
"費鴻泰_司法法制_gesture_function": compute_data_kappa(DATAFILE, "費鴻泰_司法法制_gesture_function"),
"費鴻泰_司法法制_hand": compute_data_kappa(DATAFILE, "費鴻泰_司法法制_hand"),
"費鴻泰_司法法制_head": compute_data_kappa(DATAFILE, "費鴻泰_司法法制_head"),
"費鴻泰_司法法制_speech_constant": compute_data_kappa(DATAFILE, "費鴻泰_司法法制_speech_constant"),
"費鴻泰_財政_face": compute_data_kappa(DATAFILE, "費鴻泰_財政_face"),
"費鴻泰_財政_gesture_function": compute_data_kappa(DATAFILE, "費鴻泰_財政_gesture_function"),
"費鴻泰_財政_hand": compute_data_kappa(DATAFILE, "費鴻泰_財政_hand"),
"費鴻泰_財政_head": compute_data_kappa(DATAFILE, "費鴻泰_財政_head"),
"費鴻泰_財政_speech_constant": compute_data_kappa(DATAFILE, "費鴻泰_財政_speech_constant"),
"高虹安_教育文化_face": compute_data_kappa(DATAFILE, "高虹安_教育文化_face"),
"高虹安_教育文化_gesture_function": compute_data_kappa(DATAFILE, "高虹安_教育文化_gesture_function"),
"高虹安_教育文化_hand": compute_data_kappa(DATAFILE, "高虹安_教育文化_hand"),
"高虹安_教育文化_head": compute_data_kappa(DATAFILE, "高虹安_教育文化_head"),
"高虹安_教育文化_speech_constant": compute_data_kappa(DATAFILE, "高虹安_教育文化_speech_constant"),
"高虹安_社會福利_face": compute_data_kappa(DATAFILE, "高虹安_社會福利_face"),
"高虹安_社會福利_gesture_function": compute_data_kappa(DATAFILE, "高虹安_社會福利_gesture_function"),
"高虹安_社會福利_hand": compute_data_kappa(DATAFILE, "高虹安_社會福利_hand"),
"高虹安_社會福利_head": compute_data_kappa(DATAFILE, "高虹安_社會福利_head"),
"高虹安_社會福利_speech_constant": compute_data_kappa(DATAFILE, "高虹安_社會福利_speech_constant"),
}
