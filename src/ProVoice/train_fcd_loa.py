import csv, pathlib, joblib, re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, hamming_loss
from sklearn.multioutput import MultiOutputClassifier

DATA_CSV  = pathlib.Path('data/processed_data/fcd_out.csv')
MODEL_OUT = pathlib.Path('trained_models/fcd_levels.pkl')
RANDOM_SEED = 42
BY_GROUP = False 


LEVELS = [f'Level_{i}' for i in range(1,6)]
FEATS  = [f'Feature_{i}' for i in range(1,13)]


def sanitize_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_\-]+', '_', s)[:60] or 'group'


def load_csv(path: pathlib.Path):
    X, Y, G = [], [], []
    with path.open('r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                feats = [float(row[k]) for k in FEATS]
                y = [int(float(row[k])) for k in LEVELS]
            except Exception as e:
                print(f"[Error] Failed to process row: {e}")
                continue
            X.append(feats)
            Y.append(y)
            G.append(row.get('function_group',''))
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64), np.array(G)


def build_model():
    try:
        from xgboost import XGBClassifier
        from sklearn.multiclass import OneVsRestClassifier
        base = XGBClassifier(
            objective='binary:logistic',
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, n_jobs=4,
            reg_lambda=1.0
        )
        return OneVsRestClassifier(base)
    except Exception as e:
        print(f"[Error] Failed to build model: {e}")
        from sklearn.ensemble import RandomForestClassifier
        return MultiOutputClassifier(RandomForestClassifier(n_estimators=400, max_depth=12, class_weight='balanced', random_state=RANDOM_SEED))


def report(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    print(f"\n==== {title} ====")
    print('micro-F1 :', f1_score(y_true, y_pred, average='micro'))
    print('macro-F1 :', f1_score(y_true, y_pred, average='macro'))
    print('hamming  :', hamming_loss(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=[f'L{i}' for i in range(1,6)], digits=3))


def train_global(X: np.ndarray, Y: np.ndarray, out_path: pathlib.Path):
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_SEED)
    model = build_model()
    model.fit(Xtr, Ytr)
    Yhat = model.predict(Xte)
    report(Yte, Yhat, 'FCD→Levels (global)')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print('[OK] saved', out_path)


def train_by_group(X: np.ndarray, Y: np.ndarray, G: np.ndarray, out_path: pathlib.Path):
    uniq = sorted(set(G))
    for g in uniq:
        mask = (G == g)
        if mask.sum() < 40: 
            print('[Skip]', g, 'samples too few:', int(mask.sum()))
            continue
        model = build_model()
        Xg, Yg = X[mask], Y[mask]
        Xtr, Xte, Ytr, Yte = train_test_split(Xg, Yg, test_size=0.2, random_state=RANDOM_SEED)
        model.fit(Xtr, Ytr)
        Yhat = model.predict(Xte)
        report(Yte, Yhat, f'FCD→Levels (group={g})')
        out = out_path.with_name(f"fcd_levels_{sanitize_name(str(g))}.pkl")
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out)
        print('[OK] saved', out)

if __name__ == '__main__':
    X, Y, G = load_csv(DATA_CSV)
    if not BY_GROUP:
        train_global(X, Y, MODEL_OUT)
    else:
        train_by_group(X, Y, G, MODEL_OUT)


