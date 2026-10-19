"""
train_ann.py
Pre-processes the inventory dataset, trains the ANN (MLPClassifier) on a 70:30
stratified split, evaluates it, compares it with the existing rule-based
system and saves the model, scaler, encoders, metrics and figures.
"""
import os, json, pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)

BASE = os.path.dirname(os.path.abspath(__file__))

CAT_COLS = ['category', 'demand_rate', 'location_name']
NUM_COLS = ['current_stock', 'reorder_level', 'quantity_issued', 'quantity_received',
            'unit_cost', 'lead_time_days', 'avg_daily_demand', 'predicted_rop', 'month']
ENG_COLS = ['stock_vs_rop', 'stock_coverage', 'replenish_ratio', 'consumption_pressure']
FEATURES = [c + '_enc' for c in CAT_COLS] + NUM_COLS + ENG_COLS   # 16 input features


def engineer(df):
    df = df.copy()
    df['stock_vs_rop'] = df['current_stock'] - df['predicted_rop']
    df['stock_coverage'] = df['current_stock'] / (df['avg_daily_demand'] + 0.01)
    df['replenish_ratio'] = df['quantity_received'] / (df['quantity_issued'] + 1)
    df['consumption_pressure'] = df['quantity_issued'] / (df['current_stock'] + 1)
    return df


def main():
    os.makedirs(f'{BASE}/model', exist_ok=True)
    os.makedirs(f'{BASE}/figures', exist_ok=True)
    df = pd.read_csv(f'{BASE}/data/inventory_dataset.csv')

    encoders = {}
    for c in CAT_COLS:
        le = LabelEncoder().fit(df[c])
        df[c + '_enc'] = le.transform(df[c])
        encoders[c] = le
    df = engineer(df)

    X, y = df[FEATURES].values, df['reorder_label'].values
    idx = np.arange(len(df))
    X_tr, X_te, y_tr, y_te, i_tr, i_te = train_test_split(
        X, y, idx, test_size=0.30, stratify=y, random_state=42)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu',
                          solver='adam', learning_rate_init=0.001, max_iter=500,
                          early_stopping=True, validation_fraction=0.10,
                          n_iter_no_change=30, random_state=42)
    model.fit(X_tr_s, y_tr, sample_weight=compute_sample_weight('balanced', y_tr))

    prob = model.predict_proba(X_te_s)[:, 1]
    pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y_te, pred)
    m = dict(accuracy=accuracy_score(y_te, pred), precision=precision_score(y_te, pred),
             recall=recall_score(y_te, pred), f1=f1_score(y_te, pred),
             auc=roc_auc_score(y_te, prob), iterations=int(model.n_iter_),
             tn=int(cm[0, 0]), fp=int(cm[0, 1]), fn=int(cm[1, 0]), tp=int(cm[1, 1]))

    # existing system: static minimum warning quantity rule on the same test set
    te = df.iloc[i_te]
    rule = (te['current_stock'] <= te['reorder_level']).astype(int).values
    rcm = confusion_matrix(y_te, rule)
    rule_score = (te['reorder_level'] - te['current_stock']).values.astype(float)
    m['existing'] = dict(accuracy=accuracy_score(y_te, rule), precision=precision_score(y_te, rule),
                         recall=recall_score(y_te, rule), f1=f1_score(y_te, rule),
                         auc=roc_auc_score(y_te, rule_score),
                         tn=int(rcm[0, 0]), fp=int(rcm[0, 1]), fn=int(rcm[1, 0]), tp=int(rcm[1, 1]))
    m['split'] = dict(train=len(y_tr), test=len(y_te), train_pos=int(y_tr.sum()),
                      train_neg=int((y_tr == 0).sum()), test_pos=int(y_te.sum()),
                      test_neg=int((y_te == 0).sum()))
    m['features'] = FEATURES
    m['test_by_category'] = {}
    for cat in sorted(te['category'].unique()):
        mask = (te['category'] == cat).values
        m['test_by_category'][cat] = dict(
            records=int(mask.sum()), actual_reorders=int(y_te[mask].sum()),
            detected=int(((pred == 1) & (y_te == 1) & mask).sum()),
            accuracy=accuracy_score(y_te[mask], pred[mask]))

    with open(f'{BASE}/model/ann_model.pkl', 'wb') as f: pickle.dump(model, f)
    with open(f'{BASE}/model/scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
    with open(f'{BASE}/model/encoders.pkl', 'wb') as f: pickle.dump(encoders, f)
    with open(f'{BASE}/model/metrics.json', 'w') as f: json.dump(m, f, indent=2, default=float)

    # ---------- figures ----------
    plt.rcParams.update({'font.size': 11})
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax1.plot(model.loss_curve_, color='#123a63', lw=2, label='Training Loss')
    ax1.set_xlabel('Iterations (Epochs)'); ax1.set_ylabel('Training Loss', color='#123a63')
    ax2 = ax1.twinx()
    ax2.plot(model.validation_scores_, color='#c00000', ls='--', lw=2, label='Validation Accuracy')
    ax2.set_ylabel('Validation Accuracy', color='#c00000')
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='center right'); ax1.grid(alpha=.3)
    plt.title('ANN Model Training Loss and Validation Accuracy')
    plt.tight_layout(); plt.savefig(f'{BASE}/figures/training_curve.png', dpi=150); plt.close()

    plt.figure(figsize=(6.4, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Reorder (0)', 'Reorder (1)'],
                yticklabels=['No Reorder (0)', 'Reorder (1)'], annot_kws={'size': 14})
    plt.xlabel('Predicted Label'); plt.ylabel('Actual Label')
    plt.title('Confusion Matrix of the ANN Model (Test Set)')
    plt.tight_layout(); plt.savefig(f'{BASE}/figures/confusion_matrix.png', dpi=150); plt.close()

    fpr, tpr, _ = roc_curve(y_te, prob)
    rf, rt, _ = roc_curve(y_te, rule_score)
    plt.figure(figsize=(6.6, 5))
    plt.plot(fpr, tpr, color='#123a63', lw=2.2, label=f'Proposed ANN (AUC = {m["auc"]:.4f})')
    plt.plot(rf, rt, color='#e07b00', lw=1.8, label=f'Existing Rule-Based (AUC = {m["existing"]["auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Proposed ANN vs Existing System'); plt.legend(loc='lower right'); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f'{BASE}/figures/roc_curve.png', dpi=150); plt.close()

    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    ex = [m['existing'][k] * 100 for k in ['accuracy', 'precision', 'recall', 'f1']]
    pr = [m[k] * 100 for k in ['accuracy', 'precision', 'recall', 'f1']]
    x = np.arange(len(labels)); w = 0.36
    plt.figure(figsize=(8, 4.4))
    b1 = plt.bar(x - w / 2, ex, w, color='#e07b00', label='Existing System (Rule-Based)')
    b2 = plt.bar(x + w / 2, pr, w, color='#123a63', label='Proposed ANN System')
    for b in list(b1) + list(b2):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f'{b.get_height():.2f}', ha='center', fontsize=9)
    plt.xticks(x, labels); plt.ylabel('Score (%)'); plt.ylim(0, 110); plt.legend(loc='lower right')
    plt.title('Performance Comparison: Existing vs Proposed System'); plt.grid(axis='y', alpha=.3)
    plt.tight_layout(); plt.savefig(f'{BASE}/figures/comparison.png', dpi=150); plt.close()

    print(f"Training complete: accuracy {m['accuracy']*100:.2f}%, recall {m['recall']*100:.2f}%, "
          f"F1 {m['f1']*100:.2f}%, AUC {m['auc']:.4f}, stopped after {m['iterations']} iterations.")


if __name__ == '__main__':
    main()
