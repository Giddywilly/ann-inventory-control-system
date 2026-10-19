"""ann_engine.py - loads the trained ANN and performs real-time reorder prediction."""
import os, pickle, math
import pandas as pd
from train_ann import FEATURES, CAT_COLS, engineer

BASE = os.path.dirname(os.path.abspath(__file__))
with open(f'{BASE}/model/ann_model.pkl', 'rb') as f: MODEL = pickle.load(f)
with open(f'{BASE}/model/scaler.pkl', 'rb') as f: SCALER = pickle.load(f)
with open(f'{BASE}/model/encoders.pkl', 'rb') as f: ENCODERS = pickle.load(f)

CATEGORIES = list(ENCODERS['category'].classes_)
DEMANDS = ['High', 'Medium', 'Low']
SEASON_AMP = {'Electronics': 0.15, 'Consumables': 0.30, 'Office Supplies': 0.20,
              'Raw Materials': 0.25, 'Maintenance Spares': 0.10}


def compute_rop(avg_daily_demand, lead_time, category, month):
    """Equation 4.1: ROP = d_bar x L x S_t, with S_t = 1 + A sin(2*pi*t/12)."""
    s_t = 1 + SEASON_AMP.get(category, 0.2) * math.sin(2 * math.pi * month / 12)
    return round(avg_daily_demand * lead_time * s_t, 2)


def _safe(col, values):
    """Map a value the model has not seen (e.g. a newly added store location)
    to the first known class so prediction still runs."""
    known = set(ENCODERS[col].classes_)
    return [v if v in known else ENCODERS[col].classes_[0] for v in values]


def predict_frame(df):
    df = df.copy()
    for c in CAT_COLS:
        df[c + '_enc'] = ENCODERS[c].transform(_safe(c, df[c]))
    df = engineer(df)
    X = SCALER.transform(df[FEATURES].values)
    prob = MODEL.predict_proba(X)[:, 1]
    return (prob >= 0.5).astype(int), prob


def predict(location_name, category, demand_rate, current_stock, reorder_level,
            quantity_issued, quantity_received, unit_cost, lead_time_days,
            avg_daily_demand, month):
    rop = compute_rop(avg_daily_demand, lead_time_days, category, month)
    row = pd.DataFrame([dict(location_name=location_name, category=category,
                             demand_rate=demand_rate, current_stock=current_stock,
                             reorder_level=reorder_level, quantity_issued=quantity_issued,
                             quantity_received=quantity_received, unit_cost=unit_cost,
                             lead_time_days=lead_time_days, avg_daily_demand=avg_daily_demand,
                             predicted_rop=rop, month=month)])
    pred, prob = predict_frame(row)
    return int(pred[0]), float(prob[0]), rop
