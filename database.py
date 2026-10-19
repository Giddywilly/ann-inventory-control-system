"""database.py - SQLite schema and seeding for the ANN Inventory Control System."""
import sqlite3, os
import pandas as pd
from werkzeug.security import generate_password_hash

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, 'inventory.db')

SCHEMA = '''
CREATE TABLE IF NOT EXISTS store_locations (
  location_id INTEGER PRIMARY KEY AUTOINCREMENT,
  location_name VARCHAR(100) NOT NULL,
  region VARCHAR(100));
CREATE TABLE IF NOT EXISTS suppliers (
  supplier_id VARCHAR(10) PRIMARY KEY,
  supplier_name VARCHAR(100) NOT NULL,
  city VARCHAR(60),
  lead_time_days INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS inventory_items (
  item_id VARCHAR(10) PRIMARY KEY,
  item_name VARCHAR(100) NOT NULL,
  category VARCHAR(40) NOT NULL,
  supplier_id VARCHAR(10) REFERENCES suppliers(supplier_id),
  unit_cost DECIMAL(12,2),
  reorder_level INTEGER);
CREATE TABLE IF NOT EXISTS transactions (
  transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
  location_id INTEGER REFERENCES store_locations(location_id),
  item_id VARCHAR(10) REFERENCES inventory_items(item_id),
  transaction_type VARCHAR(20),
  current_stock INTEGER,
  quantity_issued INTEGER,
  quantity_received INTEGER,
  avg_daily_demand REAL,
  predicted_rop REAL,
  demand_rate VARCHAR(10),
  month INTEGER,
  ann_decision INTEGER,
  ann_probability REAL,
  transaction_date DATETIME DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS users (
  user_id INTEGER PRIMARY KEY AUTOINCREMENT,
  username VARCHAR(50) UNIQUE NOT NULL,
  password VARCHAR(200) NOT NULL,
  full_name VARCHAR(100),
  role VARCHAR(20),
  location_id INTEGER REFERENCES store_locations(location_id));
'''

LOCATIONS = [('Main Warehouse', 'Central'), ('Branch Store North', 'North'), ('Branch Store South', 'South')]
SUPPLIERS = [('SUP001', 'Apex Electronics Ltd', 'Port Harcourt', 7),
             ('SUP002', 'Sunrise Consumables Ltd', 'Lagos', 5),
             ('SUP003', 'Metro Office Supplies', 'Port Harcourt', 4),
             ('SUP004', 'Delta Raw Materials Co.', 'Warri', 14),
             ('SUP005', 'Allied Industrial Spares', 'Aba', 10)]
USERS = [('admin', 'System Administrator', 'Administrator', 1),
         ('store_main', 'Grace Eze', 'Store Officer', 1),
         ('store_north', 'Amaka Nwosu', 'Store Officer', 2),
         ('store_south', 'Tunde Bello', 'Store Officer', 3)]


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


def init_db():
    fresh = not os.path.exists(DB_PATH)
    db = get_db()
    db.executescript(SCHEMA)
    if fresh:
        from ann_engine import predict_frame
        db.executemany('INSERT INTO store_locations(location_name,region) VALUES (?,?)', LOCATIONS)
        db.executemany('INSERT INTO suppliers VALUES (?,?,?,?)', SUPPLIERS)
        df = pd.read_csv(os.path.join(BASE, 'data', 'inventory_dataset.csv'))
        items = df.groupby('item_id').first().reset_index()
        rol = df.groupby('item_id').reorder_level.median().round().astype(int)
        for _, r in items.iterrows():
            db.execute('INSERT INTO inventory_items VALUES (?,?,?,?,?,?)',
                       (r.item_id, r.item_name, r.category, r.supplier_id, float(r.unit_cost), int(rol[r.item_id])))
        pred, prob = predict_frame(df)
        df['ann_decision'], df['ann_probability'] = pred, prob
        df['transaction_type'] = df.quantity_received.apply(lambda q: 'Receipt' if q > 0 else 'Issuance')
        rows = [(int(r.location_id), r.item_id, r.transaction_type, int(r.current_stock),
                 int(r.quantity_issued), int(r.quantity_received), float(r.avg_daily_demand),
                 float(r.predicted_rop), r.demand_rate, int(r.month), int(r.ann_decision),
                 float(r.ann_probability)) for r in df.itertuples()]
        db.executemany('''INSERT INTO transactions(location_id,item_id,transaction_type,current_stock,
            quantity_issued,quantity_received,avg_daily_demand,predicted_rop,demand_rate,month,
            ann_decision,ann_probability) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''', rows)
        for u, name, role, loc in USERS:
            db.execute('INSERT INTO users(username,password,full_name,role,location_id) VALUES (?,?,?,?,?)',
                       (u, generate_password_hash('password123'), name, role, loc))
        db.commit()
    db.close()
