import numpy as np, pandas as pd, math, os
RNG = np.random.default_rng(42)
BASE = os.path.dirname(os.path.abspath(__file__))
LOCATIONS = [(1,'Main Warehouse','Central',1.00),(2,'Branch Store North','North',0.80),(3,'Branch Store South','South',0.65)]
SUPPLIERS = [('SUP001','Apex Electronics Ltd','Port Harcourt',7),('SUP002','Sunrise Consumables Ltd','Lagos',5),
 ('SUP003','Metro Office Supplies','Port Harcourt',4),('SUP004','Delta Raw Materials Co.','Warri',14),('SUP005','Allied Industrial Spares','Aba',10)]
ITEMS = [
('ITM001','Laptop Battery','Electronics','SUP001',45000.00,0.9),
('ITM002','Computer Mouse','Electronics','SUP001',6500.00,2.2),
('ITM003','LED Monitor 24-inch','Electronics','SUP001',98000.00,0.5),
('ITM004','Printer Cartridge','Consumables','SUP002',18500.00,2.8),
('ITM005','Hand Sanitizer 500ml','Consumables','SUP002',2500.00,5.5),
('ITM006','Liquid Detergent 5L','Consumables','SUP002',7800.00,3.1),
('ITM007','A4 Paper (Ream)','Office Supplies','SUP003',6200.00,7.5),
('ITM008','Ballpoint Pen (Box)','Office Supplies','SUP003',3000.00,3.6),
('ITM009','Box File','Office Supplies','SUP003',1500.00,4.2),
('ITM010','Packaging Carton','Raw Materials','SUP004',900.00,14.0),
('ITM011','Plastic Granules (25kg)','Raw Materials','SUP004',32000.00,1.6),
('ITM012','Steel Rod 12mm','Raw Materials','SUP004',9500.00,4.8),
('ITM013','Ball Bearing 6204','Maintenance Spares','SUP005',4200.00,1.3),
('ITM014','V-Belt A-42','Maintenance Spares','SUP005',5600.00,0.8),
('ITM015','Hydraulic Oil Filter','Maintenance Spares','SUP005',12500.00,0.6)]
SEASON_AMP = {'Electronics':0.15,'Consumables':0.30,'Office Supplies':0.20,'Raw Materials':0.25,'Maintenance Spares':0.10}
PERIODS = 80
DAYS = 7
def seasonal(t, A): return 1 + A*math.sin(2*math.pi*t/12)
def simulate():
    sup_lead = {s[0]: s[3] for s in SUPPLIERS}
    rows = []
    for loc_id, loc_name, _, loc_factor in LOCATIONS:
        for item_id, name, cat, sup, cost, d0 in ITEMS:
            L = sup_lead[sup]; A = SEASON_AMP[cat]
            drift = 1 + 0.40*np.sin(np.linspace(0, 2.2*np.pi, PERIODS) + RNG.uniform(0, 6))
            issued = []
            for p in range(PERIODS):
                month = (p*DAYS//30) % 12 + 1
                lam = d0*loc_factor*drift[p]*seasonal(month, A)*DAYS
                issued.append(int(RNG.poisson(max(lam, 0.2))))
            static_rol = max(1, int(round(d0*loc_factor*DAYS*RNG.uniform(0.5, 0.9))))
            mgr_trigger = d0*loc_factor*L*RNG.uniform(1.0, 1.5)
            order_qty = max(5, int(round(d0*loc_factor*35)))
            stock = int(round(static_rol*RNG.uniform(1.8, 3.0)))
            pipeline = []; hist = []
            for p in range(PERIODS):
                month = (p*DAYS//30) % 12 + 1
                received = sum(q for (a, q) in pipeline if a == p)
                pipeline = [(a, q) for (a, q) in pipeline if a != p]
                stock += received
                iss = min(issued[p], stock)
                stock -= iss
                hist.append(issued[p]/DAYS)
                dbar = float(np.mean(hist[-4:]))
                st = seasonal(month, A)
                rop = round(dbar*L*st, 2)
                if stock <= mgr_trigger and not pipeline:
                    arrive = p + max(1, math.ceil(L/DAYS))
                    pipeline.append((arrive, order_qty))
                k = max(1, math.ceil(L/DAYS))
                fut = issued[p+1:p+1+k]
                if len(fut) < k:
                    fut = fut + [int(round(np.mean(issued[-4:])))]*(k-len(fut))
                lt_demand = sum(fut)*(L/(DAYS*k))
                label = int(stock < lt_demand)
                ratio = issued[p]/max(d0*loc_factor*DAYS, 1e-6)
                demand_rate = 'High' if ratio > 1.2 else ('Low' if ratio < 0.8 else 'Medium')
                rows.append(dict(location_id=loc_id, location_name=loc_name, item_id=item_id, item_name=name, category=cat,
                    supplier_id=sup, unit_cost=cost, period=p+1, month=month, current_stock=int(stock), reorder_level=static_rol,
                    quantity_issued=int(issued[p]), quantity_received=int(received), lead_time_days=L,
                    avg_daily_demand=round(dbar, 3), predicted_rop=rop, demand_rate=demand_rate, reorder_label=label))
    df = pd.DataFrame(rows); df.insert(0, 'transaction_id', range(1, len(df)+1)); return df
if __name__ == '__main__':
    df = simulate(); os.makedirs(f'{BASE}/data', exist_ok=True)
    df.to_csv(f'{BASE}/data/inventory_dataset.csv', index=False)
    print(df.shape, df.reorder_label.value_counts().to_dict())
