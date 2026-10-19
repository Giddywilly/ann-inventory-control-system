"""app.py - Flask web application for the ANN Inventory Control System."""
import os, json, webbrowser, threading
from functools import wraps

BASE = os.path.dirname(os.path.abspath(__file__))


def prepare_model():
    """First launch only: build the dataset and train the ANN if the model files are missing."""
    if os.path.exists(f'{BASE}/model/ann_model.pkl') and os.path.exists(f'{BASE}/model/metrics.json'):
        return
    print('First launch: preparing the dataset and training the ANN model (about one minute)...')
    if not os.path.exists(f'{BASE}/data/inventory_dataset.csv'):
        import generate_dataset
        os.makedirs(f'{BASE}/data', exist_ok=True)
        generate_dataset.simulate().to_csv(f'{BASE}/data/inventory_dataset.csv', index=False)
    import train_ann
    train_ann.main()


prepare_model()

from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
from werkzeug.security import check_password_hash, generate_password_hash
from database import get_db, init_db
from ann_engine import predict as ann_predict, CATEGORIES, DEMANDS

app = Flask(__name__)
app.secret_key = 'ANN_INV_SYS_SECURE_KEY'

with open(f'{BASE}/model/metrics.json') as f:
    METRICS = json.load(f)


@app.context_processor
def inject_globals():
    return dict(accuracy=METRICS['accuracy'] * 100)


def login_required(f):
    @wraps(f)
    def wrapper(*a, **k):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*a, **k)
    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*a, **k):
        if session.get('role') != 'Administrator':
            flash('Administrator access required.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*a, **k)
    return wrapper


# ── Authentication ───────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username=?', (request.form['username'].strip(),)).fetchone()
        db.close()
        if user and check_password_hash(user['password'], request.form['password']):
            session.update(user_id=user['user_id'], username=user['username'],
                           full_name=user['full_name'], role=user['role'],
                           location_id=user['location_id'])
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'danger')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ── Dashboard ────────────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    db = get_db()
    stats = dict(
        items=db.execute('SELECT COUNT(*) FROM inventory_items').fetchone()[0],
        suppliers=db.execute('SELECT COUNT(*) FROM suppliers').fetchone()[0],
        locations=db.execute('SELECT COUNT(*) FROM store_locations').fetchone()[0],
        transactions=db.execute('SELECT COUNT(*) FROM transactions').fetchone()[0],
        alerts=db.execute('SELECT COUNT(*) FROM transactions WHERE ann_decision=1').fetchone()[0])
    recent = db.execute('''SELECT t.transaction_id, l.location_name, i.item_name, i.category,
        t.current_stock, t.predicted_rop, t.demand_rate, t.ann_decision
        FROM transactions t JOIN store_locations l ON t.location_id=l.location_id
        JOIN inventory_items i ON t.item_id=i.item_id
        ORDER BY t.transaction_id DESC LIMIT 10''').fetchall()
    db.close()
    return render_template('dashboard.html', stats=stats, recent=recent)


# ── Inventory Items ──────────────────────────────────────────────
@app.route('/items')
@login_required
def items():
    db = get_db()
    rows = db.execute('''SELECT i.*, s.supplier_name FROM inventory_items i
        LEFT JOIN suppliers s ON i.supplier_id=s.supplier_id ORDER BY i.item_id''').fetchall()
    db.close()
    return render_template('items.html', items=rows)


@app.route('/items/add', methods=['GET', 'POST'])
@app.route('/items/edit/<item_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def item_form(item_id=None):
    db = get_db()
    if request.method == 'POST':
        f = request.form
        vals = (f['item_name'], f['category'], f['supplier_id'], float(f['unit_cost']), int(f['reorder_level']))
        if item_id:
            db.execute('''UPDATE inventory_items SET item_name=?, category=?, supplier_id=?,
                unit_cost=?, reorder_level=? WHERE item_id=?''', vals + (item_id,))
            flash('Inventory item updated successfully.', 'success')
        else:
            if db.execute('SELECT 1 FROM inventory_items WHERE item_id=?', (f['item_id'],)).fetchone():
                db.close()
                flash(f"Item ID {f['item_id']} already exists. Use a different ID.", 'danger')
                return redirect(url_for('item_form'))
            db.execute('INSERT INTO inventory_items VALUES (?,?,?,?,?,?)', (f['item_id'],) + vals)
            flash('Inventory item registered successfully.', 'success')
        db.commit(); db.close()
        return redirect(url_for('items'))
    item = db.execute('SELECT * FROM inventory_items WHERE item_id=?', (item_id,)).fetchone() if item_id else None
    sups = db.execute('SELECT * FROM suppliers').fetchall()
    db.close()
    return render_template('item_form.html', item=item, suppliers=sups, categories=CATEGORIES)


@app.route('/items/delete/<item_id>', methods=['POST'])
@login_required
@admin_required
def delete_item(item_id):
    db = get_db()
    db.execute('DELETE FROM transactions WHERE item_id=?', (item_id,))
    db.execute('DELETE FROM inventory_items WHERE item_id=?', (item_id,))
    db.commit(); db.close()
    flash('Inventory item removed.', 'info')
    return redirect(url_for('items'))


# ── Suppliers ────────────────────────────────────────────────────
@app.route('/suppliers', methods=['GET', 'POST'])
@login_required
def suppliers():
    db = get_db()
    if request.method == 'POST' and session.get('role') == 'Administrator':
        f = request.form
        if db.execute('SELECT 1 FROM suppliers WHERE supplier_id=?', (f['supplier_id'],)).fetchone():
            flash(f"Supplier ID {f['supplier_id']} already exists.", 'danger')
        else:
            db.execute('INSERT INTO suppliers VALUES (?,?,?,?)',
                       (f['supplier_id'], f['supplier_name'], f['city'], int(f['lead_time_days'])))
            db.commit(); flash('Supplier added successfully.', 'success')
    rows = db.execute('''SELECT s.*, COUNT(DISTINCT i.item_id) AS items,
        COALESCE(SUM(t.quantity_received),0) AS supplied, COALESCE(SUM(t.ann_decision),0) AS alerts
        FROM suppliers s LEFT JOIN inventory_items i ON s.supplier_id=i.supplier_id
        LEFT JOIN transactions t ON t.item_id=i.item_id
        GROUP BY s.supplier_id ORDER BY s.supplier_id''').fetchall()
    db.close()
    return render_template('suppliers.html', suppliers=rows)


# ── Stock Summary ────────────────────────────────────────────────
@app.route('/stock')
@login_required
def stock():
    db = get_db()
    rows = db.execute('''SELECT i.item_id, i.item_name, i.category,
        ROUND(AVG(t.current_stock),1) AS avg_stock, SUM(t.quantity_issued) AS issued,
        SUM(t.quantity_received) AS received, ROUND(AVG(t.predicted_rop),2) AS avg_rop,
        SUM(t.ann_decision) AS alerts
        FROM transactions t JOIN inventory_items i ON t.item_id=i.item_id
        GROUP BY i.item_id ORDER BY alerts DESC''').fetchall()
    db.close()
    return render_template('stock.html', rows=rows)


# ── Store Locations ──────────────────────────────────────────────
@app.route('/locations', methods=['GET', 'POST'])
@login_required
def locations():
    db = get_db()
    if request.method == 'POST' and session.get('role') == 'Administrator':
        db.execute('INSERT INTO store_locations(location_name,region) VALUES (?,?)',
                   (request.form['location_name'], request.form['region']))
        db.commit(); flash('Store location added successfully.', 'success')
    rows = db.execute('''SELECT l.location_id, l.location_name, l.region,
        COUNT(DISTINCT t.item_id) AS items, COUNT(t.transaction_id) AS tx,
        COALESCE(SUM(t.quantity_issued),0) AS issued, COALESCE(SUM(t.ann_decision),0) AS alerts
        FROM store_locations l LEFT JOIN transactions t ON l.location_id=t.location_id
        GROUP BY l.location_id ORDER BY l.location_id''').fetchall()
    db.close()
    return render_template('locations.html', rows=rows)


# ── Transactions ─────────────────────────────────────────────────
def _form_inputs(db, f):
    item = db.execute('''SELECT i.*, s.lead_time_days, s.supplier_name FROM inventory_items i
        JOIN suppliers s ON i.supplier_id=s.supplier_id WHERE item_id=?''', (f['item_id'],)).fetchone()
    loc = db.execute('SELECT * FROM store_locations WHERE location_id=?', (int(f['location_id']),)).fetchone()
    inputs = dict(location_name=loc['location_name'], category=item['category'],
                  demand_rate=f['demand_rate'], current_stock=int(f['current_stock']),
                  reorder_level=item['reorder_level'], quantity_issued=int(f['quantity_issued']),
                  quantity_received=int(f.get('quantity_received') or 0), unit_cost=item['unit_cost'],
                  lead_time_days=item['lead_time_days'], avg_daily_demand=float(f['avg_daily_demand']),
                  month=max(1, min(12, int(f['month']))))
    return item, loc, inputs


@app.route('/transactions')
@login_required
def transactions():
    db = get_db()
    rows = db.execute('''SELECT t.*, l.location_name, i.item_name FROM transactions t
        JOIN store_locations l ON t.location_id=l.location_id
        JOIN inventory_items i ON t.item_id=i.item_id
        ORDER BY t.transaction_id DESC LIMIT 50''').fetchall()
    db.close()
    return render_template('transactions.html', rows=rows)


@app.route('/transactions/add', methods=['GET', 'POST'])
@login_required
def add_transaction():
    db = get_db()
    if request.method == 'POST':
        f = request.form
        item, loc, x = _form_inputs(db, f)
        pred, prob, rop = ann_predict(**x)
        db.execute('''INSERT INTO transactions(location_id,item_id,transaction_type,current_stock,
            quantity_issued,quantity_received,avg_daily_demand,predicted_rop,demand_rate,month,
            ann_decision,ann_probability) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
                   (loc['location_id'], item['item_id'], f['transaction_type'], x['current_stock'],
                    x['quantity_issued'], x['quantity_received'], x['avg_daily_demand'], rop,
                    x['demand_rate'], x['month'], pred, prob))
        db.commit(); db.close()
        flash(('REORDER REQUIRED' if pred else 'STOCK SUFFICIENT') +
              f" for {item['item_name']} at {loc['location_name']} ({prob*100:.1f}% reorder probability)",
              'danger' if pred else 'success')
        return redirect(url_for('transactions'))
    locs = db.execute('SELECT * FROM store_locations').fetchall()
    its = db.execute('SELECT * FROM inventory_items ORDER BY item_id').fetchall()
    db.close()
    return render_template('transaction_form.html', locations=locs, items=its, demands=DEMANDS)


# ── ANN Prediction ───────────────────────────────────────────────
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    db = get_db()
    result = None
    form = {'location_id': '1', 'item_id': 'ITM010', 'current_stock': 95, 'quantity_issued': 113,
            'quantity_received': 0, 'avg_daily_demand': 16.2, 'demand_rate': 'High', 'month': 2}
    if request.method == 'POST':
        form = request.form
        item, loc, x = _form_inputs(db, form)
        pred, prob, rop = ann_predict(**x)
        result = dict(prediction=pred, probability=prob, rop=rop, item=dict(item),
                      location=loc['location_name'], stock=x['current_stock'])
    locs = db.execute('SELECT * FROM store_locations').fetchall()
    its = db.execute('SELECT * FROM inventory_items ORDER BY item_id').fetchall()
    db.close()
    return render_template('predict.html', result=result, form=form, locations=locs,
                           items=its, demands=DEMANDS)


# ── Model Performance and Reports ────────────────────────────────
@app.route('/performance')
@login_required
def performance():
    return render_template('performance.html', m=METRICS)


@app.route('/figures/<path:name>')
@login_required
def figures(name):
    return send_from_directory(os.path.join(BASE, 'figures'), name)


@app.route('/reports')
@login_required
def reports():
    db = get_db()
    by_cat = db.execute('''SELECT i.category, COUNT(*) AS tx, SUM(t.ann_decision) AS alerts,
        SUM(t.quantity_issued) AS issued FROM transactions t
        JOIN inventory_items i ON t.item_id=i.item_id GROUP BY i.category ORDER BY alerts DESC''').fetchall()
    by_loc = db.execute('''SELECT l.location_name, COUNT(*) AS tx, SUM(t.ann_decision) AS alerts
        FROM transactions t JOIN store_locations l ON t.location_id=l.location_id
        GROUP BY l.location_id''').fetchall()
    db.close()
    return render_template('reports.html', by_cat=by_cat, by_loc=by_loc)


# ── User Accounts ────────────────────────────────────────────────
@app.route('/users', methods=['GET', 'POST'])
@login_required
@admin_required
def users():
    db = get_db()
    if request.method == 'POST':
        f = request.form
        if db.execute('SELECT 1 FROM users WHERE username=?', (f['username'],)).fetchone():
            flash(f"Username {f['username']} is already taken.", 'danger')
        else:
            db.execute('INSERT INTO users(username,password,full_name,role,location_id) VALUES (?,?,?,?,?)',
                       (f['username'], generate_password_hash(f['password']), f['full_name'],
                        f['role'], int(f['location_id'])))
            db.commit(); flash('User account created successfully.', 'success')
    rows = db.execute('''SELECT u.*, l.location_name FROM users u
        LEFT JOIN store_locations l ON u.location_id=l.location_id''').fetchall()
    locs = db.execute('SELECT * FROM store_locations').fetchall()
    db.close()
    return render_template('users.html', users=rows, locations=locs)


@app.route('/users/delete/<int:uid>', methods=['POST'])
@login_required
@admin_required
def delete_user(uid):
    if uid == session['user_id']:
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('users'))
    db = get_db()
    db.execute('DELETE FROM users WHERE user_id=?', (uid,))
    db.commit(); db.close()
    flash('User account removed.', 'info')
    return redirect(url_for('users'))


if __name__ == '__main__':
    print('=' * 50)
    print(' ANN INVENTORY CONTROL SYSTEM')
    print(' General-Purpose Inventory Management')
    print('=' * 50)
    print('Initialising database...')
    init_db()
    print('Database initialised successfully.')
    print('Starting web server...')
    print('Open your browser at: http://127.0.0.1:5000')
    print('Keep this window open while using the system. Press CTRL+C to stop.')
    if not os.environ.get('NO_BROWSER'):
        threading.Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
    app.run(debug=False, port=5000)
