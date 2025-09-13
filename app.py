"""
app.py - Pocket Predictive Pilot (simple demo backend)

Requirements:
  pip install flask flask-cors

Run:
  python app.py
  (server listens on http://127.0.0.1:5000)

This demo uses SQLite for persistence (file: ppp.db).
It exposes endpoints for:
 - POST /expenses         (add one expense)
 - GET  /expenses         (list, optional ?category=&limit=)
 - GET  /expenses/summary (totals per category, months series, keywords, budgets, alerts)
 - POST /import           (multipart file upload: CSV)
 - POST /budgets          (set budget for a category)
 - GET  /predict          (returns per-category prediction)
 - GET  /keywords         (expose current keyword map)
 - POST /keywords        (update keywords for a category)
"""

from flask_c import Flask, request, jsonify, g
from flask_cors import CORS
import sqlite3
import os
import csv
from datetime import datetime
from math import isfinite

DB_PATH = "ppp.db"

app = Flask(__name__)
CORS(app)

def get_db():
    db = getattr(g, "_db", None)
    if db is None:
        need_init = not os.path.exists(DB_PATH)
        db = g._db = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
        if need_init:
            init_db(db)
    return db

def init_db(db):
    cur = db.cursor()
    cur.execute("""
    CREATE TABLE expenses (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      date TEXT NOT NULL,
      amount REAL NOT NULL,
      merchant TEXT,
      note TEXT,
      category TEXT
    )""")
    cur.execute("""
    CREATE TABLE keywords (
      category TEXT PRIMARY KEY,
      keywords TEXT
    )""")
    cur.execute("""
    CREATE TABLE budgets (
      category TEXT PRIMARY KEY,
      amount REAL
    )""")
    # seed default keywords
    defaults = {
      "Food": "starbucks,cafe,restaurant,ubereats,zomato,dominos,mcdonald",
      "Transport": "uber,ola,metro,bus,taxi,fuel,petrol",
      "Groceries": "bigbasket,grocery,supermarket,reliance,dmart",
      "Bills": "electricity,water,bill,netflix,spotify,subscription",
      "Rent": "rent",
      "Salary": "salary,payroll,direct deposit",
      "Entertainment": "movie,concert,theatre",
      "Others": ""
    }
    for cat, kws in defaults.items():
        cur.execute("INSERT INTO keywords(category, keywords) VALUES (?, ?)", (cat, kws))
    db.commit()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_db", None)
    if db is not None:
        db.close()

# --------------------------
# Utilities: categorization & prediction
# --------------------------
def normalize_text(s):
    return (s or "").lower()

def load_keywords():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT category, keywords FROM keywords")
    out = {}
    for r in cur.fetchall():
        out[r["category"]] = [k.strip() for k in (r["keywords"] or "").split(",") if k.strip()]
    return out

def categorize(expense_row):
    text = " ".join(filter(None, [expense_row.get("merchant",""), expense_row.get("note","")])).lower()
    kws = load_keywords()
    for cat, keys in kws.items():
        for kw in keys:
            if kw and kw in text:
                return cat
    return "Others"

def month_key(date_str):
    try:
        dt = datetime.fromisoformat(date_str)
    except Exception:
        dt = datetime.utcnow()
    return f"{dt.year}-{dt.month:02d}"

def linear_regression_predict(monthly_series, months_to_predict=3):
    # monthly_series: list of (month_key, value) sorted by month
    if not monthly_series:
        return [0.0]*months_to_predict
    # use index as x
    ys = [v for (_, v) in monthly_series]
    n = len(ys)
    xs = list(range(1, n+1))
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x*x for x in xs)
    sum_xy = sum(x*y for x,y in zip(xs, ys))
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        mean = sum_y / n
        return [round(mean,2)]*months_to_predict
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    preds = []
    for k in range(1, months_to_predict+1):
        x = n + k
        y = intercept + slope * x
        if not isfinite(y) or y < 0:
            y = 0.0
        preds.append(round(y, 2))
    return preds

# --------------------------
# Endpoints
# --------------------------
@app.route('/expenses', methods=['POST'])
def add_expense():
    payload = request.get_json()
    date = payload.get('date') or datetime.utcnow().isoformat()
    amount = float(payload.get('amount') or 0.0)
    merchant = payload.get('merchant') or ''
    note = payload.get('note') or ''
    row = {"merchant": merchant, "note": note}
    cat = categorize(row)
    db = get_db()
    cur = db.cursor()
    cur.execute("INSERT INTO expenses(date, amount, merchant, note, category) VALUES (?, ?, ?, ?, ?)",
                (date, amount, merchant, note, cat))
    db.commit()
    return jsonify({"ok": True, "category": cat}), 201

@app.route('/expenses', methods=['GET'])
def list_expenses():
    category = request.args.get('category')
    limit = int(request.args.get('limit') or 200)
    db = get_db()
    cur = db.cursor()
    if category:
        cur.execute("SELECT * FROM expenses WHERE category = ? ORDER BY date DESC LIMIT ?", (category, limit))
    else:
        cur.execute("SELECT * FROM expenses ORDER BY date DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    return jsonify(rows)

@app.route('/expenses/summary', methods=['GET'])
def expenses_summary():
    db = get_db()
    cur = db.cursor()
    # totals by category
    cur.execute("SELECT category, SUM(amount) as total FROM expenses GROUP BY category ORDER BY total DESC")
    totals = [{"category": r["category"], "value": round(r["total"] or 0,2)} for r in cur.fetchall()]
    # recent transactions
    cur.execute("SELECT date, amount, merchant, note, category FROM expenses ORDER BY date DESC LIMIT 50")
    recent = [dict(r) for r in cur.fetchall()]
    # month series for trend (last 12 months)
    cur.execute("SELECT date, amount, category FROM expenses ORDER BY date ASC")
    rows = [dict(r) for r in cur.fetchall()]
    monthly = {}
    for r in rows:
        m = month_key(r["date"])
        monthly.setdefault(m, 0.0)
        monthly[m] += float(r["amount"] or 0.0)
    month_series = [{"month": k, "total": round(v,2)} for k,v in sorted(monthly.items())]
    # budgets
    cur.execute("SELECT category, amount FROM budgets")
    budgets = {r["category"]: round(r["amount"],2) for r in cur.fetchall()}
    # keywords
    keywords = load_keywords()
    # compute alerts (predict next month per-category and compare to budgets)
    alerts = []
    # build monthly per-category series
    cur.execute("SELECT date, amount, category FROM expenses ORDER BY date ASC")
    rows = [dict(r) for r in cur.fetchall()]
    per_cat_month = {}
    for r in rows:
        m = month_key(r["date"])
        per_cat_month.setdefault(r["category"], {})
        per_cat_month[r["category"]].setdefault(m, 0.0)
        per_cat_month[r["category"]][m] += float(r["amount"] or 0.0)
    for cat, months in per_cat_month.items():
        items = sorted(months.items())
        preds = linear_regression_predict(items, months_to_predict=1)
        next_pred = preds[0] if preds else 0.0
        if cat in budgets and next_pred > budgets[cat]:
            alerts.append({"category": cat, "predicted": next_pred, "budget": budgets[cat]})
    return jsonify({
        "totalsByCategory": totals,
        "recent": recent,
        "monthSeries": month_series,
        "budgets": budgets,
        "keywords": keywords,
        "alerts": alerts
    })

@app.route('/import', methods=['POST'])
def import_csv():
    if 'file' not in request.files:
        return "missing file", 400
    f = request.files['file']
    text = f.read().decode('utf-8', errors='ignore')
    reader = csv.reader(text.splitlines())
    rows = list(reader)
    if not rows:
        return "empty", 400
    header = [h.strip().lower() for h in rows[0]]
    idx_map = {name: i for i, name in enumerate(header)}
    db = get_db()
    cur = db.cursor()
    count = 0
    for r in rows[1:]:
        def cell(n): 
            i = idx_map.get(n)
            return (r[i].strip() if i is not None and i < len(r) else '')
        date = cell('date') or cell('transaction_date') or datetime.utcnow().isoformat()
        amount = float(cell('amount') or cell('value') or 0)
        merchant = cell('merchant') or cell('description')
        note = ''
        cat = categorize({"merchant": merchant, "note": note})
        cur.execute("INSERT INTO expenses(date, amount, merchant, note, category) VALUES (?, ?, ?, ?, ?)",
                    (date, amount, merchant, note, cat))
        count += 1
    db.commit()
    return jsonify({"imported": count})

@app.route('/budgets', methods=['POST'])
def set_budget():
    payload = request.get_json()
    cat = payload.get('category')
    amount = float(payload.get('amount') or 0.0)
    db = get_db()
    cur = db.cursor()
    cur.execute("INSERT INTO budgets(category, amount) VALUES (?, ?) ON CONFLICT(category) DO UPDATE SET amount=excluded.amount", (cat, amount))
    db.commit()
    return jsonify({"ok": True, "category": cat, "amount": amount})

@app.route('/predict', methods=['GET'])
def predict_all():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT date, amount, category FROM expenses ORDER BY date ASC")
    rows = [dict(r) for r in cur.fetchall()]
    per_cat = {}
    for r in rows:
        m = month_key(r["date"])
        per_cat.setdefault(r["category"], {})
        per_cat[r["category"]].setdefault(m, 0.0)
        per_cat[r["category"]][m] += float(r["amount"] or 0.0)
    out = {}
    for cat, months in per_cat.items():
        items = sorted(months.items())
        out[cat] = linear_regression_predict(items, months_to_predict=3)
    return jsonify(out)

@app.route('/keywords', methods=['GET'])
def get_keywords():
    return jsonify(load_keywords())

@app.route('/keywords', methods=['POST'])
def update_keywords():
    payload = request.get_json()
    cat = payload.get('category')
    kws = payload.get('keywords','')
    db = get_db()
    cur = db.cursor()
    cur.execute("INSERT INTO keywords(category, keywords) VALUES (?, ?) ON CONFLICT(category) DO UPDATE SET keywords=excluded.keywords", (cat, kws))
    db.commit()
    return jsonify({"ok": True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
