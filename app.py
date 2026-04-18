import streamlit as st
import pandas as pd
import numpy as np
import warnings
import io
warnings.filterwarnings('ignore')

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import xgboost as xgb
import calendar

try:
    from pmdarima import auto_arima
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinAssist · Personal Finance",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    'income':   '#00C49A',
    'expense':  '#FF6B6B',
    'savings':  '#4ECDC4',
    'anomaly':  '#FFD93D',
    'forecast': '#A78BFA',
    'neutral':  '#94A3B8',
}

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #080C14; }
.block-container { padding: 1.5rem 2rem 3rem 2rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #111827 100%);
    border-right: 1px solid #1E293B;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #111827 0%, #1E293B 100%);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    border: 1px solid #1E293B;
    position: relative;
    overflow: hidden;
    margin-bottom: 0.5rem;
    transition: transform 0.2s;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.kpi-card.green::before  { background: linear-gradient(90deg, #00C49A, #4ECDC4); }
.kpi-card.red::before    { background: linear-gradient(90deg, #FF6B6B, #FF8E53); }
.kpi-card.blue::before   { background: linear-gradient(90deg, #4ECDC4, #556EE6); }
.kpi-card.purple::before { background: linear-gradient(90deg, #A78BFA, #7C3AED); }
.kpi-card.yellow::before { background: linear-gradient(90deg, #FFD93D, #FF9F43); }

.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748B;
    margin: 0 0 0.4rem 0;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #F1F5F9;
    margin: 0;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.75rem;
    color: #64748B;
    margin: 0.3rem 0 0 0;
}
.kpi-badge-good  { color: #00C49A; font-weight: 600; }
.kpi-badge-warn  { color: #FFD93D; font-weight: 600; }
.kpi-badge-bad   { color: #FF6B6B; font-weight: 600; }

/* ── Section Headers ── */
.sec-header {
    font-size: 1rem;
    font-weight: 700;
    color: #CBD5E1;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1E293B;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Insight Cards ── */
.insight-card {
    background: #111827;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    border-left: 4px solid;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
    color: #CBD5E1;
    line-height: 1.5;
}
.insight-card.good   { border-color: #00C49A; }
.insight-card.warn   { border-color: #FFD93D; }
.insight-card.danger { border-color: #FF6B6B; }
.insight-card .tag {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.insight-card.good   .tag { color: #00C49A; }
.insight-card.warn   .tag { color: #FFD93D; }
.insight-card.danger .tag { color: #FF6B6B; }

/* ── Page title ── */
.page-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #F1F5F9;
    margin: 0 0 0.25rem 0;
}
.page-sub {
    font-size: 0.85rem;
    color: #64748B;
    margin: 0 0 1.5rem 0;
}

/* ── Welcome screen ── */
.welcome-hero {
    background: linear-gradient(135deg, #0D1117 0%, #1a1f35 50%, #0D1117 100%);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    border: 1px solid #1E293B;
    text-align: center;
    margin: 2rem 0;
}
.welcome-hero h1 { font-size: 2.5rem; font-weight: 700; color: #F1F5F9; margin: 0 0 0.75rem 0; }
.welcome-hero p  { font-size: 1rem; color: #64748B; max-width: 550px; margin: 0 auto 2rem auto; line-height: 1.7; }

/* ── Risk badge ── */
.risk-pill {
    display: inline-block;
    padding: 0.2rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.risk-stable   { background: rgba(0,196,154,0.15); color: #00C49A; }
.risk-moderate { background: rgba(255,217,61,0.15); color: #FFD93D; }
.risk-high     { background: rgba(255,107,107,0.15); color: #FF6B6B; }

/* ── Nav radio styling ── */
div[data-testid="stRadio"] label {
    font-size: 0.88rem !important;
    padding: 0.4rem 0.75rem !important;
    border-radius: 8px !important;
    display: block !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
}
div[data-testid="stRadio"] label:hover { background: #1E293B !important; }

.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def cap_outliers_mad(series, window=12, k=2.5):
    rolling_med = series.rolling(window, min_periods=3, center=True).median()
    rolling_mad = (series - rolling_med).abs().rolling(window, min_periods=3, center=True).median()
    return series.clip(lower=rolling_med - k * rolling_mad, upper=rolling_med + k * rolling_mad)

def evaluate(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE (%)': round(mape, 2)}

def safe_mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def plotly_cfg():
    return dict(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#94A3B8', size=12),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#1E293B', bordercolor='#334155', font_size=12),
        margin=dict(l=10, r=10, t=40, b=10),
    )

def kpi(label, value, sub='', color='blue'):
    return f"""
<div class="kpi-card {color}">
  <p class="kpi-label">{label}</p>
  <p class="kpi-value">{value}</p>
  {"<p class='kpi-sub'>" + sub + "</p>" if sub else ""}
</div>"""

def insight(text, kind='good', tag=''):
    return f"""
<div class="insight-card {kind}">
  {"<div class='tag'>" + tag + "</div>" if tag else ""}
  {text}
</div>"""

def section(title):
    st.markdown(f'<div class="sec-header">{title}</div>', unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process(file_bytes):
    df_raw = pd.read_csv(io.BytesIO(file_bytes))
    df = df_raw.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['YearMonth'] = df['Date'].dt.to_period('M')
    income_cats = ['Paycheck', 'Income', 'Transfer']
    df['is_income'] = df['Category'].isin(income_cats)

    monthly_income  = df[df['is_income']].groupby('YearMonth')['Amount'].sum().rename('Income')
    monthly_expense = df[~df['is_income']].groupby('YearMonth')['Amount'].sum().rename('Expenses')

    monthly = pd.DataFrame({'Income': monthly_income, 'Expenses': monthly_expense})
    monthly.index = monthly.index.to_timestamp()
    monthly = monthly.sort_index().ffill().bfill()

    monthly['Savings']         = monthly['Income'] - monthly['Expenses']
    monthly['Savings_Ratio']   = (monthly['Savings'] / monthly['Income']).clip(-5, 5)
    monthly['Expense_Ratio']   = (monthly['Expenses'] / monthly['Income']).clip(0, 5)
    monthly['Net_Cashflow']    = monthly['Income'] - monthly['Expenses']
    monthly['Expenses_capped'] = cap_outliers_mad(monthly['Expenses'])

    for lag in [1, 2, 3, 6]:
        monthly[f'Expense_lag_{lag}'] = monthly['Expenses'].shift(lag)
        monthly[f'Income_lag_{lag}']  = monthly['Income'].shift(lag)

    monthly['Expense_rolling3'] = monthly['Expenses'].rolling(3).mean()
    monthly['Expense_rolling6'] = monthly['Expenses'].rolling(6).mean()
    monthly['Income_rolling3']  = monthly['Income'].rolling(3).mean()
    monthly['month']            = monthly.index.month
    monthly['quarter']          = monthly.index.quarter
    monthly['month_sin']        = np.sin(2 * np.pi * monthly['month'] / 12)
    monthly['month_cos']        = np.cos(2 * np.pi * monthly['month'] / 12)
    monthly['quarter_sin']      = np.sin(2 * np.pi * monthly['quarter'] / 4)
    monthly['quarter_cos']      = np.cos(2 * np.pi * monthly['quarter'] / 4)
    return df, monthly


@st.cache_data(show_spinner=False)
def run_models(file_bytes):
    df, monthly = load_and_process(file_bytes)
    target = monthly['Expenses'].dropna()
    FORECAST_MONTHS = 6
    TEST_SIZE = min(6, len(target) // 5)
    train = target.iloc[:-TEST_SIZE]
    test  = target.iloc[-TEST_SIZE:]
    future_dates = pd.date_range(
        start=target.index[-1] + pd.offsets.MonthBegin(1),
        periods=FORECAST_MONTHS, freq='MS'
    )
    results = {}

    # SARIMA
    try:
        target_log  = np.log1p(target)
        target_diff = target_log.diff().dropna()
        train_t     = target_diff.iloc[:-TEST_SIZE]
        if HAS_PMDARIMA:
            auto_model = auto_arima(
                train_t, seasonal=True, m=12, d=0, D=0,
                max_p=3, max_q=3, max_P=2, max_Q=2,
                information_criterion='aic', stepwise=True,
                suppress_warnings=True, error_action='ignore'
            )
            order, seasonal_order = auto_model.order, auto_model.seasonal_order
        else:
            order, seasonal_order = (1, 0, 1), (1, 0, 0, 12)
        sarima_fit = SARIMAX(
            train_t, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
        sarima_pred_t = sarima_fit.forecast(TEST_SIZE)
        sarima_fore_t = sarima_fit.forecast(TEST_SIZE + FORECAST_MONTHS)
        last_train_log = target_log.iloc[len(train_t)]
        pred_log = last_train_log + sarima_pred_t.cumsum()
        fore_log = target_log.iloc[-1] + sarima_fore_t[-FORECAST_MONTHS:].cumsum()
        sarima_pred = np.expm1(pred_log)
        sarima_fore = pd.Series(np.expm1(fore_log).values, index=future_dates)
        sarima_pred.index = test.index
        results['sarima_pred']    = sarima_pred
        results['sarima_fore']    = sarima_fore
        results['sarima_metrics'] = evaluate(test, sarima_pred)
    except:
        sarima_pred = pd.Series([train.mean()] * TEST_SIZE, index=test.index)
        sarima_fore = pd.Series([train.mean()] * FORECAST_MONTHS, index=future_dates)
        results['sarima_pred']    = sarima_pred
        results['sarima_fore']    = sarima_fore
        results['sarima_metrics'] = {'MAE': 0, 'RMSE': 0, 'MAPE (%)': 0}

    # Holt-Winters
    try:
        sp = 12 if len(train) >= 24 else None
        hw_fit = ExponentialSmoothing(
            train, trend='add', seasonal='add' if sp else None, seasonal_periods=sp
        ).fit(optimized=True)
        hw_pred = hw_fit.forecast(TEST_SIZE)
        hw_fore = hw_fit.forecast(TEST_SIZE + FORECAST_MONTHS)[-FORECAST_MONTHS:]
        results['hw_pred']    = hw_pred
        results['hw_fore']    = np.array(hw_fore)
        results['hw_metrics'] = evaluate(test, hw_pred)
    except:
        hw_pred = pd.Series([train.mean()] * TEST_SIZE, index=test.index)
        results['hw_pred']    = hw_pred
        results['hw_fore']    = np.array([train.mean()] * FORECAST_MONTHS)
        results['hw_metrics'] = {'MAE': 0, 'RMSE': 0, 'MAPE (%)': 0}

    # Prophet
    if HAS_PROPHET:
        try:
            prophet_df = pd.DataFrame({'ds': train.index, 'y': train.values})
            m = Prophet(
                yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                changepoint_prior_scale=0.3, seasonality_prior_scale=15,
                seasonality_mode='multiplicative'
            )
            m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            m.fit(prophet_df)
            future       = m.make_future_dataframe(periods=TEST_SIZE + FORECAST_MONTHS, freq='MS')
            forecast_all = m.predict(future)
            prophet_pred    = forecast_all[forecast_all['ds'].isin(test.index)]['yhat'].values[:len(test)]
            prophet_fore_df = forecast_all.tail(FORECAST_MONTHS)
            results['prophet_pred']    = prophet_pred
            results['prophet_fore_df'] = prophet_fore_df
            results['prophet_metrics'] = evaluate(test, prophet_pred)
        except:
            results['prophet_pred']    = np.array([train.mean()] * TEST_SIZE)
            results['prophet_fore_df'] = pd.DataFrame({'ds': future_dates, 'yhat': [train.mean()] * FORECAST_MONTHS})
            results['prophet_metrics'] = {'MAE': 0, 'RMSE': 0, 'MAPE (%)': 0}
    else:
        results['prophet_pred']    = np.array([train.mean()] * TEST_SIZE)
        results['prophet_fore_df'] = pd.DataFrame({'ds': future_dates, 'yhat': [train.mean()] * FORECAST_MONTHS})
        results['prophet_metrics'] = {'MAE': 0, 'RMSE': 0, 'MAPE (%)': 0}

    # XGBoost
    monthly_ml = monthly.dropna()
    feat_cols  = [c for c in monthly_ml.columns if 'lag' in c or 'rolling' in c] \
               + ['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos']
    feat_cols  = [c for c in feat_cols if c in monthly_ml.columns]
    X = monthly_ml[feat_cols]
    y = monthly_ml['Expenses']
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    try:
        xgb_model = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, random_state=42, verbosity=0, early_stopping_rounds=30
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        xgb_pred = xgb_model.predict(X_test)
        results['xgb_pred']    = xgb_pred
        results['xgb_actual']  = y_test
        results['xgb_metrics'] = evaluate(y_test, xgb_pred)
        results['xgb_fi']      = pd.Series(xgb_model.feature_importances_, index=feat_cols)
    except:
        results['xgb_pred']    = np.array([train.mean()] * len(y_test))
        results['xgb_actual']  = y_test
        results['xgb_metrics'] = {'MAE': 0, 'RMSE': 0, 'MAPE (%)': 0}
        results['xgb_fi']      = pd.Series(np.zeros(len(feat_cols)), index=feat_cols)

    # Ensemble
    mape_s = safe_mape(test, results['sarima_pred'].values[:len(test)])
    mape_h = safe_mape(test, results['hw_pred'].values[:len(test)])
    mape_p = safe_mape(test, results['prophet_pred'][:len(test)])
    inv     = np.array([1/(mape_s+1e-9), 1/(mape_h+1e-9), 1/(mape_p+1e-9)])
    weights = inv / inv.sum()
    ensemble_pred = (
        weights[0] * results['sarima_pred'].values[:len(test)]
        + weights[1] * results['hw_pred'].values[:len(test)]
        + weights[2] * results['prophet_pred'][:len(test)]
    )
    ensemble_fore = (
        weights[0] * results['sarima_fore'].values[:FORECAST_MONTHS]
        + weights[1] * results['hw_fore'][:FORECAST_MONTHS]
        + weights[2] * results['prophet_fore_df']['yhat'].values[:FORECAST_MONTHS]
    )
    results['ensemble_pred']    = ensemble_pred
    results['ensemble_fore']    = ensemble_fore
    results['ensemble_metrics'] = evaluate(test, ensemble_pred)
    results['weights']          = weights
    results['train']            = train
    results['test']             = test
    results['future_dates']     = future_dates
    results['FORECAST_MONTHS']  = FORECAST_MONTHS
    results['TEST_SIZE']        = TEST_SIZE
    return results


def compute_health(monthly):
    def score_row(row):
        sr = row.get('Savings_Ratio', 0)
        sr_score = np.clip(sr / 0.30, 0, 1) * 40
        cv = monthly['Expenses'].std() / (monthly['Expenses'].mean() + 1e-9)
        stability_score = np.clip((1 - cv), 0, 1) * 30
        income_score = 30 if row.get('Income', 0) > 0 else 0
        return round(sr_score + stability_score + income_score, 1)
    monthly = monthly.copy()
    monthly['Health_Score'] = monthly.apply(score_row, axis=1)
    monthly['Risk_Label']   = monthly['Health_Score'].apply(classify_risk)
    return monthly

def classify_risk(score):
    if score >= 80:   return 'Stable'
    elif score >= 50: return 'Moderate Risk'
    else:             return 'High Risk'

def risk_pill(label):
    cls = 'risk-stable' if label == 'Stable' else ('risk-moderate' if 'Moderate' in label else 'risk-high')
    icon = '🟢' if label == 'Stable' else ('🟡' if 'Moderate' in label else '🔴')
    return f'<span class="risk-pill {cls}">{icon} {label}</span>'


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem 0;">
      <div style="font-size:1.4rem; font-weight:700; color:#F1F5F9;">💎 FinAssist</div>
      <div style="font-size:0.75rem; color:#475569; margin-top:0.2rem;">Personal Finance Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#1E293B; margin: 0.75rem 0;">', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload your CSV", type=["csv"], label_visibility="collapsed",
                                help="Upload personal_transactions.csv")
    if uploaded:
        st.markdown(f'<div style="font-size:0.75rem; color:#00C49A; margin-bottom:0.5rem;">✓ {uploaded.name}</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#1E293B; margin: 0.75rem 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#475569; margin-bottom:0.5rem;">Navigation</div>', unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Overview",
        "🔍  Decomposition",
        "🔮  Forecasting",
        "🚨  Anomaly Detection",
        "🏥  Health Score",
        "🏷️  Spending Clusters",
    ], label_visibility="collapsed")

    st.markdown('<hr style="border-color:#1E293B; margin: 0.75rem 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem; color:#334155; text-align:center;">Powered by Prophet · XGBoost · SARIMA</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# WELCOME SCREEN
# ═══════════════════════════════════════════════════════════════════════════════
if uploaded is None:
    st.markdown("""
    <div class="welcome-hero">
      <div style="font-size:3.5rem; margin-bottom:0.5rem;">💎</div>
      <h1>Your Personal Finance Assistant</h1>
      <p>Upload your transaction CSV to get AI-powered forecasts, anomaly alerts,
         health scores, and deep spending insights — all in one place.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    features = [
        ("📊", "Overview", "Income, expenses & savings trends at a glance with interactive hover charts."),
        ("🔮", "AI Forecasting", "SARIMA, Prophet, XGBoost & Ensemble models predict your next 6 months."),
        ("🚨", "Anomaly Detection", "Z-Score + IQR methods flag unusual spending months automatically."),
        ("🏥", "Health Score", "0–100 financial wellness score based on savings rate & income stability."),
        ("🔍", "Decomposition", "Break expenses into trend, seasonality, and residual components."),
        ("🏷️", "Spending Clusters", "K-Means clustering groups your spending patterns by month."),
    ]
    for i, (icon, title, desc) in enumerate(features):
        col = [c1, c2, c3][i % 3]
        with col:
            st.markdown(f"""
            <div class="kpi-card blue" style="margin-bottom:1rem;">
              <div style="font-size:1.5rem; margin-bottom:0.5rem;">{icon}</div>
              <div style="font-weight:700; color:#CBD5E1; margin-bottom:0.3rem;">{title}</div>
              <div style="font-size:0.8rem; color:#475569; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#111827; border-radius:12px; padding:1.25rem 1.5rem; border:1px solid #1E293B; margin-top:1rem;">
      <div style="font-size:0.85rem; font-weight:600; color:#CBD5E1; margin-bottom:0.75rem;">📋 Expected CSV columns</div>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.4rem; font-size:0.8rem; color:#64748B;">
        <div><code style="color:#A78BFA">Date</code> — Transaction date</div>
        <div><code style="color:#A78BFA">Amount</code> — Transaction amount</div>
        <div><code style="color:#A78BFA">Category</code> — Spending category</div>
        <div><code style="color:#A78BFA">Transaction Type</code> — debit / credit</div>
        <div><code style="color:#A78BFA">Account Name</code> — Account label</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Load data ─────────────────────────────────────────────────────────────────
file_bytes = uploaded.read()
with st.spinner("Crunching your numbers…"):
    df, monthly = load_and_process(file_bytes)

income_cats = ['Paycheck', 'Income', 'Transfer']
df['is_income'] = df['Category'].isin(income_cats)
monthly = compute_health(monthly)

avg_income  = monthly['Income'].mean()
avg_expense = monthly['Expenses'].mean()
avg_savings = monthly['Savings'].mean()
avg_sr      = monthly['Savings_Ratio'].mean() * 100
latest_score= monthly['Health_Score'].iloc[-1]
latest_risk = monthly['Risk_Label'].iloc[-1]

date_range  = f"{monthly.index.min().strftime('%b %Y')} – {monthly.index.max().strftime('%b %Y')}"


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown(f'<p class="page-title">🏠 Financial Overview</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="page-sub">{date_range} · {len(monthly)} months of data</p>', unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    sr_badge = f'<span class="kpi-badge-{"good" if avg_sr > 20 else "warn" if avg_sr > 5 else "bad"}">{avg_sr:.1f}% savings rate</span>'
    score_badge = f'<span class="kpi-badge-{"good" if latest_score >= 80 else "warn" if latest_score >= 50 else "bad"}">{latest_risk}</span>'
    with c1: st.markdown(kpi("Avg Monthly Income", f"${avg_income:,.0f}", "past performance", "green"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Avg Monthly Expenses", f"${avg_expense:,.0f}", "past performance", "red"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("Avg Monthly Savings", f"${avg_savings:,.0f}", sr_badge, "blue"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("Health Score", f"{latest_score:.0f}/100", score_badge, "purple"), unsafe_allow_html=True)
    with c5: st.markdown(kpi("Months Tracked", f"{len(monthly)}", date_range, "yellow"), unsafe_allow_html=True)

    # Auto-insights
    section("💡 Quick Insights")
    recent_3   = monthly['Expenses'].tail(3).mean()
    previous_3 = monthly['Expenses'].iloc[-6:-3].mean()
    pct_change = (recent_3 - previous_3) / (previous_3 + 1e-9) * 100
    direction  = "up" if pct_change > 0 else "down"
    monthly_avg = monthly.groupby('month')['Expenses'].mean()
    peak_month  = calendar.month_name[monthly_avg.idxmax()]
    low_savings_pct = (monthly['Savings_Ratio'] < 0.10).mean() * 100
    income_cv   = monthly['Income'].std() / (monthly['Income'].mean() + 1e-9)

    ic1, ic2 = st.columns(2)
    with ic1:
        kind = 'warn' if pct_change > 5 else ('good' if pct_change < -5 else 'warn')
        st.markdown(insight(
            f"Spending is <b>{direction} {abs(pct_change):.1f}%</b> over the last 3 months vs the prior quarter.",
            kind=kind, tag="📈 Trend"
        ), unsafe_allow_html=True)
        st.markdown(insight(
            f"Highest average spending occurs in <b>{peak_month}</b> — plan ahead for seasonal spikes.",
            kind='warn', tag="📅 Seasonality"
        ), unsafe_allow_html=True)
    with ic2:
        kind = 'danger' if low_savings_pct > 30 else ('warn' if low_savings_pct > 10 else 'good')
        st.markdown(insight(
            f"<b>{low_savings_pct:.1f}%</b> of months had a savings rate below 10%. {'Consider automating savings transfers.' if low_savings_pct > 10 else 'Keep it up!'}",
            kind=kind, tag="💰 Savings Risk"
        ), unsafe_allow_html=True)
        kind = 'warn' if income_cv > 0.3 else 'good'
        st.markdown(insight(
            f"Income volatility (CV) = <b>{income_cv:.2f}</b>. {'Irregular income detected — maintain an emergency fund.' if income_cv > 0.3 else 'Income is stable — great foundation.'}",
            kind=kind, tag="⚡ Income Stability"
        ), unsafe_allow_html=True)

    # Main chart
    section("📈 Income · Expenses · Savings")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=('Monthly Income', 'Monthly Expenses', 'Monthly Savings'),
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(
        x=monthly.index, y=monthly['Income'], mode='lines', name='Income',
        line=dict(color=COLORS['income'], width=2), fill='tozeroy',
        fillcolor='rgba(0,196,154,0.08)',
        hovertemplate='%{x|%b %Y}<br>Income: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=monthly.index, y=monthly['Expenses'], mode='lines', name='Expenses',
        line=dict(color=COLORS['expense'], width=2), fill='tozeroy',
        fillcolor='rgba(255,107,107,0.08)',
        hovertemplate='%{x|%b %Y}<br>Expenses: $%{y:,.0f}<extra></extra>'
    ), row=2, col=1)
    sav_colors = [COLORS['income'] if s >= 0 else COLORS['expense'] for s in monthly['Savings']]
    fig.add_trace(go.Bar(
        x=monthly.index, y=monthly['Savings'], name='Savings',
        marker_color=sav_colors, opacity=0.85,
        hovertemplate='%{x|%b %Y}<br>Savings: $%{y:,.0f}<extra></extra>'
    ), row=3, col=1)
    fig.update_layout(**plotly_cfg(), height=600, showlegend=True)
    fig.update_yaxes(title_text='Amount ($)', gridcolor='#1E293B', zerolinecolor='#1E293B')
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        section("🔥 Correlation Heatmap")
        corr_cols = ['Income', 'Expenses', 'Savings', 'Savings_Ratio', 'Expense_Ratio', 'Net_Cashflow']
        cm = monthly[corr_cols].corr().round(2)
        fig2 = go.Figure(go.Heatmap(
            z=cm.values, x=cm.columns.tolist(), y=cm.index.tolist(),
            colorscale='RdYlGn', zmid=0,
            text=cm.values, texttemplate='%{text:.2f}',
            hovertemplate='%{y} × %{x}<br>r = %{z:.2f}<extra></extra>',
        ))
        fig2.update_layout(**plotly_cfg(), height=380)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        section("💸 Top Spending Categories")
        expense_cats = (df[~df['is_income']].groupby('Category')['Amount'].sum()
                        .sort_values(ascending=False).head(10))
        fig3 = go.Figure(go.Bar(
            x=expense_cats.values[::-1], y=expense_cats.index[::-1], orientation='h',
            marker=dict(color=expense_cats.values[::-1], colorscale='Reds_r', showscale=False),
            hovertemplate='<b>%{y}</b><br>Total: $%{x:,.0f}<extra></extra>'
        ))
        fig3.update_layout(**plotly_cfg(), height=380, hovermode='y unified',
                           xaxis_title='Total Spending ($)')
        fig3.update_xaxes(gridcolor='#1E293B')
        fig3.update_yaxes(gridcolor='#1E293B')
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════
elif "Decomposition" in page:
    st.markdown('<p class="page-title">🔍 Time Series Decomposition</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Break down your expense series into trend, seasonality, and residual components.</p>', unsafe_allow_html=True)

    target = monthly['Expenses'].dropna()
    win    = min(6, len(target) // 3)
    adf_result    = adfuller(target.dropna())
    is_stationary = adf_result[1] < 0.05

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(kpi("ADF p-value", f"{adf_result[1]:.4f}", "stationarity test", "blue"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Stationarity", "✅ Stationary" if is_stationary else "⚠️ Non-stationary",
                              "p < 0.05 → stationary", "green" if is_stationary else "red"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("Rolling Window", f"{win} months", "for rolling stats", "purple"), unsafe_allow_html=True)

    if len(target) >= 24:
        section("📊 Seasonal Decomposition")
        decomp = seasonal_decompose(target, model='additive', period=12)
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=('Observed', 'Trend', 'Seasonality', 'Residual'),
                            vertical_spacing=0.05)
        for i, (comp, col, name) in enumerate([
            (decomp.observed, COLORS['expense'],  'Observed'),
            (decomp.trend,    '#7F8C8D',           'Trend'),
            (decomp.seasonal, COLORS['savings'],  'Seasonal'),
            (decomp.resid,    COLORS['anomaly'],  'Residual'),
        ], start=1):
            mode = 'lines' if name != 'Residual' else 'markers'
            fig.add_trace(go.Scatter(
                x=comp.index, y=comp.values, mode=mode, name=name,
                line=dict(color=col, width=2) if name != 'Residual' else None,
                marker=dict(color=col, size=5) if name == 'Residual' else None,
                hovertemplate=f'%{{x|%b %Y}}<br>{name}: $%{{y:,.0f}}<extra></extra>'
            ), row=i, col=1)
        fig.update_layout(**plotly_cfg(), height=700, showlegend=False)
        fig.update_yaxes(gridcolor='#1E293B', zerolinecolor='#334155')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Only {len(target)} months — need ≥ 24 for decomposition.")

    section("📉 Rolling Statistics")
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=(f'{win}-Month Rolling Mean', f'{win}-Month Rolling Std Dev'),
                         vertical_spacing=0.1)
    fig2.add_trace(go.Scatter(x=target.index, y=target.values, mode='lines', name='Actual',
        line=dict(color=COLORS['expense'], width=1.5), opacity=0.5,
        hovertemplate='%{x|%b %Y}<br>Actual: $%{y:,.0f}<extra></extra>'), row=1, col=1)
    fig2.add_trace(go.Scatter(x=target.index, y=target.rolling(win).mean().values, mode='lines',
        name=f'{win}M Mean', line=dict(color='white', width=2),
        hovertemplate='%{x|%b %Y}<br>Rolling Mean: $%{y:,.0f}<extra></extra>'), row=1, col=1)
    fig2.add_trace(go.Scatter(x=target.index, y=target.rolling(win).std().values, mode='lines',
        name='Std Dev', line=dict(color=COLORS['anomaly'], width=2),
        hovertemplate='%{x|%b %Y}<br>Std Dev: $%{y:,.0f}<extra></extra>'), row=2, col=1)
    fig2.update_layout(**plotly_cfg(), height=480)
    fig2.update_yaxes(gridcolor='#1E293B', zerolinecolor='#334155')
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════
elif "Forecasting" in page:
    st.markdown('<p class="page-title">🔮 Expense Forecasting</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">SARIMA · Holt-Winters · Prophet · XGBoost · Weighted Ensemble — 6-month horizon</p>', unsafe_allow_html=True)

    with st.spinner("Training models… this takes about a minute ⏳"):
        res = run_models(file_bytes)

    train        = res['train']
    test         = res['test']
    future_dates = res['future_dates']

    # Model metrics
    section("📊 Model Performance (Test Set)")
    metrics_df = pd.DataFrame({
        'SARIMA':       res['sarima_metrics'],
        'Holt-Winters': res['hw_metrics'],
        'Prophet':      res['prophet_metrics'],
        'XGBoost':      res['xgb_metrics'],
        'Ensemble':     res['ensemble_metrics'],
    }).T.reset_index().rename(columns={'index': 'Model'})
    metrics_df = metrics_df.sort_values('MAPE (%)').reset_index(drop=True)
    best = metrics_df.iloc[0]['Model']

    # Metric KPIs
    mc = st.columns(5)
    for i, row in metrics_df.iterrows():
        color = 'green' if row['Model'] == best else 'blue'
        with mc[i]:
            st.markdown(kpi(row['Model'], f"{row['MAPE (%)']:.1f}%",
                            f"MAE ${row['MAE']:,.0f}", color), unsafe_allow_html=True)

    st.markdown(insight(f"🏆 Best model by MAPE: <b>{best}</b>", kind='good', tag="MODEL SELECTION"), unsafe_allow_html=True)

    # Forecast chart
    section("📈 Forecast Comparison")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values, mode='lines', name='Historical',
        line=dict(color='#334155', width=1.5),
        hovertemplate='%{x|%b %Y}<br>Historical: $%{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, mode='lines+markers', name='Actual',
        line=dict(color='#F1F5F9', dash='dash', width=2),
        hovertemplate='%{x|%b %Y}<br>Actual: $%{y:,.0f}<extra></extra>'))

    model_colors = ['#60A5FA', '#FB923C', '#4ADE80', COLORS['forecast']]
    for (name, pred, fore, color) in [
        ('SARIMA',       res['sarima_pred'].values,  res['sarima_fore'].values,              model_colors[0]),
        ('Holt-Winters', res['hw_pred'].values,      res['hw_fore'],                         model_colors[1]),
        ('Prophet',      res['prophet_pred'],         res['prophet_fore_df']['yhat'].values,  model_colors[2]),
        ('Ensemble',     res['ensemble_pred'],        res['ensemble_fore'],                   model_colors[3]),
    ]:
        fig.add_trace(go.Scatter(x=test.index, y=pred[:len(test)], mode='lines+markers', name=name,
            line=dict(color=color, width=2),
            hovertemplate=f'%{{x|%b %Y}}<br>{name}: $%{{y:,.0f}}<extra></extra>'))
        fig.add_trace(go.Scatter(x=future_dates, y=fore[:len(future_dates)], mode='lines+markers',
            name=f'{name} →', line=dict(color=color, dash='dot', width=1.5),
            hovertemplate=f'%{{x|%b %Y}}<br>{name} Forecast: $%{{y:,.0f}}<extra></extra>'))

    fig.add_vrect(x0=future_dates[0], x1=future_dates[-1],
                  fillcolor='rgba(167,139,250,0.04)', line_width=0,
                  annotation_text="Forecast Zone", annotation_position="top left",
                  annotation_font_color="#A78BFA")

    fig.update_layout(**plotly_cfg(), height=520, yaxis_title='Expenses ($)',
                      xaxis=dict(rangeslider=dict(visible=True), type='date', gridcolor='#1E293B'),
                      yaxis=dict(gridcolor='#1E293B'))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        section("🎯 XGBoost Feature Importance")
        fi = res['xgb_fi'].sort_values(ascending=True)
        fig2 = go.Figure(go.Bar(
            x=fi.values, y=fi.index.tolist(), orientation='h',
            marker=dict(color=fi.values, colorscale='Purples', showscale=False),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))
        fig2.update_layout(**plotly_cfg(), height=380, hovermode='y unified',
                           xaxis_title='Importance Score',
                           xaxis=dict(gridcolor='#1E293B'), yaxis=dict(gridcolor='#1E293B'))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        section("🗓️ 6-Month Risk Report")
        fore_df = res['prophet_fore_df'].copy()
        fore_df['income_forecast'] = monthly['Income'].mean()
        fore_df['savings_forecast'] = fore_df['income_forecast'] - fore_df['yhat']
        fore_df['savings_ratio_forecast'] = fore_df['savings_forecast'] / fore_df['income_forecast']
        for _, row in fore_df.iterrows():
            month_str = pd.to_datetime(row['ds']).strftime('%b %Y')
            exp, sr   = row['yhat'], row['savings_ratio_forecast']
            inc        = row['income_forecast']
            if exp > inc:
                st.markdown(insight(f"<b>{month_str}</b> — ${exp:,.0f} forecast<br>Expenses exceed income", 'danger', '🚨 HIGH RISK'), unsafe_allow_html=True)
            elif exp > 0.85 * inc:
                st.markdown(insight(f"<b>{month_str}</b> — ${exp:,.0f} forecast<br>Expense ratio > 85%", 'warn', '⚠️ WATCH'), unsafe_allow_html=True)
            else:
                st.markdown(insight(f"<b>{month_str}</b> — ${exp:,.0f} forecast<br>Looking healthy", 'good', '✅ HEALTHY'), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
elif "Anomaly" in page:
    st.markdown('<p class="page-title">🚨 Anomaly Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Unusual spending months flagged using Z-Score and IQR methods.</p>', unsafe_allow_html=True)

    monthly = monthly.copy()
    monthly['z_score']        = (monthly['Expenses'] - monthly['Expenses'].mean()) / monthly['Expenses'].std()
    monthly['anomaly_zscore'] = monthly['z_score'].abs() > 2.5
    Q1, Q3 = monthly['Expenses'].quantile(0.25), monthly['Expenses'].quantile(0.75)
    IQR    = Q3 - Q1
    monthly['anomaly_iqr']    = (monthly['Expenses'] < Q1 - 1.5*IQR) | (monthly['Expenses'] > Q3 + 1.5*IQR)
    monthly['anomaly']        = monthly['anomaly_zscore'] | monthly['anomaly_iqr']
    anomaly_df = monthly[monthly['anomaly']]

    c1, c2, c3, c4 = st.columns(4)
    pct_anomaly = len(anomaly_df) / len(monthly) * 100
    with c1: st.markdown(kpi("Anomalies Found", str(len(anomaly_df)), f"{pct_anomaly:.1f}% of months", "red"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Normal Months", str(len(monthly) - len(anomaly_df)), "within expected range", "green"), unsafe_allow_html=True)
    with c3:
        if len(anomaly_df) > 0:
            max_idx = anomaly_df['Expenses'].idxmax()
            st.markdown(kpi("Highest Anomaly", f"${anomaly_df.loc[max_idx,'Expenses']:,.0f}", max_idx.strftime('%b %Y'), "yellow"), unsafe_allow_html=True)
        else:
            st.markdown(kpi("Highest Anomaly", "None", "no anomalies", "yellow"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("Z-Score Threshold", "±2.5σ", "flagging method", "blue"), unsafe_allow_html=True)

    section("📊 Expense Timeline with Anomalies")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly.index, y=monthly['Expenses'], mode='lines', name='Expenses',
        line=dict(color=COLORS['expense'], width=2),
        fill='tozeroy', fillcolor='rgba(255,107,107,0.05)',
        hovertemplate='%{x|%b %Y}<br>Expenses: $%{y:,.0f}<extra></extra>'
    ))
    if len(anomaly_df) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_df.index, y=anomaly_df['Expenses'], mode='markers', name='Anomaly',
            marker=dict(color=COLORS['anomaly'], size=14, symbol='x',
                        line=dict(color='white', width=2)),
            hovertemplate='%{x|%b %Y}<br>⚠️ Anomaly: $%{y:,.0f}<extra></extra>'
        ))
    # Z-score bands
    mean_e = monthly['Expenses'].mean()
    std_e  = monthly['Expenses'].std()
    fig.add_hrect(y0=mean_e - 2.5*std_e, y1=mean_e + 2.5*std_e,
                  fillcolor='rgba(0,196,154,0.04)', line_width=0,
                  annotation_text="Normal range (±2.5σ)", annotation_position="bottom right",
                  annotation_font_color="#00C49A")
    fig.update_layout(**plotly_cfg(), height=430, yaxis_title='Expenses ($)',
                      yaxis=dict(gridcolor='#1E293B'), xaxis=dict(gridcolor='#1E293B'))
    st.plotly_chart(fig, use_container_width=True)

    if len(anomaly_df) > 0:
        section("📋 Anomaly Details")
        display = anomaly_df[['Income', 'Expenses', 'Savings', 'z_score']].copy()
        display.index = display.index.strftime('%b %Y')
        display.columns = ['Income ($)', 'Expenses ($)', 'Savings ($)', 'Z-Score']
        st.dataframe(
            display.style.format({
                'Income ($)': '${:,.0f}', 'Expenses ($)': '${:,.0f}',
                'Savings ($)': '${:,.0f}', 'Z-Score': '{:.2f}'
            }).background_gradient(subset=['Z-Score'], cmap='Reds'),
            use_container_width=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HEALTH SCORE
# ═══════════════════════════════════════════════════════════════════════════════
elif "Health" in page:
    st.markdown('<p class="page-title">🏥 Financial Health Score</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">A 0–100 wellness score based on savings rate, income stability, and cashflow.</p>', unsafe_allow_html=True)

    latest_score = monthly['Health_Score'].iloc[-1]
    latest_risk  = monthly['Risk_Label'].iloc[-1]
    avg_score    = monthly['Health_Score'].mean()
    best_score   = monthly['Health_Score'].max()
    worst_score  = monthly['Health_Score'].min()

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi("Current Score", f"{latest_score:.0f}/100",
                              risk_pill(latest_risk), "green" if latest_score >= 80 else "yellow" if latest_score >= 50 else "red"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Average Score", f"{avg_score:.1f}/100", "all-time average", "blue"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("Best Month", f"{best_score:.0f}/100", monthly['Health_Score'].idxmax().strftime('%b %Y'), "green"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("Worst Month", f"{worst_score:.0f}/100", monthly['Health_Score'].idxmin().strftime('%b %Y'), "red"), unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        section("📊 Health Score Over Time")
        hs_colors = ['#00C49A' if s >= 80 else '#FFD93D' if s >= 50 else '#FF6B6B'
                     for s in monthly['Health_Score']]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=('Financial Health Score', 'Savings Rate (%)'),
                            vertical_spacing=0.1)
        fig.add_trace(go.Bar(x=monthly.index, y=monthly['Health_Score'], name='Health Score',
            marker_color=hs_colors, opacity=0.9,
            hovertemplate='%{x|%b %Y}<br>Score: %{y:.1f}/100<extra></extra>'), row=1, col=1)
        fig.add_hline(y=80, line=dict(color='#00C49A', dash='dash', width=1),
                      annotation_text='Stable ≥ 80', annotation_font_color='#00C49A', row=1, col=1)
        fig.add_hline(y=50, line=dict(color='#FFD93D', dash='dash', width=1),
                      annotation_text='Moderate ≥ 50', annotation_font_color='#FFD93D', row=1, col=1)
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly['Savings_Ratio'] * 100,
            mode='lines', name='Savings Rate', line=dict(color=COLORS['savings'], width=2),
            fill='tozeroy', fillcolor='rgba(78,205,196,0.06)',
            hovertemplate='%{x|%b %Y}<br>Savings Rate: %{y:.1f}%<extra></extra>'), row=2, col=1)
        fig.add_hline(y=0, line=dict(color='#FF6B6B', dash='dash', width=0.8), row=2, col=1)
        fig.add_hline(y=20, line=dict(color='#00C49A', dash='dot', width=0.8),
                      annotation_text='Target 20%', annotation_font_color='#00C49A', row=2, col=1)
        fig.update_yaxes(title_text='Score (0–100)', range=[0, 108], row=1, col=1, gridcolor='#1E293B')
        fig.update_yaxes(title_text='Savings Rate (%)', row=2, col=1, gridcolor='#1E293B')
        fig.update_xaxes(gridcolor='#1E293B')
        fig.update_layout(**plotly_cfg(), height=560)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("🎯 Health Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode='gauge+number+delta',
            value=latest_score,
            delta={'reference': avg_score, 'valueformat': '.1f'},
            title={'text': 'Current Score', 'font': {'color': '#94A3B8', 'size': 14}},
            number={'font': {'color': '#F1F5F9', 'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#334155', 'tickfont': {'color': '#64748B'}},
                'bar': {'color': '#00C49A' if latest_score >= 80 else '#FFD93D' if latest_score >= 50 else '#FF6B6B', 'thickness': 0.25},
                'bgcolor': '#111827',
                'bordercolor': '#1E293B',
                'steps': [
                    {'range': [0, 50],  'color': 'rgba(255,107,107,0.12)'},
                    {'range': [50, 80], 'color': 'rgba(255,217,61,0.12)'},
                    {'range': [80, 100],'color': 'rgba(0,196,154,0.12)'},
                ],
                'threshold': {
                    'line': {'color': '#F1F5F9', 'width': 2},
                    'thickness': 0.75,
                    'value': avg_score
                }
            }
        ))
        fig_gauge.update_layout(**{**plotly_cfg(), 'hovermode': False}, height=280,
                                margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        section("📋 Risk Distribution")
        risk_counts = monthly['Risk_Label'].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=risk_counts.index.tolist(),
            values=risk_counts.values,
            hole=0.55,
            marker=dict(colors=['#00C49A', '#FFD93D', '#FF6B6B'],
                        line=dict(color='#080C14', width=2)),
            hovertemplate='<b>%{label}</b><br>%{value} months (%{percent})<extra></extra>',
        ))
        fig_pie.update_layout(**{**plotly_cfg(), 'hovermode': False}, height=240,
                              showlegend=True, legend=dict(font=dict(color='#94A3B8')),
                              margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    section("📋 Recent 12-Month Risk Table")
    display = monthly[['Income', 'Expenses', 'Savings_Ratio', 'Health_Score', 'Risk_Label']].tail(12).copy()
    display.index = display.index.strftime('%b %Y')
    display.columns = ['Income ($)', 'Expenses ($)', 'Savings Rate', 'Health Score', 'Status']
    st.dataframe(
        display.style.format({
            'Income ($)': '${:,.0f}', 'Expenses ($)': '${:,.0f}',
            'Savings Rate': '{:.1%}', 'Health Score': '{:.1f}'
        }).background_gradient(subset=['Health Score'], cmap='RdYlGn', vmin=0, vmax=100),
        use_container_width=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SPENDING CLUSTERS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Cluster" in page:
    st.markdown('<p class="page-title">🏷️ Spending Clusters</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">K-Means clustering groups months by category spending patterns.</p>', unsafe_allow_html=True)

    cat_pivot = (
        df[~df['is_income']]
        .assign(YM=lambda x: x['Date'].dt.to_period('M'))
        .groupby(['YM', 'Category'])['Amount'].sum()
        .unstack(fill_value=0)
    )
    scaler     = StandardScaler()
    cat_scaled = scaler.fit_transform(cat_pivot.fillna(0))
    best_k     = 3
    km_final   = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cat_pivot['Cluster'] = km_final.fit_predict(cat_scaled)
    cluster_profiles     = cat_pivot.groupby('Cluster').mean().T

    inertias = []
    K_range  = range(2, min(7, len(cat_pivot)))
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(cat_scaled)
        inertias.append(km.inertia_)

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(kpi("Clusters", str(best_k), "spending personas", "purple"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Months Analysed", str(len(cat_pivot)), "total data points", "blue"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("Categories", str(len(cluster_profiles)), "spending dimensions", "green"), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        section("📐 Elbow Chart (Optimal K)")
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(K_range), y=inertias, mode='lines+markers',
            line=dict(color=COLORS['savings'], width=2), marker=dict(size=9, color=COLORS['savings']),
            fill='tozeroy', fillcolor='rgba(78,205,196,0.06)',
            hovertemplate='K = %{x}<br>Inertia: %{y:,.0f}<extra></extra>'
        ))
        fig_elbow.add_vline(x=best_k, line=dict(color=COLORS['anomaly'], dash='dash'),
                            annotation_text=f'Chosen K={best_k}', annotation_font_color=COLORS['anomaly'])
        fig_elbow.update_layout(**plotly_cfg(), height=320,
                                xaxis_title='Number of Clusters', yaxis_title='Inertia',
                                xaxis=dict(gridcolor='#1E293B'), yaxis=dict(gridcolor='#1E293B'))
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col2:
        section("🥧 Months per Cluster")
        cluster_counts = cat_pivot['Cluster'].value_counts().sort_index()
        cluster_colors = ['#A78BFA', '#4ECDC4', '#FFD93D']
        fig_pie = go.Figure(go.Pie(
            labels=[f'Cluster {i}' for i in cluster_counts.index],
            values=cluster_counts.values, hole=0.5,
            marker=dict(colors=cluster_colors[:best_k], line=dict(color='#080C14', width=2)),
            hovertemplate='<b>%{label}</b><br>%{value} months (%{percent})<extra></extra>',
        ))
        fig_pie.update_layout(**{**plotly_cfg(), 'hovermode': False}, height=320,
                              showlegend=True, legend=dict(font=dict(color='#94A3B8')),
                              margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    section("📊 Category Spending by Cluster")
    fig_bar = go.Figure()
    for i, col in enumerate(cluster_profiles.columns):
        fig_bar.add_trace(go.Bar(
            name=f'Cluster {col}',
            x=cluster_profiles.index.tolist(),
            y=cluster_profiles[col].values,
            marker_color=cluster_colors[i % len(cluster_colors)],
            hovertemplate='<b>%{x}</b><br>Cluster ' + str(col) + ': $%{y:,.1f}<extra></extra>'
        ))
    fig_bar.update_layout(**plotly_cfg(), barmode='group', height=430,
                          xaxis_title='Category', yaxis_title='Avg Monthly Spend ($)',
                          xaxis=dict(tickangle=-35, gridcolor='#1E293B'),
                          yaxis=dict(gridcolor='#1E293B'),
                          legend=dict(font=dict(color='#94A3B8')))
    st.plotly_chart(fig_bar, use_container_width=True)