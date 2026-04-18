import streamlit as st
import pandas as pd
import numpy as np
import warnings
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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Time Series Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    'income':   '#2ECC71',
    'expense':  '#E74C3C',
    'savings':  '#3498DB',
    'anomaly':  '#E67E22',
    'forecast': '#9B59B6',
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        border-left: 4px solid;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; color: #aaa; }
    .metric-card p  { margin: 0; font-size: 1.6rem; font-weight: 700; color: #fff; }
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #fff;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #2e3250;
    }
    .stAlert { border-radius: 8px; }
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

@st.cache_data(show_spinner=False)
def load_and_process(file_bytes):
    df_raw = pd.read_csv(file_bytes)
    df = df_raw.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['YearMonth'] = df['Date'].dt.to_period('M')

    income_cats = ['Paycheck', 'Income', 'Transfer']
    df['is_income'] = df['Category'].isin(income_cats)

    monthly_income  = df[df['is_income']].groupby('YearMonth')['Amount'].sum().rename('Income')
    monthly_expense = df[~df['is_income']].groupby('YearMonth')['Amount'].sum().rename('Expenses')

    monthly = pd.DataFrame({'Income': monthly_income, 'Expenses': monthly_expense})
    monthly.index = monthly.index.to_timestamp()
    monthly = monthly.sort_index().fillna(method='ffill').fillna(method='bfill')

    monthly['Savings']          = monthly['Income'] - monthly['Expenses']
    monthly['Savings_Ratio']    = (monthly['Savings'] / monthly['Income']).clip(-5, 5)
    monthly['Expense_Ratio']    = (monthly['Expenses'] / monthly['Income']).clip(0, 5)
    monthly['Net_Cashflow']     = monthly['Income'] - monthly['Expenses']
    monthly['Expenses_capped']  = cap_outliers_mad(monthly['Expenses'])

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
def run_models(monthly_bytes):
    # Re-derive monthly from bytes (cache key must be serialisable)
    import io
    df, monthly = load_and_process(io.BytesIO(monthly_bytes))

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

    # ── SARIMA ────────────────────────────────────────────────────────────────
    try:
        target_log  = np.log1p(target)
        target_diff = target_log.diff().dropna()
        train_t = target_diff.iloc[:-TEST_SIZE]
        test_t  = target_diff.iloc[-TEST_SIZE:]

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
        results['sarima_pred'] = sarima_pred
        results['sarima_fore'] = sarima_fore
        results['sarima_metrics'] = evaluate(test, sarima_pred)
    except Exception as e:
        sarima_pred = pd.Series([train.mean()] * TEST_SIZE, index=test.index)
        sarima_fore = pd.Series([train.mean()] * FORECAST_MONTHS, index=future_dates)
        results['sarima_pred'] = sarima_pred
        results['sarima_fore'] = sarima_fore
        results['sarima_metrics'] = {'MAE': 0, 'RMSE': 0, 'MAPE (%)': 0}

    # ── Holt-Winters ──────────────────────────────────────────────────────────
    try:
        sp = 12 if len(train) >= 24 else None
        hw_fit  = ExponentialSmoothing(
            train, trend='add',
            seasonal='add' if sp else None,
            seasonal_periods=sp
        ).fit(optimized=True)
        hw_pred = hw_fit.forecast(TEST_SIZE)
        hw_fore = hw_fit.forecast(TEST_SIZE + FORECAST_MONTHS)[-FORECAST_MONTHS:]
        results['hw_pred'] = hw_pred
        results['hw_fore'] = np.array(hw_fore)
        results['hw_metrics'] = evaluate(test, hw_pred)
    except:
        hw_pred = pd.Series([train.mean()] * TEST_SIZE, index=test.index)
        results['hw_pred'] = hw_pred
        results['hw_fore'] = np.array([train.mean()] * FORECAST_MONTHS)
        results['hw_metrics'] = {'MAE': 0, 'RMSE': 0, 'MAPE (%)': 0}

    # ── Prophet ───────────────────────────────────────────────────────────────
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

    # ── XGBoost ───────────────────────────────────────────────────────────────
    monthly_ml = monthly.dropna()
    feat_cols = (
        [c for c in monthly_ml.columns if 'lag' in c or 'rolling' in c]
        + ['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos']
    )
    feat_cols = [c for c in feat_cols if c in monthly_ml.columns]
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

    # ── Ensemble ──────────────────────────────────────────────────────────────
    mape_s = safe_mape(test, results['sarima_pred'].values[:len(test)])
    mape_h = safe_mape(test, results['hw_pred'].values[:len(test)])
    mape_p = safe_mape(test, results['prophet_pred'][:len(test)])
    inv    = np.array([1/(mape_s+1e-9), 1/(mape_h+1e-9), 1/(mape_p+1e-9)])
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


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("💰 FinTS Dashboard")
    st.markdown("---")
    uploaded = st.file_uploader("Upload `personal_transactions.csv`", type=["csv"])
    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio("", [
        "📊 Overview",
        "🔍 Decomposition",
        "🔮 Forecasting",
        "🚨 Anomaly Detection",
        "🏥 Health Score",
        "🏷️ Spending Clusters",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Built with Streamlit · Plotly · Prophet · XGBoost")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if uploaded is None:
    st.title("💰 Financial Time Series Dashboard")
    st.info("👈 Upload your `personal_transactions.csv` in the sidebar to get started.")
    st.markdown("""
    **Expected CSV columns:**
    | Column | Description |
    |---|---|
    | `Date` | Transaction date |
    | `Amount` | Transaction amount |
    | `Category` | Spending category |
    | `Transaction Type` | debit / credit |
    | `Account Name` | Account label |
    """)
    st.stop()

# Load data
file_bytes = uploaded.read()
import io
with st.spinner("Processing data…"):
    df, monthly = load_and_process(io.BytesIO(file_bytes))

income_cats = ['Paycheck', 'Income', 'Transfer']
df['is_income'] = df['Category'].isin(income_cats)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 Financial Overview")

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    avg_income   = monthly['Income'].mean()
    avg_expense  = monthly['Expenses'].mean()
    avg_savings  = monthly['Savings'].mean()
    avg_sr       = monthly['Savings_Ratio'].mean() * 100

    with c1:
        st.metric("Avg Monthly Income",   f"${avg_income:,.0f}")
    with c2:
        st.metric("Avg Monthly Expenses", f"${avg_expense:,.0f}")
    with c3:
        st.metric("Avg Monthly Savings",  f"${avg_savings:,.0f}")
    with c4:
        st.metric("Avg Savings Rate",     f"{avg_sr:.1f}%")

    st.markdown('<div class="section-header">Income · Expenses · Savings Over Time</div>', unsafe_allow_html=True)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=('Monthly Income', 'Monthly Expenses', 'Monthly Savings'),
                        vertical_spacing=0.07)
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly['Income'], mode='lines', name='Income',
        line=dict(color=COLORS['income'], width=2), fill='tozeroy', fillcolor='rgba(46,204,113,0.12)',
        hovertemplate='%{x|%b %Y}<br>Income: $%{y:,.0f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly['Expenses'], mode='lines', name='Expenses',
        line=dict(color=COLORS['expense'], width=2), fill='tozeroy', fillcolor='rgba(231,76,60,0.12)',
        hovertemplate='%{x|%b %Y}<br>Expenses: $%{y:,.0f}<extra></extra>'), row=2, col=1)
    sav_colors = [COLORS['income'] if s >= 0 else COLORS['expense'] for s in monthly['Savings']]
    fig.add_trace(go.Bar(x=monthly.index, y=monthly['Savings'], name='Savings',
        marker_color=sav_colors, opacity=0.8,
        hovertemplate='%{x|%b %Y}<br>Savings: $%{y:,.0f}<extra></extra>'), row=3, col=1)
    fig.update_layout(hovermode='x unified', template='plotly_dark', height=620, showlegend=True)
    fig.update_yaxes(title_text='Amount ($)')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    corr_cols = ['Income', 'Expenses', 'Savings', 'Savings_Ratio', 'Expense_Ratio', 'Net_Cashflow']
    cm = monthly[corr_cols].corr().round(2)
    fig2 = go.Figure(go.Heatmap(
        z=cm.values, x=cm.columns.tolist(), y=cm.index.tolist(),
        colorscale='RdYlGn', zmid=0,
        text=cm.values, texttemplate='%{text:.2f}',
        hovertemplate='%{y} × %{x}<br>r = %{z:.2f}<extra></extra>',
    ))
    fig2.update_layout(template='plotly_dark', height=400)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Top 10 Spending Categories</div>', unsafe_allow_html=True)
    expense_cats = (df[~df['is_income']].groupby('Category')['Amount'].sum()
                    .sort_values(ascending=False).head(10))
    fig3 = go.Figure(go.Bar(
        x=expense_cats.values[::-1], y=expense_cats.index[::-1], orientation='h',
        marker=dict(color=expense_cats.values[::-1], colorscale='Reds', showscale=False),
        hovertemplate='<b>%{y}</b><br>Total: $%{x:,.0f}<extra></extra>'
    ))
    fig3.update_layout(template='plotly_dark', height=400, hovermode='y unified',
                       xaxis_title='Total Spending ($)')
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Decomposition":
    st.title("🔍 Time Series Decomposition")
    target = monthly['Expenses'].dropna()
    win    = min(6, len(target) // 3)

    # ADF test
    adf_result = adfuller(target.dropna())
    is_stationary = adf_result[1] < 0.05
    st.info(f"**ADF Test** — p-value: `{adf_result[1]:.4f}` → "
            f"{'✅ Stationary' if is_stationary else '⚠️ Non-stationary (differencing may help)'}")

    if len(target) >= 24:
        decomp = seasonal_decompose(target, model='additive', period=12)
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=('Observed', 'Trend', 'Seasonality', 'Residual'),
                            vertical_spacing=0.06)
        for i, (comp, col, name) in enumerate([
            (decomp.observed, COLORS['expense'],  'Observed'),
            (decomp.trend,    '#7f8c8d',           'Trend'),
            (decomp.seasonal, COLORS['savings'],  'Seasonal'),
            (decomp.resid,    COLORS['anomaly'],  'Residual'),
        ], start=1):
            mode = 'lines' if name != 'Residual' else 'markers'
            fig.add_trace(go.Scatter(x=comp.index, y=comp.values, mode=mode, name=name,
                line=dict(color=col, width=2) if name != 'Residual' else None,
                marker=dict(color=col, size=5) if name == 'Residual' else None,
                hovertemplate=f'%{{x|%b %Y}}<br>{name}: $%{{y:,.0f}}<extra></extra>'), row=i, col=1)
        fig.update_layout(hovermode='x unified', template='plotly_dark', height=750, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Only {len(target)} months available — need ≥ 24 for decomposition.")

    st.markdown('<div class="section-header">Rolling Statistics</div>', unsafe_allow_html=True)
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
    fig2.update_layout(hovermode='x unified', template='plotly_dark', height=500)
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FORECASTING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Forecasting":
    st.title("🔮 Expense Forecasting")

    with st.spinner("Running models… this may take a minute ⏳"):
        res = run_models(file_bytes)

    train        = res['train']
    test         = res['test']
    future_dates = res['future_dates']

    # ── Model metrics ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Performance (Test Set)</div>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame({
        'SARIMA':           res['sarima_metrics'],
        'Holt-Winters':     res['hw_metrics'],
        'Prophet':          res['prophet_metrics'],
        'XGBoost':          res['xgb_metrics'],
        'Ensemble':         res['ensemble_metrics'],
    }).T.reset_index().rename(columns={'index': 'Model'})
    metrics_df = metrics_df.sort_values('MAPE (%)').reset_index(drop=True)
    st.dataframe(metrics_df.style.highlight_min(subset=['MAPE (%)','MAE','RMSE'],
                 color='#1a472a'), use_container_width=True)

    best = metrics_df.iloc[0]['Model']
    st.success(f"🏆 Best model by MAPE: **{best}**")

    # ── Forecast chart ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Forecast Comparison</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values, mode='lines', name='Train',
        line=dict(color='#636e72', width=1.5),
        hovertemplate='%{x|%b %Y}<br>Train: $%{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, mode='lines+markers', name='Actual',
        line=dict(color='white', dash='dash', width=2),
        hovertemplate='%{x|%b %Y}<br>Actual: $%{y:,.0f}<extra></extra>'))
    for name, pred, fore, color in [
        ('SARIMA',       res['sarima_pred'].values,   res['sarima_fore'].values,   '#3498DB'),
        ('Holt-Winters', res['hw_pred'].values,       res['hw_fore'],              '#E67E22'),
        ('Prophet',      res['prophet_pred'],          res['prophet_fore_df']['yhat'].values, '#2ECC71'),
        ('Ensemble',     res['ensemble_pred'],         res['ensemble_fore'],        COLORS['forecast']),
    ]:
        fig.add_trace(go.Scatter(x=test.index, y=pred[:len(test)], mode='lines+markers', name=name,
            line=dict(color=color),
            hovertemplate=f'%{{x|%b %Y}}<br>{name}: $%{{y:,.0f}}<extra></extra>'))
        fig.add_trace(go.Scatter(x=future_dates, y=fore[:len(future_dates)], mode='lines+markers',
            name=f'{name} Forecast', line=dict(color=color, dash='dot'),
            hovertemplate=f'%{{x|%b %Y}}<br>{name} Forecast: $%{{y:,.0f}}<extra></extra>'))
    fig.update_layout(hovermode='x unified', template='plotly_dark', height=520,
                      xaxis=dict(rangeslider=dict(visible=True), type='date'),
                      yaxis_title='Expenses ($)', title='All-Model Forecast Comparison')
    st.plotly_chart(fig, use_container_width=True)

    # ── XGBoost feature importance ─────────────────────────────────────────────
    st.markdown('<div class="section-header">XGBoost Feature Importance</div>', unsafe_allow_html=True)
    fi = res['xgb_fi'].sort_values(ascending=True)
    fig2 = go.Figure(go.Bar(x=fi.values, y=fi.index.tolist(), orientation='h',
        marker_color=COLORS['forecast'],
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'))
    fig2.update_layout(template='plotly_dark', height=400, hovermode='y unified',
                       xaxis_title='Importance Score')
    st.plotly_chart(fig2, use_container_width=True)

    # ── 6-month risk report ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">6-Month Forward Risk Report</div>', unsafe_allow_html=True)
    avg_income = monthly['Income'].mean()
    fore_df    = res['prophet_fore_df'].copy()
    fore_df['income_forecast']       = avg_income
    fore_df['savings_forecast']      = avg_income - fore_df['yhat']
    fore_df['savings_ratio_forecast'] = fore_df['savings_forecast'] / avg_income

    for _, row in fore_df.iterrows():
        month_str = pd.to_datetime(row['ds']).strftime('%b %Y')
        exp, sr = row['yhat'], row['savings_ratio_forecast']
        if exp > avg_income:
            st.error(f"**{month_str}** — Forecast: ${exp:,.0f} 🚨 Expenses exceed income!")
        elif exp > 0.85 * avg_income:
            st.warning(f"**{month_str}** — Forecast: ${exp:,.0f} ⚠️ Expense ratio > 85%")
        else:
            st.success(f"**{month_str}** — Forecast: ${exp:,.0f} ✅ Healthy")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🚨 Anomaly Detection":
    st.title("🚨 Anomaly Detection")

    monthly['z_score']       = (monthly['Expenses'] - monthly['Expenses'].mean()) / monthly['Expenses'].std()
    monthly['anomaly_zscore'] = monthly['z_score'].abs() > 2.5
    Q1, Q3 = monthly['Expenses'].quantile(0.25), monthly['Expenses'].quantile(0.75)
    IQR = Q3 - Q1
    monthly['anomaly_iqr']   = (monthly['Expenses'] < Q1 - 1.5*IQR) | (monthly['Expenses'] > Q3 + 1.5*IQR)
    iso = IsolationForest(contamination=0.1, random_state=42)
    monthly['anomaly_if']    = iso.fit_predict(monthly[['Expenses']].fillna(0)) == -1
    monthly['anomaly']       = monthly['anomaly_zscore'] | monthly['anomaly_iqr']
    anomaly_df = monthly[monthly['anomaly']]

    st.metric("Anomalies Detected", len(anomaly_df))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly['Expenses'], mode='lines+markers',
        name='Expenses', line=dict(color=COLORS['expense'], width=2),
        hovertemplate='%{x|%b %Y}<br>Expenses: $%{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=anomaly_df.index, y=anomaly_df['Expenses'], mode='markers',
        name='Anomaly', marker=dict(color=COLORS['anomaly'], size=14,
        symbol='x', line=dict(color='white', width=1.5)),
        hovertemplate='%{x|%b %Y}<br>⚠️ Anomaly: $%{y:,.0f}<extra></extra>'))
    fig.update_layout(hovermode='x unified', template='plotly_dark', height=450,
                      title='Expense Anomalies (Z-Score + IQR)', yaxis_title='Expenses ($)')
    st.plotly_chart(fig, use_container_width=True)

    if len(anomaly_df) > 0:
        st.markdown('<div class="section-header">Anomaly Details</div>', unsafe_allow_html=True)
        st.dataframe(
            anomaly_df[['Income', 'Expenses', 'Savings', 'z_score']]
            .rename(columns={'z_score': 'Z-Score'}).round(2),
            use_container_width=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HEALTH SCORE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🏥 Health Score":
    st.title("🏥 Financial Health Score")

    def compute_health_score(row):
        sr = row.get('Savings_Ratio', 0)
        sr_score = np.clip(sr / 0.30, 0, 1) * 40
        cv = monthly['Expenses'].std() / (monthly['Expenses'].mean() + 1e-9)
        stability_score = np.clip((1 - cv), 0, 1) * 30
        income_score = 30 if row.get('Income', 0) > 0 else 0
        return round(sr_score + stability_score + income_score, 1)

    monthly['Health_Score'] = monthly.apply(compute_health_score, axis=1)

    def classify_risk(score):
        if score >= 80:   return 'Stable 🟢'
        elif score >= 50: return 'Moderate Risk 🟡'
        else:             return 'High Risk 🔴'

    monthly['Risk_Label'] = monthly['Health_Score'].apply(classify_risk)

    latest_score = monthly['Health_Score'].iloc[-1]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Latest Health Score", f"{latest_score}/100")
    with c2:
        st.metric("Risk Status", classify_risk(latest_score))
    with c3:
        st.metric("Avg Health Score", f"{monthly['Health_Score'].mean():.1f}/100")

    hs_colors = ['#2ECC71' if s >= 80 else '#F39C12' if s >= 50 else '#E74C3C'
                 for s in monthly['Health_Score']]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('Financial Health Score', 'Savings Rate (%)'),
                        vertical_spacing=0.1)
    fig.add_trace(go.Bar(x=monthly.index, y=monthly['Health_Score'], name='Health Score',
        marker_color=hs_colors,
        hovertemplate='%{x|%b %Y}<br>Score: %{y:.1f}<extra></extra>'), row=1, col=1)
    fig.add_hline(y=80, line=dict(color='#2ECC71', dash='dash'), annotation_text='Stable (80)', row=1, col=1)
    fig.add_hline(y=50, line=dict(color='#F39C12', dash='dash'), annotation_text='Moderate (50)', row=1, col=1)
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly['Savings_Ratio'] * 100,
        mode='lines', name='Savings Rate', line=dict(color=COLORS['savings'], width=2),
        hovertemplate='%{x|%b %Y}<br>Savings Rate: %{y:.1f}%<extra></extra>'), row=2, col=1)
    fig.add_hline(y=0, line=dict(color='red', dash='dash', width=0.8), row=2, col=1)
    fig.update_yaxes(title_text='Score (0–100)', range=[0, 105], row=1, col=1)
    fig.update_yaxes(title_text='Savings Rate (%)', row=2, col=1)
    fig.update_layout(hovermode='x unified', template='plotly_dark', height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Month-by-Month Risk Table (Last 12)</div>', unsafe_allow_html=True)
    st.dataframe(
        monthly[['Income', 'Expenses', 'Savings_Ratio', 'Health_Score', 'Risk_Label']]
        .tail(12).rename(columns={'Savings_Ratio': 'Savings Rate'})
        .style.format({'Income': '${:,.0f}', 'Expenses': '${:,.0f}',
                       'Savings Rate': '{:.1%}', 'Health_Score': '{:.1f}'}),
        use_container_width=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SPENDING CLUSTERS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🏷️ Spending Clusters":
    st.title("🏷️ Spending Clusters")

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

    # Elbow chart
    inertias = []
    K_range = range(2, min(7, len(cat_pivot)))
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(cat_scaled)
        inertias.append(km.inertia_)

    c1, c2 = st.columns(2)
    with c1:
        fig_elbow = go.Figure(go.Scatter(x=list(K_range), y=inertias, mode='lines+markers',
            line=dict(color=COLORS['savings'], width=2), marker=dict(size=8),
            hovertemplate='K=%{x}<br>Inertia: %{y:,.0f}<extra></extra>'))
        fig_elbow.add_vline(x=best_k, line=dict(color='yellow', dash='dash'),
                            annotation_text=f'Best K={best_k}')
        fig_elbow.update_layout(title='Elbow Chart', xaxis_title='# Clusters',
                                yaxis_title='Inertia', template='plotly_dark', height=350)
        st.plotly_chart(fig_elbow, use_container_width=True)

    with c2:
        cluster_counts = cat_pivot['Cluster'].value_counts().sort_index()
        fig_pie = go.Figure(go.Pie(
            labels=[f'Cluster {i}' for i in cluster_counts.index],
            values=cluster_counts.values,
            hole=0.4,
            hovertemplate='<b>%{label}</b><br>Months: %{value}<br>Share: %{percent}<extra></extra>',
            marker=dict(colors=px.colors.qualitative.Set2[:best_k])
        ))
        fig_pie.update_layout(title='Months per Cluster', template='plotly_dark', height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="section-header">Cluster Spending Profiles</div>', unsafe_allow_html=True)
    fig_bar = go.Figure()
    cluster_colors = px.colors.qualitative.Set2
    for i, col in enumerate(cluster_profiles.columns):
        fig_bar.add_trace(go.Bar(
            name=f'Cluster {col}',
            x=cluster_profiles.index.tolist(),
            y=cluster_profiles[col].values,
            marker_color=cluster_colors[i % len(cluster_colors)],
            hovertemplate='<b>%{x}</b><br>Cluster ' + str(col) + ': $%{y:,.1f}<extra></extra>'
        ))
    fig_bar.update_layout(barmode='group', template='plotly_dark', height=480,
                          hovermode='x unified', xaxis_tickangle=-40,
                          xaxis_title='Category', yaxis_title='Avg Monthly Spend ($)',
                          title='Category Spending by Cluster')
    st.plotly_chart(fig_bar, use_container_width=True)
