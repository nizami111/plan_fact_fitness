from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import pandas as pd
import math

app = FastAPI(title='Fitness Club Plan-Fact API', version='1.0.0')

EXPECTED_COLUMNS = [
    'period', 'club',
    'revenue_plan', 'revenue_fact',
    'members_start_plan', 'members_end_plan',
    'members_start_fact', 'members_end_fact',
    'arpu_plan', 'arpu_fact',
    'new_sales_plan', 'new_sales_fact',
    'churn_plan', 'churn_fact'
]

class AnalyzeRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., description='Rows from Google Sheets / n8n')
    period_from: Optional[str] = None
    period_to: Optional[str] = None
    clubs: Optional[List[str]] = None
    top_n_alerts: int = 10


def normalize_col_name(col: str) -> str:
    return (
        str(col).strip().lower()
        .replace(' ', '_')
        .replace('-', '_')
        .replace('/', '_')
    )


def to_float(val: Any) -> Optional[float]:
    if val is None or val == '':
        return None
    if isinstance(val, (int, float)):
        if isinstance(val, float) and math.isnan(val):
            return None
        return float(val)
    s = str(val).strip().replace('\xa0', '').replace(' ', '').replace(',', '.')
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def pct_or_none(num: float, den: float) -> Optional[float]:
    if den in (None, 0) or pd.isna(den):
        return None
    return num / den


def round_or_none(val: Optional[float], digits: int = 4):
    if val is None or pd.isna(val):
        return None
    return round(float(val), digits)


def to_period_string(x: Any) -> Optional[str]:
    if x is None or x == '':
        return None
    try:
        ts = pd.to_datetime(x, errors='coerce')
        if pd.isna(ts):
            return None
        return ts.strftime('%Y-%m')
    except Exception:
        return None


def load_and_validate(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        raise HTTPException(status_code=400, detail='rows is empty')

    df = pd.DataFrame(rows)
    df.columns = [normalize_col_name(c) for c in df.columns]

    aliases = {
        'month': 'period',
        'date': 'period',
        'club_name': 'club',
    }
    df = df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                'message': 'Missing expected columns',
                'missing_columns': missing,
                'expected_columns': EXPECTED_COLUMNS,
            },
        )

    df['period'] = df['period'].apply(to_period_string)
    df['club'] = df['club'].astype(str).str.strip()

    numeric_cols = [c for c in EXPECTED_COLUMNS if c not in ['period', 'club']]
    for col in numeric_cols:
        df[col] = df[col].apply(to_float)

    if df['period'].isna().any():
        raise HTTPException(status_code=400, detail='Could not parse one or more period values')

    return df


def apply_filters(df: pd.DataFrame, period_from: Optional[str], period_to: Optional[str], clubs: Optional[List[str]]) -> pd.DataFrame:
    out = df.copy()
    if period_from:
        out = out[out['period'] >= period_from]
    if period_to:
        out = out[out['period'] <= period_to]
    if clubs:
        clubs_set = {str(c).strip() for c in clubs}
        out = out[out['club'].isin(clubs_set)]
    if out.empty:
        raise HTTPException(status_code=400, detail='No rows left after applying filters')
    return out


def metric_block(plan: float, fact: float) -> Dict[str, Optional[float]]:
    variance = None if plan is None or fact is None else fact - plan
    variance_pct = pct_or_none(variance, plan) if variance is not None else None
    achievement_pct = pct_or_none(fact, plan)
    return {
        'plan': round_or_none(plan, 2),
        'fact': round_or_none(fact, 2),
        'variance_abs': round_or_none(variance, 2),
        'variance_pct': round_or_none(variance_pct, 4),
        'achievement_pct': round_or_none(achievement_pct, 4),
    }


def derive_group_metrics(g: pd.DataFrame) -> Dict[str, Any]:
    revenue_plan = g['revenue_plan'].sum()
    revenue_fact = g['revenue_fact'].sum()

    start_members_plan = g['members_start_plan'].sum()
    end_members_plan = g['members_end_plan'].sum()
    start_members_fact = g['members_start_fact'].sum()
    end_members_fact = g['members_end_fact'].sum()

    new_sales_plan = g['new_sales_plan'].sum()
    new_sales_fact = g['new_sales_fact'].sum()
    churn_plan = g['churn_plan'].sum()
    churn_fact = g['churn_fact'].sum()

    avg_arpu_plan = g['arpu_plan'].mean()
    avg_arpu_fact = g['arpu_fact'].mean()

    net_add_plan = new_sales_plan - churn_plan
    net_add_fact = new_sales_fact - churn_fact

    ending_members_gap = end_members_fact - end_members_plan
    start_members_gap = start_members_fact - start_members_plan

    revenue_per_member_plan = pct_or_none(revenue_plan, end_members_plan)
    revenue_per_member_fact = pct_or_none(revenue_fact, end_members_fact)

    return {
        'revenue': metric_block(revenue_plan, revenue_fact),
        'members_end': metric_block(end_members_plan, end_members_fact),
        'members_start': metric_block(start_members_plan, start_members_fact),
        'new_sales': metric_block(new_sales_plan, new_sales_fact),
        'churn': metric_block(churn_plan, churn_fact),
        'avg_arpu': metric_block(avg_arpu_plan, avg_arpu_fact),
        'net_add': metric_block(net_add_plan, net_add_fact),
        'revenue_per_member_end': metric_block(revenue_per_member_plan, revenue_per_member_fact),
        'diagnostics': {
            'member_bridge_plan': round_or_none(start_members_plan + new_sales_plan - churn_plan - end_members_plan, 2),
            'member_bridge_fact': round_or_none(start_members_fact + new_sales_fact - churn_fact - end_members_fact, 2),
            'ending_members_gap_abs': round_or_none(ending_members_gap, 2),
            'starting_members_gap_abs': round_or_none(start_members_gap, 2),
        }
    }


def period_summary(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for period, g in df.groupby('period', sort=True):
        metrics = derive_group_metrics(g)
        rows.append({
            'period': period,
            'revenue_plan': metrics['revenue']['plan'],
            'revenue_fact': metrics['revenue']['fact'],
            'revenue_variance_abs': metrics['revenue']['variance_abs'],
            'revenue_variance_pct': metrics['revenue']['variance_pct'],
            'members_end_plan': metrics['members_end']['plan'],
            'members_end_fact': metrics['members_end']['fact'],
            'members_end_variance_abs': metrics['members_end']['variance_abs'],
            'new_sales_plan': metrics['new_sales']['plan'],
            'new_sales_fact': metrics['new_sales']['fact'],
            'churn_plan': metrics['churn']['plan'],
            'churn_fact': metrics['churn']['fact'],
            'avg_arpu_plan': metrics['avg_arpu']['plan'],
            'avg_arpu_fact': metrics['avg_arpu']['fact'],
            'net_add_plan': metrics['net_add']['plan'],
            'net_add_fact': metrics['net_add']['fact'],
        })
    return rows


def club_summary(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for club, g in df.groupby('club', sort=True):
        metrics = derive_group_metrics(g)
        rows.append({
            'club': club,
            'revenue_plan': metrics['revenue']['plan'],
            'revenue_fact': metrics['revenue']['fact'],
            'revenue_variance_abs': metrics['revenue']['variance_abs'],
            'revenue_variance_pct': metrics['revenue']['variance_pct'],
            'revenue_achievement_pct': metrics['revenue']['achievement_pct'],
            'members_end_plan': metrics['members_end']['plan'],
            'members_end_fact': metrics['members_end']['fact'],
            'members_end_variance_abs': metrics['members_end']['variance_abs'],
            'new_sales_plan': metrics['new_sales']['plan'],
            'new_sales_fact': metrics['new_sales']['fact'],
            'new_sales_variance_abs': metrics['new_sales']['variance_abs'],
            'churn_plan': metrics['churn']['plan'],
            'churn_fact': metrics['churn']['fact'],
            'churn_variance_abs': metrics['churn']['variance_abs'],
            'avg_arpu_plan': metrics['avg_arpu']['plan'],
            'avg_arpu_fact': metrics['avg_arpu']['fact'],
            'avg_arpu_variance_abs': metrics['avg_arpu']['variance_abs'],
            'net_add_plan': metrics['net_add']['plan'],
            'net_add_fact': metrics['net_add']['fact'],
            'net_add_variance_abs': metrics['net_add']['variance_abs'],
        })
    rows = sorted(rows, key=lambda x: (x['revenue_variance_abs'] if x['revenue_variance_abs'] is not None else 0))
    return rows


def generate_alerts(clubs: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []

    for row in clubs:
        club = row['club']

        if row['revenue_variance_abs'] is not None and row['revenue_variance_abs'] < 0:
            alerts.append({
                'severity': abs(row['revenue_variance_abs']),
                'type': 'revenue_below_plan',
                'club': club,
                'message': f"{club}: revenue below plan by {row['revenue_variance_abs']:.2f}",
            })

        if row['new_sales_variance_abs'] is not None and row['new_sales_variance_abs'] < 0:
            alerts.append({
                'severity': abs(row['new_sales_variance_abs']),
                'type': 'new_sales_below_plan',
                'club': club,
                'message': f"{club}: new sales below plan by {row['new_sales_variance_abs']:.2f}",
            })

        if row['churn_variance_abs'] is not None and row['churn_variance_abs'] > 0:
            alerts.append({
                'severity': abs(row['churn_variance_abs']),
                'type': 'churn_above_plan',
                'club': club,
                'message': f"{club}: churn above plan by {row['churn_variance_abs']:.2f}",
            })

        if row['avg_arpu_variance_abs'] is not None and row['avg_arpu_variance_abs'] < 0:
            alerts.append({
                'severity': abs(row['avg_arpu_variance_abs']),
                'type': 'arpu_below_plan',
                'club': club,
                'message': f"{club}: ARPU below plan by {row['avg_arpu_variance_abs']:.2f}",
            })

        if row['members_end_variance_abs'] is not None and row['members_end_variance_abs'] < 0:
            alerts.append({
                'severity': abs(row['members_end_variance_abs']),
                'type': 'members_end_below_plan',
                'club': club,
                'message': f"{club}: ending members below plan by {row['members_end_variance_abs']:.2f}",
            })

    alerts = sorted(alerts, key=lambda x: x['severity'], reverse=True)
    return alerts[:top_n]


def build_kpi_dictionary(df: pd.DataFrame) -> Dict[str, Any]:
    overall = derive_group_metrics(df)
    clubs = club_summary(df)
    periods = period_summary(df)
    alerts = generate_alerts(clubs, 10)

    return {
        'overall': overall,
        'by_period': periods,
        'by_club': clubs,
        'alerts': alerts,
    }


def build_llm_payload(result: Dict[str, Any], filtered_df: pd.DataFrame) -> Dict[str, Any]:
    period_from = filtered_df['period'].min()
    period_to = filtered_df['period'].max()
    clubs = sorted(filtered_df['club'].unique().tolist())

    compact_clubs = [
        {
            'club': row['club'],
            'revenue_variance_abs': row['revenue_variance_abs'],
            'revenue_variance_pct': row['revenue_variance_pct'],
            'members_end_variance_abs': row['members_end_variance_abs'],
            'new_sales_variance_abs': row['new_sales_variance_abs'],
            'churn_variance_abs': row['churn_variance_abs'],
            'avg_arpu_variance_abs': row['avg_arpu_variance_abs'],
        }
        for row in result['by_club'][:10]
    ]

    return {
        'context': {
            'business_type': 'fitness_club_chain',
            'analysis_type': 'plan_fact',
            'period_from': period_from,
            'period_to': period_to,
            'club_count': len(clubs),
            'clubs': clubs,
        },
        'overall': result['overall'],
        'top_negative_clubs': compact_clubs,
        'alerts': result['alerts'],
        'by_period': result['by_period'],
        'instructions_for_llm': {
            'objective': 'Explain main drivers of plan-fact deviation and prioritize management actions.',
            'focus_areas': [
                'revenue', 'member_base', 'new_sales', 'churn', 'arpu'
            ],
            'required_output_sections': [
                'executive_summary',
                'main_positive_drivers',
                'main_negative_drivers',
                'clubs_to_watch',
                'recommended_actions'
            ]
        }
    }


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/analyze')
def analyze(req: AnalyzeRequest):
    df = load_and_validate(req.rows)
    df = apply_filters(df, req.period_from, req.period_to, req.clubs)

    result = build_kpi_dictionary(df)
    llm_payload = build_llm_payload(result, df)

    return {
        'status': 'ok',
        'input_summary': {
            'rows_received': len(req.rows),
            'rows_used': len(df),
            'period_from': df['period'].min(),
            'period_to': df['period'].max(),
            'clubs_used': sorted(df['club'].unique().tolist()),
        },
        **result,
        'llm_payload': llm_payload,
    }
