"""
UI Styling Module for AI-Powered Aviation Knowledge Assistant
Provides modern, clean, professional aviation-themed styling, custom CSS,
and HTML component generators.
"""

from typing import Any

AVIATION_CUSTOM_CSS = """
<style>
/* Modern Aviation Typography & Layout */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --aviation-navy: #0B192C;
    --aviation-deep-blue: #1E3E62;
    --aviation-sky-blue: #0077C2;
    --aviation-cyan: #00A8E8;
    --aviation-light-bg: #F4F7FB;
    --aviation-card-bg: #FFFFFF;
    --aviation-text-main: #1E293B;
    --aviation-text-muted: #64748B;
    --aviation-border: #E2E8F0;
    --aviation-success: #059669;
    --aviation-warning: #D97706;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* App Header Styling */
.aviation-header-container {
    background: linear-gradient(135deg, #0B192C 0%, #1E3E62 60%, #0077C2 100%);
    color: #FFFFFF;
    padding: 1.8rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 25px -5px rgba(11, 25, 44, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
}

.aviation-header-container::after {
    content: "✈";
    position: absolute;
    right: 25px;
    bottom: -15px;
    font-size: 6rem;
    color: rgba(255, 255, 255, 0.06);
    pointer-events: none;
    transform: rotate(-15deg);
}

.aviation-header-title {
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
    color: #FFFFFF !important;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.aviation-header-subtitle {
    font-size: 1.05rem;
    font-weight: 400;
    color: #E2E8F0 !important;
    margin-top: 0.4rem;
    margin-bottom: 0;
    max-width: 800px;
}

/* Safety Disclaimer Banner */
.safety-disclaimer-banner {
    background-color: #FEF3C7;
    border-left: 4px solid #D97706;
    padding: 0.85rem 1.1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.88rem;
    color: #92400E;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    line-height: 1.45;
}

/* Modern Metric Cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.aviation-metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    border-top: 3px solid #0077C2;
}

.aviation-metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0, 119, 194, 0.1);
}

.metric-label {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
    margin-bottom: 0.35rem;
}

.metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #0B192C;
    line-height: 1.1;
}

.metric-subtext {
    font-size: 0.78rem;
    color: #94A3B8;
    margin-top: 0.35rem;
}

/* Chat Message Styling */
.source-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    color: #1D4ED8;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 500;
    margin-right: 0.4rem;
    margin-top: 0.3rem;
}

.source-tag:hover {
    background: #DBEAFE;
}

.context-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 0.9rem;
    margin-top: 0.6rem;
    font-size: 0.88rem;
    line-height: 1.5;
}

.context-meta {
    font-size: 0.8rem;
    font-weight: 600;
    color: #0077C2;
    margin-bottom: 0.4rem;
    display: flex;
    justify-content: space-between;
}

/* Status Badges */
.badge-indexed {
    background-color: #ECFDF5;
    color: #065F46;
    border: 1px solid #A7F3D0;
    padding: 0.2rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
}

.badge-warning {
    background-color: #FFFBEB;
    color: #92400E;
    border: 1px solid #FDE68A;
    padding: 0.2rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
}

.badge-error {
    background-color: #FEF2F2;
    color: #991B1B;
    border: 1px solid #FECACA;
    padding: 0.2rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* Sidebar Customization */
section[data-testid="stSidebar"] {
    background-color: #0B192C !important;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
section[data-testid="stSidebar"] label {
    color: #E2E8F0 !important;
}

section[data-testid="stSidebar"] .stRadio > div {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 8px;
    padding: 0.5rem;
}

/* Quick Question Chip */
.quick-chip-button {
    text-align: left;
    margin-bottom: 0.4rem;
}
</style>
"""


def render_header(title: str, subtitle: str) -> str:
    """Generates the aviation-themed hero header HTML."""
    return f"""
    <div class="aviation-header-container">
        <h1 class="aviation-header-title">✈️ {title}</h1>
        <p class="aviation-header-subtitle">{subtitle}</p>
    </div>
    """


def render_disclaimer(disclaimer_text: str) -> str:
    """Generates the aviation compliance safety banner HTML."""
    return f"""
    <div class="safety-disclaimer-banner">
        <span style="font-size: 1.1rem; line-height: 1;">🛡️</span>
        <div>
            <strong>Aviation Safety Notice:</strong> {disclaimer_text}
        </div>
    </div>
    """


def render_metric_card(label: str, value: Any, subtext: str = "") -> str:
    """Generates a styled metric card HTML."""
    return f"""
    <div class="aviation-metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-subtext">{subtext}</div>' if subtext else ''}
    </div>
    """
