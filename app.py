"""
Startup Valuation Calculator
Application Streamlit pour calculer la valorisation d'une startup selon plusieurs méthodes
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import math
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Startup Valuation Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .method-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Données de référence pour les multiples sectoriels
SECTOR_MULTIPLES = {
    "Technologie": {"Revenue": 6.5, "EBITDA": 15.2},
    "SaaS": {"Revenue": 8.2, "EBITDA": 18.5},
    "E-commerce": {"Revenue": 3.1, "EBITDA": 12.8},
    "Fintech": {"Revenue": 7.8, "EBITDA": 16.9},
    "Biotech": {"Revenue": 12.4, "EBITDA": 25.6},
    "Cleantech": {"Revenue": 4.7, "EBITDA": 13.1},
    "Marketplace": {"Revenue": 5.3, "EBITDA": 14.7},
    "Media": {"Revenue": 2.8, "EBITDA": 9.4},
    "Manufacturing": {"Revenue": 1.9, "EBITDA": 8.2},
    "Retail": {"Revenue": 1.4, "EBITDA": 6.8}
}

class ValuationCalculator:
    """Classe principale pour les calculs de valorisation"""
    
    @staticmethod
    def dcf_valuation(cash_flows, growth_rate, discount_rate, terminal_growth=0.02):
        """
        Calcul DCF (Discounted Cash Flow)
        
        Args:
            cash_flows: Liste des flux de trésorerie prévisionnels
            growth_rate: Taux de croissance annuel
            discount_rate: Taux d'actualisation (WACC)
            terminal_growth: Taux de croissance terminal
        
        Returns:
            dict: Valorisation et détails du calcul
        """
        if not cash_flows or len(cash_flows) == 0:
            return {"valuation": 0, "error": "Flux de trésorerie requis"}
        
        # Calcul des flux actualisés
        discounted_flows = []
        cumulative_pv = 0
        
        for i, cf in enumerate(cash_flows):
            year = i + 1
            discounted_cf = cf / ((1 + discount_rate) ** year)
            discounted_flows.append(discounted_cf)
            cumulative_pv += discounted_cf
        
        # Valeur terminale
        if len(cash_flows) > 0:
            terminal_cf = cash_flows[-1] * (1 + terminal_growth)
            terminal_value = terminal_cf / (discount_rate - terminal_growth)
            terminal_pv = terminal_value / ((1 + discount_rate) ** len(cash_flows))
        else:
            terminal_pv = 0
        
        total_valuation = cumulative_pv + terminal_pv
        
        return {
            "valuation": total_valuation,
            "operating_value": cumulative_pv,
            "terminal_value": terminal_pv,
            "discounted_flows": discounted_flows,
            "terminal_pv": terminal_pv
        }
    
    @staticmethod
    def market_multiples_valuation(revenue_or_ebitda, multiple, metric_type="Revenue"):
        """
        Valorisation par multiples de marché
        
        Args:
            revenue_or_ebitda: Chiffre d'affaires ou EBITDA
            multiple: Multiple sectoriel
            metric_type: Type de métrique ("Revenue" ou "EBITDA")
        
        Returns:
            dict: Valorisation et détails
        """
        valuation = revenue_or_ebitda * multiple
        
        return {
            "valuation": valuation,
            "metric": revenue_or_ebitda,
            "multiple": multiple,
            "metric_type": metric_type
        }
    
    @staticmethod
    def scorecard_valuation(base_valuation, criteria_scores, criteria_weights=None):
        """
        Scorecard Method
        
        Args:
            base_valuation: Valorisation de base de référence
            criteria_scores: Dict des scores par critère (0-5)
            criteria_weights: Dict des pondérations par critère
        
        Returns:
            dict: Valorisation ajustée et détails
        """
        if criteria_weights is None:
            criteria_weights = {
                "team": 0.25,
                "product": 0.20,
                "market": 0.20,
                "competition": 0.15,
                "financial": 0.10,
                "legal": 0.10
            }
        
        # Score pondéré (3 = moyenne, facteur neutre)
        weighted_score = 0
        for criterion, score in criteria_scores.items():
            weight = criteria_weights.get(criterion, 0)
            # Conversion score (0-5) vers facteur multiplicateur (0.5-1.5)
            factor = 0.5 + (score / 5.0)
            weighted_score += weight * factor
        
        adjusted_valuation = base_valuation * weighted_score
        
        return {
            "valuation": adjusted_valuation,
            "base_valuation": base_valuation,
            "adjustment_factor": weighted_score,
            "criteria_analysis": {
                criterion: {
                    "score": score,
                    "weight": criteria_weights.get(criterion, 0),
                    "contribution": criteria_weights.get(criterion, 0) * (0.5 + score/5.0)
                }
                for criterion, score in criteria_scores.items()
            }
        }
    
    @staticmethod
    def berkus_valuation(criteria_scores):
        """
        Berkus Method - Méthode spécifique aux startups pré-revenus
        
        Args:
            criteria_scores: Dict des scores (0-5) pour les 5 critères Berkus
        
        Returns:
            dict: Valorisation et répartition par critère
        """
        max_value_per_criterion = 500000  # 500k€ max par critère
        
        criteria_mapping = {
            "concept": "Solidité du concept/idée",
            "prototype": "Prototype/MVP fonctionnel",
            "team": "Qualité de l'équipe dirigeante",
            "strategic_relationships": "Relations stratégiques",
            "product_rollout": "Lancement produit/premiers clients"
        }
        
        valuation_breakdown = {}
        total_valuation = 0
        
        for criterion, score in criteria_scores.items():
            criterion_value = (score / 5.0) * max_value_per_criterion
            valuation_breakdown[criterion] = {
                "name": criteria_mapping.get(criterion, criterion),
                "score": score,
                "value": criterion_value
            }
            total_valuation += criterion_value
        
        return {
            "valuation": total_valuation,
            "breakdown": valuation_breakdown,
            "max_possible": len(criteria_scores) * max_value_per_criterion
        }
    
    @staticmethod
    def risk_factor_summation(base_valuation, risk_factors):
        """
        Risk Factor Summation Method
        
        Args:
            base_valuation: Valorisation de base
            risk_factors: Dict des facteurs de risque (-2 à +2)
        
        Returns:
            dict: Valorisation ajustée par les risques
        """
        risk_categories = {
            "management": "Risque de gestion",
            "stage": "Risque lié au stade de développement",
            "legislation": "Risque législatif/politique",
            "manufacturing": "Risque de production",
            "sales": "Risque commercial/marketing",
            "funding": "Risque de financement",
            "competition": "Risque concurrentiel",
            "technology": "Risque technologique",
            "litigation": "Risque juridique",
            "international": "Risque international",
            "reputation": "Risque de réputation",
            "exit": "Risque de sortie/liquidité"
        }
        
        # Chaque facteur peut ajuster la valorisation de -25% à +25%
        total_adjustment = 0
        risk_analysis = {}
        
        for factor, rating in risk_factors.items():
            # Rating de -2 (très risqué) à +2 (très favorable)
            adjustment_pct = rating * 0.125  # Max ±25% total, ±12.5% par facteur
            total_adjustment += adjustment_pct
            
            risk_analysis[factor] = {
                "name": risk_categories.get(factor, factor),
                "rating": rating,
                "adjustment": adjustment_pct
            }
        
        # Limitation de l'ajustement total à ±50%
        total_adjustment = max(-0.5, min(0.5, total_adjustment))
        
        adjusted_valuation = base_valuation * (1 + total_adjustment)
        
        return {
            "valuation": adjusted_valuation,
            "base_valuation": base_valuation,
            "total_adjustment": total_adjustment,
            "risk_analysis": risk_analysis
        }
    
    @staticmethod
    def venture_capital_method(expected_revenue, exit_multiple, required_return, years_to_exit=5, investment_needed=None):
        """
        Venture Capital Method
        
        Args:
            expected_revenue: Revenus attendus à la sortie
            exit_multiple: Multiple de sortie (ex: 5x revenue)
            required_return: Retour sur investissement annuel requis
            years_to_exit: Années jusqu'à la sortie
            investment_needed: Montant d'investissement nécessaire
        
        Returns:
            dict: Valorisation pré-money et post-money
        """
        # Valeur à la sortie
        exit_value = expected_revenue * exit_multiple
        
        # Valeur actuelle (valeur terminale actualisée)
        present_value = exit_value / ((1 + required_return) ** years_to_exit)
        
        result = {
            "exit_value": exit_value,
            "present_value": present_value,
            "expected_return_multiple": (exit_value / present_value) if present_value > 0 else 0,
            "annualized_return": ((exit_value / present_value) ** (1/years_to_exit) - 1) if present_value > 0 else 0
        }
        
        if investment_needed:
            # Pourcentage de participation nécessaire
            ownership_needed = investment_needed / present_value if present_value > 0 else 0
            pre_money_valuation = present_value - investment_needed
            post_money_valuation = present_value
            
            result.update({
                "ownership_percentage": ownership_needed,
                "pre_money_valuation": pre_money_valuation,
                "post_money_valuation": post_money_valuation,
                "investment_needed": investment_needed
            })
        
        return result

def create_dcf_chart(result, cash_flows):
    """Créer un graphique pour la méthode DCF"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Flux de trésorerie par année', 'Valeurs actualisées',
                       'Répartition de la valeur', 'Analyse de sensibilité'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "heatmap"}]]
    )
    
    # Graphique 1: Flux de trésorerie
    years = [f"Année {i+1}" for i in range(len(cash_flows))]
    fig.add_trace(
        go.Bar(x=years, y=cash_flows, name="Flux de trésorerie", marker_color='lightblue'),
        row=1, col=1
    )
    
    # Graphique 2: Valeurs actualisées
    fig.add_trace(
        go.Bar(x=years, y=result['discounted_flows'], name="Valeurs actualisées", marker_color='darkblue'),
        row=1, col=2
    )
    
    # Graphique 3: Répartition de la valeur
    fig.add_trace(
        go.Pie(
            labels=['Valeur opérationnelle', 'Valeur terminale'],
            values=[result['operating_value'], result['terminal_pv']],
            hole=0.3
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Analyse DCF Complète")
    return fig

def create_comparison_chart(valuations_dict):
    """Créer un graphique de comparaison des méthodes"""
    methods = list(valuations_dict.keys())
    values = list(valuations_dict.values())
    
    fig = go.Figure()
    
    # Graphique en barres
    fig.add_trace(go.Bar(
        x=methods,
        y=values,
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
        text=[f"€{v:,.0f}" for v in values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Comparaison des Méthodes de Valorisation",
        xaxis_title="Méthodes",
        yaxis_title="Valorisation (€)",
        height=500,
        showlegend=False
    )
    
    return fig

def generate_pdf_report(valuations_dict, company_name="Ma Startup"):
    """Générer un rapport PDF avec les résultats"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Titre
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center
    )
    story.append(Paragraph(f"Rapport de Valorisation - {company_name}", title_style))
    story.append(Spacer(1, 20))
    
    # Date
    story.append(Paragraph(f"Date: {datetime.now().strftime('%d/%m/%Y')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Résumé exécutif
    story.append(Paragraph("Résumé Exécutif", styles['Heading2']))
    
    # Tableau des résultats
    data = [['Méthode de Valorisation', 'Valorisation (€)']]
    for method, value in valuations_dict.items():
        data.append([method, f"{value:,.0f} €"])
    
    # Statistiques
    values = list(valuations_dict.values())
    avg_valuation = np.mean(values)
    min_valuation = min(values)
    max_valuation = max(values)
    
    data.append(['', ''])
    data.append(['Valorisation Moyenne', f"{avg_valuation:,.0f} €"])
    data.append(['Valorisation Minimale', f"{min_valuation:,.0f} €"])
    data.append(['Valorisation Maximale', f"{max_valuation:,.0f} €"])
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 30))
    
    # Recommandations
    story.append(Paragraph("Recommandations", styles['Heading2']))
    story.append(Paragraph(
        "Cette évaluation fournit une fourchette de valorisation basée sur différentes méthodes reconnues. "
        "Il est recommandé de considérer l'ensemble des résultats plutôt qu'une seule méthode pour obtenir "
        "une vision complète de la valeur de votre startup.", 
        styles['Normal']
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def main():
    """Application principale"""
    
    # En-tête
    st.markdown('<h1 class="main-header">🚀 Startup Valuation Calculator</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Avertissement:</strong> Ces calculs sont fournis à titre indicatif uniquement. 
    La valorisation d'une startup dépend de nombreux facteurs qualitatifs et quantitatifs. 
    Consultez toujours des experts financiers pour des décisions d'investissement importantes.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Sélection des méthodes
    st.sidebar.header("🎯 Configuration")
    
    # Informations générales de la startup
    st.sidebar.subheader("Informations Générales")
    company_name = st.sidebar.text_input("Nom de la startup", value="Ma Startup")
    company_sector = st.sidebar.selectbox("Secteur d'activité", list(SECTOR_MULTIPLES.keys()))
    
    # Sélection des méthodes
    st.sidebar.subheader("Méthodes de Valorisation")
    methods = {
        "DCF": st.sidebar.checkbox("Discounted Cash Flow (DCF)", value=True),
        "Multiples": st.sidebar.checkbox("Multiples de marché", value=True),
        "Scorecard": st.sidebar.checkbox("Scorecard Method", value=True),
        "Berkus": st.sidebar.checkbox("Berkus Method", value=False),
        "Risk Factor": st.sidebar.checkbox("Risk Factor Summation", value=False),
        "VC Method": st.sidebar.checkbox("Venture Capital Method", value=False)
    }
    
    # Stockage des résultats
    valuations = {}
    detailed_results = {}
    
    # Interface principale avec tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Calculs", "📈 Comparaison", "📋 Rapport", "ℹ️ Aide"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # DCF Method
            if methods["DCF"]:
                st.markdown('<div class="method-card">', unsafe_allow_html=True)
                st.subheader("💰 Discounted Cash Flow (DCF)")
                
                dcf_col1, dcf_col2 = st.columns(2)
                
                with dcf_col1:
                    st.write("**Flux de trésorerie prévisionnels (€)**")
                    cf_years = st.number_input("Nombre d'années de projection", min_value=3, max_value=10, value=5, key="dcf_years")
                    cash_flows = []
                    for i in range(cf_years):
                        cf = st.number_input(f"Année {i+1}", min_value=0, value=100000*(i+1), key=f"cf_{i}")
                        cash_flows.append(cf)
                
                with dcf_col2:
                    discount_rate = st.slider("Taux d'actualisation (%)", 5.0, 25.0, 12.0, 0.5, key="discount_rate") / 100
                    terminal_growth = st.slider("Croissance terminale (%)", 0.0, 5.0, 2.0, 0.1, key="terminal_growth") / 100
                
                if st.button("Calculer DCF", key="calc_dcf"):
                    dcf_result = ValuationCalculator.dcf_valuation(cash_flows, 0.1, discount_rate, terminal_growth)
                    valuations["DCF"] = dcf_result["valuation"]
                    detailed_results["DCF"] = dcf_result
                    
                    # Affichage des résultats
                    st.success(f"**Valorisation DCF: {dcf_result['valuation']:,.0f} €**")
                    
                    # Graphique DCF
                    fig_dcf = create_dcf_chart(dcf_result, cash_flows)
                    st.plotly_chart(fig_dcf, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Market Multiples Method
            if methods["Multiples"]:
                st.markdown('<div class="method-card">', unsafe_allow_html=True)
                st.subheader("📊 Multiples de Marché")
                
                mult_col1, mult_col2 = st.columns(2)
                
                with mult_col1:
                    metric_type = st.selectbox("Métrique", ["Revenue", "EBITDA"], key="metric_type")
                    metric_value = st.number_input(f"{metric_type} annuel (€)", min_value=0, value=500000, key="metric_value")
                
                with mult_col2:
                    default_multiple = SECTOR_MULTIPLES[company_sector][metric_type]
                    multiple = st.number_input(f"Multiple {metric_type}", min_value=0.1, value=default_multiple, key="multiple")
                    st.info(f"Multiple moyen du secteur {company_sector}: {default_multiple}")
                
                if st.button("Calculer Multiples", key="calc_mult"):
                    mult_result = ValuationCalculator.market_multiples_valuation(metric_value, multiple, metric_type)
                    valuations["Multiples"] = mult_result["valuation"]
                    detailed_results["Multiples"] = mult_result
                    
                    st.success(f"**Valorisation par Multiples: {mult_result['valuation']:,.0f} €**")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Scorecard Method
            if methods["Scorecard"]:
                st.markdown('<div class="method-card">', unsafe_allow_html=True)
                st.