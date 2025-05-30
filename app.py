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
    .saved-result {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 0.75rem;
        margin: 0.5rem 0;
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
                       'Répartition de la valeur', 'Projection'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "scatter"}]]
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
    
    # Graphique 4: Projection
    years_extended = list(range(1, len(cash_flows) + 1))
    fig.add_trace(
        go.Scatter(x=years_extended, y=cash_flows, mode='lines+markers', 
                  name="Flux projetés", line=dict(color='green')),
        row=2, col=2
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

def save_result(method_name, result):
    """Fonction pour sauvegarder les résultats de manière fiable"""
    if 'valuations' not in st.session_state:
        st.session_state.valuations = {}
    if 'detailed_results' not in st.session_state:
        st.session_state.detailed_results = {}
    
    # Extraction de la valorisation selon le type de résultat
    if isinstance(result, dict):
        if method_name == "VC Method":
            valuation = result.get('pre_money_valuation', result.get('present_value', 0))
        else:
            valuation = result.get('valuation', 0)
    else:
        valuation = result
    
    st.session_state.valuations[method_name] = valuation
    st.session_state.detailed_results[method_name] = result
    
    return valuation

def delete_result(method_name):
    """Fonction pour supprimer un résultat sauvegardé"""
    if method_name in st.session_state.valuations:
        del st.session_state.valuations[method_name]
    if method_name in st.session_state.detailed_results:
        del st.session_state.detailed_results[method_name]

def show_saved_result(method_name, valuation):
    """Afficher un résultat sauvegardé avec style"""
    st.markdown(f"""
    <div class="saved-result">
        <strong>💾 {method_name} sauvegardé:</strong> {valuation:,.0f} €
    </div>
    """, unsafe_allow_html=True)

def main():
    """Application principale"""
    
    # Initialisation du session state de manière plus robuste
    if 'valuations' not in st.session_state:
        st.session_state.valuations = {}
    if 'detailed_results' not in st.session_state:
        st.session_state.detailed_results = {}
    if 'company_name' not in st.session_state:
        st.session_state.company_name = "Ma Startup"
    if 'company_sector' not in st.session_state:
        st.session_state.company_sector = "Technologie"
    if 'current_calculations' not in st.session_state:
        st.session_state.current_calculations = {}
    
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
    company_name = st.sidebar.text_input("Nom de la startup", value=st.session_state.company_name)
    company_sector = st.sidebar.selectbox("Secteur d'activité", list(SECTOR_MULTIPLES.keys()), 
                                        index=list(SECTOR_MULTIPLES.keys()).index(st.session_state.company_sector))
    
    # Mise à jour du session state
    st.session_state.company_name = company_name
    st.session_state.company_sector = company_sector
    
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
    
    # Bouton pour effacer tous les résultats
    st.sidebar.subheader("🗑️ Gestion des Résultats")
    if st.sidebar.button("Effacer tous les résultats", type="secondary"):
        st.session_state.valuations = {}
        st.session_state.detailed_results = {}
        st.session_state.current_calculations = {}
        st.rerun()
    
    # Affichage des résultats sauvegardés dans la sidebar
    if st.session_state.valuations:
        st.sidebar.subheader("📊 Résultats Sauvegardés")
        for method, value in st.session_state.valuations.items():
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.write(f"**{method}:** {value:,.0f} €")
            with col2:
                if st.button("🗑️", key=f"delete_{method}", help=f"Supprimer {method}"):
                    delete_result(method)
                    st.rerun()
    
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
                
                # Boutons d'action pour DCF
                dcf_button_col1, dcf_button_col2, dcf_button_col3 = st.columns([2, 1, 1])
                
                with dcf_button_col1:
                    if st.button("🧮 Calculer DCF", key="calc_dcf", type="primary"):
                        dcf_result = ValuationCalculator.dcf_valuation(cash_flows, 0.1, discount_rate, terminal_growth)
                        st.session_state.current_calculations["DCF"] = dcf_result
                        st.success(f"**Valorisation DCF: {dcf_result['valuation']:,.0f} €**")
                        
                        # Graphique DCF
                        fig_dcf = create_dcf_chart(dcf_result, cash_flows)
                        st.plotly_chart(fig_dcf, use_container_width=True)
                
                with dcf_button_col2:
                    if st.button("💾 Sauvegarder", key="save_dcf"):
                        if "DCF" in st.session_state.current_calculations:
                            result = st.session_state.current_calculations["DCF"]
                            save_result("DCF", result)
                            st.success("✅ DCF sauvegardé!")
                            st.rerun()
                        else:
                            st.error("Calculez d'abord la valorisation DCF")
                
                with dcf_button_col3:
                    if st.button("🗑️ Effacer", key="clear_dcf"):
                        if "DCF" in st.session_state.valuations:
                            delete_result("DCF")
                            st.success("🗑️ DCF effacé!")
                            st.rerun()
                
                # Affichage des résultats sauvegardés
                if "DCF" in st.session_state.valuations:
                    show_saved_result("DCF", st.session_state.valuations["DCF"])
                
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
                
                # Boutons d'action pour Multiples
                mult_button_col1, mult_button_col2, mult_button_col3 = st.columns([2, 1, 1])
                
                with mult_button_col1:
                    if st.button("🧮 Calculer Multiples", key="calc_mult", type="primary"):
                        mult_result = ValuationCalculator.market_multiples_valuation(metric_value, multiple, metric_type)
                        st.session_state.current_calculations["Multiples"] = mult_result
                        st.success(f"**Valorisation par Multiples: {mult_result['valuation']:,.0f} €**")
                
                with mult_button_col2:
                    if st.button("💾 Sauvegarder", key="save_mult"):
                        if "Multiples" in st.session_state.current_calculations:
                            result = st.session_state.current_calculations["Multiples"]
                            save_result("Multiples", result)
                            st.success("✅ Multiples sauvegardé!")
                            st.rerun()
                        else:
                            st.error("Calculez d'abord la valorisation par Multiples")
                
                with mult_button_col3:
                    if st.button("🗑️ Effacer", key="clear_mult"):
                        if "Multiples" in st.session_state.valuations:
                            delete_result("Multiples")
                            st.success("🗑️ Multiples effacé!")
                            st.rerun()
                
                # Affichage des résultats sauvegardés
                if "Multiples" in st.session_state.valuations:
                    show_saved_result("Multiples", st.session_state.valuations["Multiples"])
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Scorecard Method
            if methods["Scorecard"]:
                st.markdown('<div class="method-card">', unsafe_allow_html=True)
                st.subheader("📝 Scorecard Method")
                
                score_col1, score_col2 = st.columns(2)
                
                with score_col1:
                    base_valuation = st.number_input("Valorisation de base (€)", min_value=0, value=1000000, key="score_base")
                    st.write("**Évaluation des critères (0-5)**")
                    
                    criteria_scores = {}
                    criteria_scores["team"] = st.slider("👥 Équipe dirigeante", 0, 5, 3, key="score_team")
                    criteria_scores["product"] = st.slider("🚀 Produit/Service", 0, 5, 3, key="score_product")
                    criteria_scores["market"] = st.slider("🎯 Marché/Opportunité", 0, 5, 3, key="score_market")
                
                with score_col2:
                    st.write("**Pondérations (%)**")
                    weights = {}
                    weights["team"] = st.slider("👥 Équipe", 10, 40, 25, key="weight_team") / 100
                    weights["product"] = st.slider("🚀 Produit", 10, 30, 20, key="weight_product") / 100
                    weights["market"] = st.slider("🎯 Marché", 10, 30, 20, key="weight_market") / 100
                    weights["competition"] = st.slider("⚔️ Concurrence", 5, 25, 15, key="weight_competition") / 100
                    weights["financial"] = st.slider("💰 Finances", 5, 20, 10, key="weight_financial") / 100
                    weights["legal"] = st.slider("⚖️ Légal", 5, 15, 10, key="weight_legal") / 100
                    
                    criteria_scores["competition"] = st.slider("⚔️ Position concurrentielle", 0, 5, 3, key="score_competition")
                    criteria_scores["financial"] = st.slider("💰 Situation financière", 0, 5, 3, key="score_financial")
                    criteria_scores["legal"] = st.slider("⚖️ Aspects légaux", 0, 5, 3, key="score_legal")
                
                # Vérification que les poids totalisent 100%
                total_weight = sum(weights.values())
                if abs(total_weight - 1.0) > 0.01:
                    st.warning(f"⚠️ Les pondérations totalisent {total_weight*100:.1f}% au lieu de 100%")
                
                # Boutons d'action pour Scorecard
                score_button_col1, score_button_col2, score_button_col3 = st.columns([2, 1, 1])
                
                with score_button_col1:
                    if st.button("🧮 Calculer Scorecard", key="calc_scorecard", type="primary"):
                        scorecard_result = ValuationCalculator.scorecard_valuation(base_valuation, criteria_scores, weights)
                        st.session_state.current_calculations["Scorecard"] = scorecard_result
                        st.success(f"**Valorisation Scorecard: {scorecard_result['valuation']:,.0f} €**")
                        st.info(f"Facteur d'ajustement: {scorecard_result['adjustment_factor']:.2f}")
                        
                        # Graphique des contributions
                        criteria_names = []
                        contributions = []
                        for criterion, analysis in scorecard_result['criteria_analysis'].items():
                            criteria_names.append(criterion.title())
                            contributions.append(analysis['contribution'])
                        
                        fig_scorecard = go.Figure(data=[
                            go.Bar(x=criteria_names, y=contributions, marker_color='lightgreen')
                        ])
                        fig_scorecard.update_layout(
                            title="Contribution de chaque critère",
                            xaxis_title="Critères",
                            yaxis_title="Contribution à l'ajustement"
                        )
                        st.plotly_chart(fig_scorecard, use_container_width=True)
                
                with score_button_col2:
                    if st.button("💾 Sauvegarder", key="save_scorecard"):
                        if "Scorecard" in st.session_state.current_calculations:
                            result = st.session_state.current_calculations["Scorecard"]
                            save_result("Scorecard", result)
                            st.success("✅ Scorecard sauvegardé!")
                            st.rerun()
                        else:
                            st.error("Calculez d'abord la valorisation Scorecard")
                
                with score_button_col3:
                    if st.button("🗑️ Effacer", key="clear_scorecard"):
                        if "Scorecard" in st.session_state.valuations:
                            delete_result("Scorecard")
                            st.success("🗑️ Scorecard effacé!")
                            st.rerun()
                
                # Affichage des résultats sauvegardés
                if "Scorecard" in st.session_state.valuations:
                    show_saved_result("Scorecard", st.session_state.valuations["Scorecard"])
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Berkus Method
            if methods["Berkus"]:
                st.markdown('<div class="method-card">', unsafe_allow_html=True)
                st.subheader("🎯 Berkus Method")
                st.info("Méthode spécialement conçue pour les startups pré-revenus. Maximum 500k€ par critère.")
                
                berkus_col1, berkus_col2 = st.columns(2)
                
                with berkus_col1:
                    berkus_scores = {}
                    berkus_scores["concept"] = st.slider("💡 Solidité du concept", 0, 5, 3, key="berkus_concept")
                    berkus_scores["prototype"] = st.slider("🔧 Prototype/MVP", 0, 5, 3, key="berkus_prototype")
                    berkus_scores["team"] = st.slider("👨‍💼 Équipe dirigeante", 0, 5, 3, key="berkus_team")
                
                with berkus_col2:
                    berkus_scores["strategic_relationships"] = st.slider("🤝 Relations stratégiques", 0, 5, 3, key="berkus_relations")
                    berkus_scores["product_rollout"] = st.slider("📊 Lancement produit", 0, 5, 3, key="berkus_rollout")
                
                # Boutons d'action pour Berkus
                berkus_button_col1, berkus_button_col2, berkus_button_col3 = st.columns([2, 1, 1])
                
                with berkus_button_col1:
                    if st.button("🧮 Calculer Berkus", key="calc_berkus", type="primary"):
                        berkus_result = ValuationCalculator.berkus_valuation(berkus_scores)
                        st.session_state.current_calculations["Berkus"] = berkus_result
                        st.success(f"**Valorisation Berkus: {berkus_result['valuation']:,.0f} €**")
                        st.info(f"Potentiel maximum: {berkus_result['max_possible']:,.0f} €")
                        
                        # Graphique en barres pour Berkus
                        criteria_names = [data['name'] for data in berkus_result['breakdown'].values()]
                        criteria_values = [data['value'] for data in berkus_result['breakdown'].values()]
                        
                        fig_berkus = go.Figure(data=[
                            go.Bar(x=criteria_names, y=criteria_values, marker_color='orange')
                        ])
                        fig_berkus.update_layout(
                            title="Répartition de la valorisation Berkus",
                            xaxis_title="Critères",
                            yaxis_title="Valeur (€)"
                        )
                        st.plotly_chart(fig_berkus, use_container_width=True)
                
                with berkus_button_col2:
                    if st.button("💾 Sauvegarder", key="save_berkus"):
                        if "Berkus" in st.session_state.current_calculations:
                            result = st.session_state.current_calculations["Berkus"]
                            save_result("Berkus", result)
                            st.success("✅ Berkus sauvegardé!")
                            st.rerun()
                        else:
                            st.error("Calculez d'abord la valorisation Berkus")
                
                with berkus_button_col3:
                    if st.button("🗑️ Effacer", key="clear_berkus"):
                        if "Berkus" in st.session_state.valuations:
                            delete_result("Berkus")
                            st.success("🗑️ Berkus effacé!")
                            st.rerun()
                
                # Affichage des résultats sauvegardés
                if "Berkus" in st.session_state.valuations:
                    show_saved_result("Berkus", st.session_state.valuations["Berkus"])
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Risk Factor Summation Method
            if methods["Risk Factor"]:
                st.markdown('<div class="method-card">', unsafe_allow_html=True)
                st.subheader("⚠️ Risk Factor Summation")
                
                risk_base_val = st.number_input("Valorisation de base (€)", min_value=0, value=1000000, key="risk_base")
                
                st.write("**Évaluation des facteurs de risque (-2: Très risqué, 0: Neutre, +2: Très favorable)**")
                
                risk_col1, risk_col2 = st.columns(2)
                
                with risk_col1:
                    risk_factors = {}
                    risk_factors["management"] = st.slider("👨‍💼 Risque de gestion", -2, 2, 0, key="risk_mgmt")
                    risk_factors["stage"] = st.slider("🚀 Stade de développement", -2, 2, 0, key="risk_stage")
                    risk_factors["legislation"] = st.slider("⚖️ Risque législatif", -2, 2, 0, key="risk_legal")
                    risk_factors["manufacturing"] = st.slider("🏭 Risque de production", -2, 2, 0, key="risk_manuf")
                    risk_factors["sales"] = st.slider("💼 Risque commercial", -2, 2, 0, key="risk_sales")
                    risk_factors["funding"] = st.slider("💰 Risque de financement", -2, 2, 0, key="risk_funding")
                
                with risk_col2:
                    risk_factors["competition"] = st.slider("⚔️ Risque concurrentiel", -2, 2, 0, key="risk_comp")
                    risk_factors["technology"] = st.slider("💻 Risque technologique", -2, 2, 0, key="risk_tech")
                    risk_factors["litigation"] = st.slider("⚖️ Risque juridique", -2, 2, 0, key="risk_litigation")
                    risk_factors["international"] = st.slider("🌍 Risque international", -2, 2, 0, key="risk_intl")
                    risk_factors["reputation"] = st.slider("🏆 Risque de réputation", -2, 2, 0, key="risk_rep")
                    risk_factors["exit"] = st.slider("🚪 Risque de sortie", -2, 2, 0, key="risk_exit")
                
                # Boutons d'action pour Risk Factor
                risk_button_col1, risk_button_col2, risk_button_col3 = st.columns([2, 1, 1])
                
                with risk_button_col1:
                    if st.button("🧮 Calculer Risk Factor", key="calc_risk", type="primary"):
                        risk_result = ValuationCalculator.risk_factor_summation(risk_base_val, risk_factors)
                        st.session_state.current_calculations["Risk Factor"] = risk_result
                        st.success(f"**Valorisation ajustée: {risk_result['valuation']:,.0f} €**")
                        st.info(f"Ajustement total: {risk_result['total_adjustment']*100:+.1f}%")
                        
                        # Graphique des ajustements de risque
                        risk_names = [analysis['name'] for analysis in risk_result['risk_analysis'].values()]
                        risk_adjustments = [analysis['adjustment']*100 for analysis in risk_result['risk_analysis'].values()]
                        
                        colors_risk = ['red' if x < 0 else 'green' if x > 0 else 'gray' for x in risk_adjustments]
                        
                        fig_risk = go.Figure(data=[
                            go.Bar(x=risk_names, y=risk_adjustments, marker_color=colors_risk)
                        ])
                        fig_risk.update_layout(
                            title="Impact des facteurs de risque (%)",
                            xaxis_title="Facteurs de risque",
                            yaxis_title="Ajustement (%)"
                        )
                        st.plotly_chart(fig_risk, use_container_width=True)
                
                with risk_button_col2:
                    if st.button("💾 Sauvegarder", key="save_risk"):
                        if "Risk Factor" in st.session_state.current_calculations:
                            result = st.session_state.current_calculations["Risk Factor"]
                            save_result("Risk Factor", result)
                            st.success("✅ Risk Factor sauvegardé!")
                            st.rerun()
                        else:
                            st.error("Calculez d'abord la valorisation Risk Factor")
                
                with risk_button_col3:
                    if st.button("🗑️ Effacer", key="clear_risk"):
                        if "Risk Factor" in st.session_state.valuations:
                            delete_result("Risk Factor")
                            st.success("🗑️ Risk Factor effacé!")
                            st.rerun()
                
                # Affichage des résultats sauvegardés
                if "Risk Factor" in st.session_state.valuations:
                    show_saved_result("Risk Factor", st.session_state.valuations["Risk Factor"])
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Venture Capital Method
            if methods["VC Method"]:
                st.markdown('<div class="method-card">', unsafe_allow_html=True)
                st.subheader("🏦 Venture Capital Method")
                
                vc_col1, vc_col2 = st.columns(2)
                
                with vc_col1:
                    expected_revenue = st.number_input("Revenus attendus à la sortie (€)", min_value=0, value=10000000, key="vc_revenue")
                    exit_multiple = st.number_input("Multiple de sortie", min_value=0.1, value=5.0, key="vc_multiple")
                    required_return = st.slider("Retour annuel requis (%)", 15.0, 50.0, 25.0, key="vc_return") / 100
                
                with vc_col2:
                    years_to_exit = st.number_input("Années jusqu'à la sortie", min_value=1, max_value=10, value=5, key="vc_years")
                    investment_needed = st.number_input("Investissement nécessaire (€)", min_value=0, value=2000000, key="vc_investment")
                
                # Boutons d'action pour VC Method
                vc_button_col1, vc_button_col2, vc_button_col3 = st.columns([2, 1, 1])
                
                with vc_button_col1:
                    if st.button("🧮 Calculer VC Method", key="calc_vc", type="primary"):
                        vc_result = ValuationCalculator.venture_capital_method(
                            expected_revenue, exit_multiple, required_return, years_to_exit, investment_needed
                        )
                        st.session_state.current_calculations["VC Method"] = vc_result
                        st.success(f"**Valorisation pré-money: {vc_result.get('pre_money_valuation', vc_result['present_value']):,.0f} €**")
                        
                        # Métriques VC
                        vc_metrics_col1, vc_metrics_col2 = st.columns(2)
                        with vc_metrics_col1:
                            st.metric("Valeur à la sortie", f"{vc_result['exit_value']:,.0f} €")
                            st.metric("Multiple de retour", f"{vc_result['expected_return_multiple']:.1f}x")
                        
                        with vc_metrics_col2:
                            if 'ownership_percentage' in vc_result:
                                st.metric("Part nécessaire", f"{vc_result['ownership_percentage']*100:.1f}%")
                            st.metric("Retour annualisé", f"{vc_result['annualized_return']*100:.1f}%")
                
                with vc_button_col2:
                    if st.button("💾 Sauvegarder", key="save_vc"):
                        if "VC Method" in st.session_state.current_calculations:
                            result = st.session_state.current_calculations["VC Method"]
                            save_result("VC Method", result)
                            st.success("✅ VC Method sauvegardé!")
                            st.rerun()
                        else:
                            st.error("Calculez d'abord la valorisation VC Method")
                
                with vc_button_col3:
                    if st.button("🗑️ Effacer", key="clear_vc"):
                        if "VC Method" in st.session_state.valuations:
                            delete_result("VC Method")
                            st.success("🗑️ VC Method effacé!")
                            st.rerun()
                
                # Affichage des résultats sauvegardés
                if "VC Method" in st.session_state.valuations:
                    show_saved_result("VC Method", st.session_state.valuations["VC Method"])
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Sidebar des résultats
        with col2:
            if st.session_state.valuations:
                st.subheader("📈 Résultats Actuels")
                
                for method, value in st.session_state.valuations.items():
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{method}</h4>
                        <h3>{value:,.0f} €</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Statistiques résumées
                if len(st.session_state.valuations) > 1:
                    values = list(st.session_state.valuations.values())
                    st.markdown("### 📊 Statistiques")
                    st.metric("Moyenne", f"{np.mean(values):,.0f} €")
                    st.metric("Médiane", f"{np.median(values):,.0f} €")
                    st.metric("Écart-type", f"{np.std(values):,.0f} €")
                    st.metric("Min - Max", f"{min(values):,.0f} € - {max(values):,.0f} €")
            else:
                st.info("Calculez et sauvegardez des résultats pour voir l'analyse comparative")
    
    with tab2:
        st.header("📈 Analyse Comparative")
        
        if len(st.session_state.valuations) >= 2:
            # Graphique de comparaison
            fig_comparison = create_comparison_chart(st.session_state.valuations)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Analyse statistique
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Statistiques Descriptives")
                values = list(st.session_state.valuations.values())
                stats_df = pd.DataFrame({
                    'Métrique': ['Moyenne', 'Médiane', 'Écart-type', 'Minimum', 'Maximum', 'Coefficient de variation'],
                    'Valeur': [
                        f"{np.mean(values):,.0f} €",
                        f"{np.median(values):,.0f} €",
                        f"{np.std(values):,.0f} €",
                        f"{min(values):,.0f} €",
                        f"{max(values):,.0f} €",
                        f"{np.std(values)/np.mean(values)*100:.1f}%"
                    ]
                })
                st.dataframe(stats_df, hide_index=True)
            
            with col2:
                st.subheader("🎯 Recommandations")
                cv = np.std(values) / np.mean(values)
                
                if cv < 0.3:
                    st.success("✅ **Convergence forte** - Les méthodes donnent des résultats cohérents")
                elif cv < 0.6:
                    st.warning("⚠️ **Convergence modérée** - Variabilité acceptable entre les méthodes")
                else:
                    st.error("❌ **Forte divergence** - Revoir les hypothèses ou se concentrer sur les méthodes les plus pertinentes")
                
                # Fourchette de valorisation recommandée
                percentile_25 = np.percentile(values, 25)
                percentile_75 = np.percentile(values, 75)
                st.info(f"**Fourchette recommandée:** {percentile_25:,.0f} € - {percentile_75:,.0f} €")
        else:
            st.info("💡 Calculez et sauvegardez au moins 2 méthodes de valorisation pour voir l'analyse comparative.")
            if st.session_state.valuations:
                st.write("**Résultats disponibles:**")
                for method, value in st.session_state.valuations.items():
                    st.write(f"- {method}: {value:,.0f} €")
    
    with tab3:
        st.header("📋 Rapport de Valorisation")
        
        if st.session_state.valuations:
            # Génération du rapport PDF
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("Résumé du Rapport")
                
                # Tableau des résultats
                results_df = pd.DataFrame([
                    {'Méthode': method, 'Valorisation (€)': f"{value:,.0f}", 'Valorisation': value}
                    for method, value in st.session_state.valuations.items()
                ])
                
                st.dataframe(results_df[['Méthode', 'Valorisation (€)']], hide_index=True)
                
                # Synthèse
                values = list(st.session_state.valuations.values())
                st.markdown(f"""
                ### 🎯 Synthèse Exécutive
                
                **Entreprise:** {st.session_state.company_name}  
                **Secteur:** {st.session_state.company_sector}  
                **Date d'évaluation:** {datetime.now().strftime('%d/%m/%Y')}
                
                **Fourchette de valorisation:** {min(values):,.0f} € - {max(values):,.0f} €  
                **Valorisation médiane:** {np.median(values):,.0f} €  
                **Nombre de méthodes utilisées:** {len(st.session_state.valuations)}
                """)
            
            with col2:
                # Bouton de téléchargement PDF
                if st.button("📥 Générer Rapport PDF", type="primary"):
                    try:
                        pdf_buffer = generate_pdf_report(st.session_state.valuations, st.session_state.company_name)
                        
                        st.download_button(
                            label="⬇️ Télécharger PDF",
                            data=pdf_buffer,
                            file_name=f"rapport_valorisation_{st.session_state.company_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
                        st.success("✅ Rapport PDF généré avec succès!")
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la génération du PDF: {str(e)}")
        else:
            st.info("📊 Aucune valorisation calculée. Retournez à l'onglet 'Calculs' pour commencer.")
            st.markdown("""
            ### Comment procéder:
            1. Retournez à l'onglet **📊 Calculs**
            2. Sélectionnez une ou plusieurs méthodes de valorisation
            3. Remplissez les paramètres requis
            4. Cliquez sur **🧮 Calculer** puis **💾 Sauvegarder**
            5. Répétez pour d'autres méthodes
            6. Revenez ici pour générer votre rapport
            """)
    
    with tab4:
        st.header("ℹ️ Guide d'Utilisation")
        
        st.markdown("""
        ## 🎯 Comment utiliser ce calculateur
        
        ### 1. Configuration initiale
        - Renseignez le nom de votre startup et son secteur d'activité
        - Sélectionnez les méthodes de valorisation pertinentes pour votre situation
        
        ### 2. Choix des méthodes
        
        **Pour les startups avec revenus :**
        - ✅ DCF (si flux de trésorerie prévisibles)
        - ✅ Multiples de marché
        - ✅ Scorecard Method
        
        **Pour les startups pré-revenus :**
        - ✅ Berkus Method
        - ✅ Scorecard Method
        - ✅ Risk Factor Summation
        
        **Pour les levées de fonds :**
        - ✅ Venture Capital Method
        - ✅ DCF
        
        ### 3. Flux de travail recommandé
        
        **Étape 1:** Configuration
        - Configurez le nom et secteur dans la sidebar
        - Cochez les méthodes pertinentes
        
        **Étape 2:** Calculs
        - Pour chaque méthode sélectionnée:
          1. Remplissez les paramètres requis
          2. Cliquez sur **🧮 Calculer**
          3. Vérifiez les résultats et graphiques
          4. Cliquez sur **💾 Sauvegarder** pour conserver le résultat
        
        **Étape 3:** Analyse
        - Consultez l'onglet **📈 Comparaison** pour analyser les écarts
        - Générez le rapport PDF dans l'onglet **📋 Rapport**
        
        ### 4. Gestion des résultats
        
        **Boutons de contrôle pour chaque méthode :**
        - 🧮 **Calculer** : Lance le calcul avec les paramètres actuels
        - 💾 **Sauvegarder** : Enregistre le résultat pour la comparaison (requis!)
        - 🗑️ **Effacer** : Supprime le résultat sauvegardé de cette méthode
        
        **Contrôle global dans la sidebar :**
        - 🗑️ **Effacer tous les résultats** : Reset complet
        - 🗑️ Boutons individuels pour supprimer chaque méthode sauvegardée
        
        ### 5. Interprétation des résultats
        
        #### 🟢 Convergence forte (CV < 30%)
        Les méthodes donnent des résultats similaires → Valorisation fiable
        
        #### 🟡 Convergence modérée (CV 30-60%)
        Variabilité acceptable → Utiliser une fourchette
        
        #### 🔴 Forte divergence (CV > 60%)
        Revoir les hypothèses ou se concentrer sur les méthodes les plus adaptées
        
        ### 6. Limites et précautions
        
        ⚠️ **Important :** Ces calculs sont indicatifs uniquement
        - La valorisation dépend de nombreux facteurs qualitatifs
        - Le contexte de marché influence fortement les résultats
        - Consultez des experts pour des décisions importantes
        
        ### 7. Résolution des problèmes courants
        
        **Problème:** Le bouton "Sauvegarder" ne fonctionne pas
        - **Solution:** Assurez-vous d'avoir d'abord cliqué sur "🧮 Calculer"
        
        **Problème:** Pas de graphique de comparaison
        - **Solution:** Sauvegardez au moins 2 méthodes de valorisation
        
        **Problème:** Impossible de générer le PDF
        - **Solution:** Vérifiez qu'au moins une méthode est sauvegardée
        
        **Problème:** Les pondérations ne totalisent pas 100%
        - **Solution:** Ajustez les sliders pour que la somme soit exactement 100%
        
        ### 8. Conseils d'utilisation avancée
        
        **Analyse de sensibilité:**
        - Testez différents scénarios en modifiant les paramètres
        - Comparez les résultats pour identifier les variables critiques
        
        **Choix des méthodes selon le contexte:**
        - **Startup tech avec traction:** DCF + Multiples + VC Method
        - **Startup pré-revenus:** Berkus + Scorecard + Risk Factor
        - **Levée de fonds série A:** DCF + VC Method + Multiples
        
        ### 9. Sources et références
        
        - **DCF :** Damodaran, Aswath. "Investment Valuation"
        - **Multiples :** PwC Money Tree Reports, CB Insights
        - **Berkus Method :** Dave Berkus, "Basic Angel Investing"
        - **Scorecard :** Bill Payne, Angel Capital Association
        - **Risk Factor :** Various VC and Angel methodologies
        - **VC Method :** Venture Capital industry standard practices
        
        ### 10. Support et améliorations
        
        Cette application est en développement continu. Pour signaler des bugs ou suggérer des améliorations:
        - Vérifiez d'abord ce guide de dépannage
        - Documentez précisément les étapes qui posent problème
        - Proposez des améliorations sur le repository GitHub
        """)

if __name__ == "__main__":
    main()
