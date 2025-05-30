"""
Startup Valuation Calculator - Version Simplifiée
Application Streamlit pour calculer la valorisation d'une startup selon plusieurs méthodes
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
        """Calcul DCF (Discounted Cash Flow)"""
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
        """Valorisation par multiples de marché"""
        valuation = revenue_or_ebitda * multiple
        
        return {
            "valuation": valuation,
            "metric": revenue_or_ebitda,
            "multiple": multiple,
            "metric_type": metric_type
        }
    
    @staticmethod
    def scorecard_valuation(base_valuation, criteria_scores, criteria_weights=None):
        """Scorecard Method"""
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

def main():
    """Application principale"""
    
    # Initialisation du session state
    if 'valuations' not in st.session_state:
        st.session_state.valuations = {}
    if 'company_name' not in st.session_state:
        st.session_state.company_name = "Ma Startup"
    if 'company_sector' not in st.session_state:
        st.session_state.company_sector = "Technologie"
    
    # En-tête
    st.markdown('<h1 class="main-header">🚀 Startup Valuation Calculator</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Avertissement:</strong> Ces calculs sont fournis à titre indicatif uniquement. 
    La valorisation d'une startup dépend de nombreux facteurs qualitatifs et quantitatifs. 
    Consultez toujours des experts financiers pour des décisions d'investissement importantes.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Configuration
    st.sidebar.header("🎯 Configuration")
    
    # Informations générales de la startup
    st.sidebar.subheader("Informations Générales")
    company_name = st.sidebar.text_input("Nom de la startup", value=st.session_state.company_name)
    company_sector = st.sidebar.selectbox("Secteur d'activité", list(SECTOR_MULTIPLES.keys()), 
                                        index=list(SECTOR_MULTIPLES.keys()).index(st.session_state.company_sector))
    
    # Mise à jour du session state
    st.session_state.company_name = company_name
    st.session_state.company_sector = company_sector
    
    # Bouton pour effacer tous les résultats
    if st.sidebar.button("🗑️ Effacer tous les résultats", type="secondary"):
        st.session_state.valuations = {}
        st.rerun()
    
    # Affichage des résultats sauvegardés dans la sidebar
    if st.session_state.valuations:
        st.sidebar.subheader("📊 Résultats Sauvegardés")
        for method, value in st.session_state.valuations.items():
            st.sidebar.write(f"**{method}:** {value:,.0f} €")
    
    # Interface principale avec tabs
    tab1, tab2, tab3 = st.tabs(["📊 Calculs", "📈 Comparaison", "ℹ️ Aide"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # DCF Method
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
                st.session_state.valuations["DCF"] = dcf_result["valuation"]
                st.success(f"**Valorisation DCF: {dcf_result['valuation']:,.0f} €**")
            
            # Affichage des résultats existants
            if "DCF" in st.session_state.valuations:
                st.info(f"💾 **Résultat sauvegardé DCF:** {st.session_state.valuations['DCF']:,.0f} €")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Market Multiples Method
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
                st.session_state.valuations["Multiples"] = mult_result["valuation"]
                st.success(f"**Valorisation par Multiples: {mult_result['valuation']:,.0f} €**")
            
            # Affichage des résultats existants
            if "Multiples" in st.session_state.valuations:
                st.info(f"💾 **Résultat sauvegardé Multiples:** {st.session_state.valuations['Multiples']:,.0f} €")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Scorecard Method
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
            
            if st.button("Calculer Scorecard", key="calc_scorecard"):
                scorecard_result = ValuationCalculator.scorecard_valuation(base_valuation, criteria_scores, weights)
                st.session_state.valuations["Scorecard"] = scorecard_result["valuation"]
                st.success(f"**Valorisation Scorecard: {scorecard_result['valuation']:,.0f} €**")
                st.info(f"Facteur d'ajustement: {scorecard_result['adjustment_factor']:.2f}")
            
            # Affichage des résultats existants
            if "Scorecard" in st.session_state.valuations:
                st.info(f"💾 **Résultat sauvegardé Scorecard:** {st.session_state.valuations['Scorecard']:,.0f} €")
            
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
            st.info("Calculez au moins 2 méthodes de valorisation pour voir l'analyse comparative.")
    
    with tab3:
        st.header("ℹ️ Guide d'Utilisation")
        
        st.markdown("""
        ## 🎯 Comment utiliser ce calculateur
        
        ### 1. Configuration initiale
        - Renseignez le nom de votre startup et son secteur d'activité
        - Les résultats sont automatiquement sauvegardés
        
        ### 2. Méthodes disponibles
        
        **DCF (Discounted Cash Flow) :**
        - Basé sur les flux de trésorerie futurs actualisés
        - Idéal pour les startups avec revenus prévisibles
        
        **Multiples de marché :**
        - Compare avec des entreprises similaires du secteur
        - Utilise des ratios Revenue ou EBITDA
        
        **Scorecard Method :**
        - Évalue selon plusieurs critères pondérés
        - Adapté à tous types de startups
        
        ### 3. Interprétation des résultats
        
        #### 🟢 Convergence forte (CV < 30%)
        Les méthodes donnent des résultats similaires → Valorisation fiable
        
        #### 🟡 Convergence modérée (CV 30-60%)
        Variabilité acceptable → Utiliser une fourchette
        
        #### 🔴 Forte divergence (CV > 60%)
        Revoir les hypothèses ou se concentrer sur les méthodes les plus adaptées
        
        ### 4. Limites et précautions
        
        ⚠️ **Important :** Ces calculs sont indicatifs uniquement
        - La valorisation dépend de nombreux facteurs qualitatifs
        - Le contexte de marché influence fortement les résultats
        - Consultez des experts pour des décisions importantes
        """)

if __name__ == "__main__":
    main()
