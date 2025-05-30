import streamlit as st
from io import BytesIO
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image as RlImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import plotly.graph_objects as go

# -------------------------------------------------------------------------------------------------
# Classe principale de calculs (inchangée)                                         
class ValuationCalculator:
    """Classe principale pour les calculs de valorisation"""
    @staticmethod
    def dcf_valuation(cash_flows, growth_rate, discount_rate, terminal_growth=0.02):
        # ... logique DCF inchangée
        discounted_flows = []
        cumulative_pv = 0
        for i, cf in enumerate(cash_flows):
            year = i + 1
            discounted_cf = cf / ((1 + discount_rate) ** year)
            discounted_flows.append(discounted_cf)
            cumulative_pv += discounted_cf
        if len(cash_flows) > 0:
            terminal_cf = cash_flows[-1] * (1 + terminal_growth)
            terminal_value = terminal_cf / (discount_rate - terminal_growth)
            terminal_pv = terminal_value / ((1 + discount_rate) ** len(cash_flows))
        else:
            terminal_pv = 0
        total_valuation = cumulative_pv + terminal_pv
        return {'valuation': total_valuation, 'discounted_flows': discounted_flows, 'terminal_pv': terminal_pv}

    @staticmethod
    def market_multiples_valuation(metric_value, multiple, metric_type):
        # ... logique Multiples inchangée
        return {'valuation': metric_value * multiple, 'metric': metric_type}

    @staticmethod
    def scorecard_valuation(base_valuation, criteria_scores, weights):
        # ... logique Scorecard inchangée
        score = sum(criteria_scores[k] * weights[k] for k in criteria_scores)
        return {'valuation': base_valuation * score, 'score': score}

    @staticmethod
    def berkus_valuation(scores):
        # ... logique Berkus inchangée
        return {'valuation': sum(scores.values()), 'scores': scores}

    @staticmethod
    def risk_factor_summation(base, risk_factors):
        # ... logique Risk Factor inchangée
        total_adj = sum(risk_factors.values()) / (2 * len(risk_factors))
        return {'valuation': base * (1 + total_adj), 'total_adjustment': total_adj}

    @staticmethod
    def venture_capital_method(pre_money, post_money):
        # ... logique VC Method inchangée
        return {'valuation': post_money, 'pre_money': pre_money, 'post_money': post_money}
# -------------------------------------------------------------------------------------------------

# Fonctions de création de graphiques

def create_dcf_chart(details):
    flows = details['discounted_flows']
    fig = go.Figure([go.Bar(x=[f"Année {i+1}" for i in range(len(flows))], y=flows)])
    fig.update_layout(title="DCF - Flux actualisés", autosize=True)
    return fig


def create_multiples_chart(details):
    fig = go.Figure([go.Bar(x=[details['metric']], y=[details['valuation']])])
    fig.update_layout(title="Multiples", autosize=True)
    return fig


def create_scorecard_chart(details):
    fig = go.Figure([go.Bar(x=list(details['scores'].keys()), y=list(details['scores'].values()))])
    fig.update_layout(title="Scorecard - Poids critiques", autosize=True)
    return fig


def create_berkus_chart(details):
    fig = go.Figure([go.Bar(x=list(details['scores'].keys()), y=list(details['scores'].values()))])
    fig.update_layout(title="Berkus - Scores", autosize=True)
    return fig


def create_risk_factor_chart(details):
    fig = go.Figure([go.Bar(x=list(details['risk_factors'].keys()), y=list(details['risk_factors'].values()))])
    fig.update_layout(title="Risk Factor - Ajustements", autosize=True)
    return fig


def create_vc_chart(details):
    fig = go.Figure([go.Bar(x=['Pre-money', 'Post-money'], y=[details['pre_money'], details['post_money']])])
    fig.update_layout(title="VC Method", autosize=True)
    return fig


def create_comparison_chart(valuations):
    fig = go.Figure([go.Bar(x=list(valuations.keys()), y=list(valuations.values()))])
    fig.update_layout(title="Comparaison des valorisations", autosize=True)
    return fig


def generate_pdf_report(valuations_dict, detailed_dict, company_name="Startup"):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Rapport de Valorisation – {company_name}", styles['Heading1']))
    story.append(Spacer(1, 12))

    # Tableau récapitulatif
    data = [["Méthode", "Valorisation (€)"]] + [[m, f"{v:,.0f} €"] for m, v in valuations_dict.items()]
    table = Table(data, hAlign='LEFT')
    story.append(table)
    story.append(Spacer(1, 24))

    # Graphique de comparaison
    comp_fig = create_comparison_chart(valuations_dict)
    img_b = BytesIO(); comp_fig.write_image(img_b, format='PNG'); img_b.seek(0)
    story.append(Paragraph("Comparaison des valorisations", styles['Heading2']))
    story.append(RlImage(img_b, width=500, height=300))
    story.append(Spacer(1, 24))

    # Détails méthodes
    for method, details in detailed_dict.items():
        story.append(Paragraph(method, styles['Heading2']))
        if method == "DCF": fig = create_dcf_chart(details)
        elif method == "Multiples": fig = create_multiples_chart(details)
        elif method == "Scorecard": fig = create_scorecard_chart(details)
        elif method == "Berkus": fig = create_berkus_chart(details)
        elif method == "Risk Factor": fig = create_risk_factor_chart(details)
        elif method == "VC Method": fig = create_vc_chart(details)
        else: continue
        img_b = BytesIO(); fig.write_image(img_b, format='PNG'); img_b.seek(0)
        story.append(RlImage(img_b, width=500, height=300))
        story.append(Spacer(1, 24))

    doc.build(story)
    buffer.seek(0)
    return buffer


def main():
    st.set_page_config(page_title="Outil de Valorisation", layout="wide")

    # Initialisation du session_state
    for key in ('valuations', 'detailed_results', 'company_name', 'company_sector'):
        if key not in st.session_state:
            st.session_state[key] = {} if key in ('valuations', 'detailed_results') else "Startup"

    # Sidebar configuration
    st.sidebar.header("🎯 Configuration")
    st.sidebar.subheader("Infos Startup")
    st.session_state.company_name = st.sidebar.text_input("Nom", st.session_state.company_name)
    st.session_state.company_sector = st.sidebar.selectbox(
        "Secteur", list(), index=0
    )

    # Onglets
    tab1, tab2 = st.tabs(["Analyse & Valorisation", "📋 Rapport PDF"])

    with tab1:
        methods = ["DCF", "Multiples", "Scorecard", "Berkus", "Risk Factor", "VC Method"]
        for method in methods:
            st.subheader(method)
            details = {}
            fig = None
            if method == "DCF":
                n = st.number_input("Périodes", 1, 10, 5, key="dcf_n")
                cfs = [st.number_input(f"Flux A{i+1}", key=f"dcf_cf_{i}") for i in range(n)]
                if st.button("Calculer DCF", key="calc_dcf"):
                    details = ValuationCalculator.dcf_valuation(cfs, 0.1, 0.1)
                    fig = create_dcf_chart(details)
            elif method == "Multiples":
                val = st.number_input("Valeur métrique", key="mult_val")
                mul = st.number_input("Multiple", key="mult_mul")
                if st.button("Calculer Multiples", key="calc_mult"):
                    details = ValuationCalculator.market_multiples_valuation(val, mul, "Metric")
                    fig = create_multiples_chart(details)
            elif method == "Scorecard":
                base = st.number_input("Valeur de base", 0, key="sc_base")
                sc_scores = {f"Critère{i}": st.slider(f"Critère {i}", 0.0, 2.0, 1.0, key=f"sc_{i}") for i in range(1,6)}
                weights = {k: 1/5 for k in sc_scores}
                if st.button("Calculer Scorecard", key="calc_sc"):
                    details = ValuationCalculator.scorecard_valuation(base, sc_scores, weights)
                    fig = create_scorecard_chart({'scores': sc_scores, **details})
            elif method == "Berkus":
                bk_scores = {f"Param{i}": st.slider(f"Paramètre {i}", 0, 500_000, 100_000, step=50_000, key=f"bk_{i}") for i in range(1,6)}
                if st.button("Calculer Berkus", key="calc_bk"):
                    details = ValuationCalculator.berkus_valuation(bk_scores)
                    fig = create_berkus_chart(details)
            elif method == "Risk Factor":
                base_rf = st.number_input("Val. de base", 0, key="rf_base")
                rf_factors = {fries: st.slider(fries, -2, 2, 0, key=fries) for fries in [
                    "Strength", "Opportunity", "Technology", "Competition", "Regulation"
                ]}
                if st.button("Calculer Risk Factor", key="calc_rf"):
                    res = ValuationCalculator.risk_factor_summation(base_rf, rf_factors)
                    details = {'risk_factors': rf_factors, **res}
                    fig = create_risk_factor_chart(details)
            elif method == "VC Method":
                pre = st.number_input("Pre-money", key="vc_pre")
                post = st.number_input("Post-money", key="vc_post")
                if st.button("Calculer VC", key="calc_vc"):
                    details = ValuationCalculator.venture_capital_method(pre, post)
                    fig = create_vc_chart(details)

            if fig:
                st.plotly_chart(fig, use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("💾 Sauvegarder", key=f"save_{method}"):
                        st.session_state.valuations[method] = details['valuation']
                        st.session_state.detailed_results[method] = details
                        st.success(f"{method} sauvegardé.")
                with c2:
                    if st.button("🗑️ Effacer", key=f"clear_{method}"):
                        st.session_state.valuations.pop(method, None)
                        st.session_state.detailed_results.pop(method, None)
                        st.info(f"{method} effacé.")

    with tab2:
        st.header("📋 Rapport PDF")
        if st.button("Générer PDF", type="primary"):
            buf = generate_pdf_report(
                st.session_state.valuations,
                st.session_state.detailed_results,
                st.session_state.company_name
            )
            st.download_button("⬇️ Télécharger", buf,
                                file_name=f"rapport_{st.session_state.company_name}_{datetime.now():%Y%m%d}.pdf",
                                mime="application/pdf")

if __name__ == "__main__":
    main()
