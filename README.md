# Startup Valuation Calculator 📊

Une application web interactive permettant aux entrepreneurs de calculer rapidement la valorisation indicative de leur startup en utilisant plusieurs méthodes reconnues de valorisation.

## 🎯 Description

Cette application offre six méthodes de valorisation différentes pour évaluer votre startup :

- **Discounted Cash Flow (DCF)** - Valorisation basée sur les flux de trésorerie futurs actualisés
- **Multiples de marché** - Comparaison avec des entreprises similaires du secteur
- **Scorecard Method** - Évaluation basée sur des critères pondérés
- **Berkus Method** - Méthode spécifique aux startups pré-revenus
- **Risk Factor Summation** - Ajustement de la valorisation selon les facteurs de risque
- **Venture Capital Method** - Approche basée sur les attentes de sortie des investisseurs

## 🚀 Fonctionnalités principales

- **Interface intuitive** avec Streamlit
- **Calculs en temps réel** pour chaque méthode
- **Visualisations interactives** avec Plotly
- **Analyse comparative** entre différentes méthodes
- **Génération de rapports PDF** professionnels
- **Analyses de sensibilité** pour comprendre l'impact des paramètres

## 📦 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation locale

1. Clonez le repository :
```bash
git clone https://github.com/brainfkt/startup-valuation-calculator.git
cd startup-valuation-calculator
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Lancez l'application :
```bash
streamlit run app.py
```

4. Ouvrez votre navigateur à l'adresse : `http://localhost:8501`

## 🌐 Démo en ligne

[Lien vers la démo Streamlit Cloud - À ajouter après déploiement]

## 🏗️ Structure du projet

```
startup-valuation-calculator/
├── app.py                 # Application principale Streamlit
├── requirements.txt       # Dépendances Python
├── README.md             # Documentation
└── assets/               # Resources (images, données de référence)
    └── sector_multiples.json
```

## 🧮 Méthodes de valorisation expliquées

### 1. Discounted Cash Flow (DCF)
Méthode fondamentale basée sur l'actualisation des flux de trésorerie futurs. Particulièrement adaptée aux entreprises avec des revenus prévisibles.

### 2. Multiples de marché
Compare votre startup à des entreprises similaires en utilisant des ratios comme P/E, EV/EBITDA, Price/Sales.

### 3. Scorecard Method
Évalue la startup selon plusieurs critères (équipe, marché, produit, etc.) avec des pondérations ajustables.

### 4. Berkus Method
Méthode spécialement conçue pour les startups pré-revenus, évaluant 5 critères clés jusqu'à 500k€ chacun.

### 5. Risk Factor Summation
Ajuste une valorisation de base en fonction de 12 facteurs de risque standard.

### 6. Venture Capital Method
Calcule la valorisation basée sur les attentes de retour sur investissement des VCs.

## 📚 Références et ressources

- [Damodaran Online - Valuation](http://pages.stern.nyu.edu/~adamodar/)
- [Angel Capital Association - Valuation Methods](https://www.angelcapitalassociation.org/)
- [NVCA - Venture Capital Valuation](https://nvca.org/)
- [Berkus Method Official](https://berkusmethod.com/)

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

*Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub.*
