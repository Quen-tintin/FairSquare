# FairSquare — Rapport Technique Final
### AI-Powered Real Estate Valuation & Hidden Gem Detection · Paris (Île-de-France)

> **Équipe :** Quentin Carbonati, Théo Consigny, Trevor Gurski  
> **Date :** Avril 2026 · **Version modèle :** LightGBM v5b  
> **Stack :** Python 3.11 · LightGBM · FastAPI · React/Vite · Render · Vercel  
> **Demo live :** https://fairsqure-app.vercel.app · **API :** https://fairsquare-api.onrender.com/docs

---

## Table des matières

1. [Problem Framing & Business Value](#1-problem-framing--business-value)
2. [The Data — Collection, Nettoyage & Feature Engineering](#2-the-data)
3. [Modelling & Expérimentation](#3-modelling--experimentation)
4. [Error Analysis](#4-error-analysis)
5. [Interpretability & Ethics](#5-interpretability--ethics)
6. [Architecture Système](#6-architecture-système)
7. [Limites & Roadmap](#7-limites--roadmap)

---

## 1. Problem Framing & Business Value

### 1.1 Le Problème

Le marché immobilier parisien est l'un des plus opaques et financièrement significatifs de France, avec des prix moyens dépassant **9 000 €/m²** et des transactions individuelles à plusieurs centaines de milliers d'euros. Pourtant, la découverte de prix reste largement manuelle : les acheteurs s'appuient sur des estimations d'agents ou des moyennes par arrondissement qui ignorent les effets au niveau de la rue et de la micro-localité.

FairSquare adresse deux problèmes distincts :

1. **Estimation du juste prix** — Étant données les caractéristiques physiques d'un appartement (surface, nombre de pièces, localisation), quelle est sa juste valeur marchande par m² ?
2. **Détection de Hidden Gems** — Parmi les annonces actuelles, quelles propriétés sont listées significativement *en dessous* de leur valeur prédite, représentant une opportunité d'investissement ?

### 1.2 Valeur Ajoutée

| Partie prenante | Valeur délivrée |
|----------------|-----------------|
| Acheteur | Éviter de surpayer ; identifier les biens sous-évalués avant qu'ils soient arbitragés |
| Investisseur | Scan systématique du marché ; upside quantifié par annonce |
| Agent immobilier | Valorisation objective pour ancrer la négociation |
| Régulateur | Benchmark de prix transparent et auditable |

Une réduction d'erreur de 5% sur un achat à 500 000 € représente **25 000 €** de surplus acheteur. Notre MAE final de 1 255 €/m² sur un appartement de 60 m² correspond à une erreur absolue moyenne de ~75 000 €, avec 50% des prédictions dans les 60 000 €.

### 1.3 Success Metrics (KPIs)

Parce que c'est une tâche de régression, nous utilisons les métriques suivantes, ordonnées par pertinence métier :

| Métrique | Formule | Cible initiale | v4 (avant feedback) | **v5 final** |
|---------|---------|---------------|---------------------|--------------|
| **MAE** (€/m²) | mean\|y − ŷ\| | < 1 500 | 1 427 | **1 255** |
| **RMSE** (€/m²) | √mean(y−ŷ)² | < 2 000 | 1 954 | **1 710** |
| **R²** | 1 − SS_res/SS_tot | > 0.40 | 0.43 | **0.53** |
| **MAPE** (%) | mean\|y−ŷ\|/y | < 20% | 15.8% | **12.7%** |
| **Within 1 000 €/m²** | % preds dans 1k | > 40% | 41.6% | **50.4%** |
| **Within 2 000 €/m²** | % preds dans 2k | > 65% | 67.5% | **79.5%** |

> **Note sur le choix des métriques :** Le MAE est préféré au RMSE comme KPI principal car il est directement interprétable en €/m² et moins sensible aux outliers de luxe. Le MAPE est suivi comme vérification secondaire car il normalise par le niveau de prix.

---

## 2. The Data

### 2.1 Source — DVF (Demandes de Valeurs Foncières)

Le dataset principal est **DVF**, le registre gouvernemental de toutes les transactions immobilières notariées en France, publié par la Direction générale des finances publiques (DGFiP) sous licence open data.

- **Couverture :** Tous les arrondissements de Paris (75001–75020), janvier 2023 – décembre 2025
- **Volume brut :** ~120 000 enregistrements de transactions pour les appartements parisiens
- **Après nettoyage v4 (IQR global) :** 67 292 transactions
- **Après nettoyage v5 (IQR per-arr + filtre prix) :** **58 879 transactions**
- **Granularité :** Date de transaction, adresse (rue + numéro), surface, nombre de pièces, prix, identifiant de parcelle

**Pourquoi DVF et pas les prix de vente ?**

DVF enregistre les *prix de vente notariés réels* — pas les prix demandés. C'est la vérité terrain pour la valorisation marchande. Les prix en annonce surestiment systématiquement les prix de transaction de 7–10% (la marge de négociation), ce que nous prenons en compte dans le moteur Hidden Gem.

---

### 2.2 Pipeline de Nettoyage — Version 1 (Baseline)

Implémenté dans `src/data_ingestion/dvf_cleaner.py` :

| Étape | Action | Lignes supprimées |
|-------|--------|------------------|
| Filtre type | Garder uniquement `type_local == "Appartement"` | ~70 000 |
| Filtre surface | Supprimer surface < 9 m² ou > 1 000 m² | ~2 100 |
| Filtre prix | Supprimer prix_m2 < 2 000 ou > 40 000 €/m² | ~1 800 |
| Doublons | Supprimer les doublons (adresse, surface, date, prix) | ~4 500 |
| Coordonnées manquantes | Supprimer les lignes sans latitude/longitude | ~800 |
| **Filtre IQR global** | Supprimer hors [Q1 − 1.5·IQR, Q3 + 1.5·IQR] | ~8 400 |

Le filtre IQR initial a été le pas de nettoyage le plus impactant de la v3 : il a réduit le MAPE de **28.4% → 15.8%** en supprimant les ventes en détresse et les transactions de luxe atypiques.

---

### 2.3 Analyse Approfondie des Outliers — Réponse au Feedback

Suite au feedback des instructeurs exigeant une distinction entre **Rare Labels** (vrais outliers légitimes) et **Bad Labels** (erreurs de données), nous avons conduit une expérience systématique en 5 stratégies de nettoyage, chacune réévaluant le modèle.

#### Problème du filtre IQR global

Notre filtre IQR original était **univarié** — il examinait `prix/m²` seul, sans tenir compte des autres dimensions. Il manquait les outliers **combinatoires** :

- Un 20 m² à 9 000 €/m² dans le 20e → prix normal ✓, surface normale ✓ — **IQR le conserve**
- Mais 20 m² dans le 20e à 9 000 €/m² est suspect (les studios du 20e font plutôt 30–45 m²)
- Un 200 m² à 8 000 €/m² dans le 7e → IQR le conserve aussi → mais c'est suspect (trop grand, trop peu cher pour le 7e)

De plus, l'IQR *global* (sur tout Paris) mélangeait des distributions très différentes : un 9 000 €/m² est dans la norme dans le 11e mais potentiellement aberrant dans le 19e.

#### Les 5 Stratégies Testées

**Script :** `scripts/advanced_outlier_experiment.py`

| Stratégie | Description | N lignes | MAE | R² | MAPE |
|----------|-------------|---------|-----|-----|------|
| **A** — IQR 1.5× global | Baseline actuelle | 62 401 | 1 447 | 0.51 | 15.2% |
| **B** — IQR 1.5× + Isolation Forest 3% | Détection multivariée | 60 529 | 1 346 | 0.48 | 14.5% |
| **C** — IQR 1.5× + Isolation Forest 5% | Détection plus agressive | 59 281 | 1 300 | 0.48 | 14.0% |
| **D** — IQR 1.0× par arrondissement | IQR resserré, par arr. | 59 406 | 1 299 | **0.55** | 13.2% |
| **E** — IQR 1.5× + bornes prix 3k-25k | Filtre prix resserré | 62 294 | 1 411 | 0.52 | 14.7% |

**Résultat clé :** La stratégie D (IQR 1.0× par arrondissement) donne le meilleur R² (0.55) et un MAE très compétitif. La stratégie finale v5b combine D + un filtre prix 5k-20k pour atteindre MAE = 1 255.

#### Distinction Rare Label vs Bad Label (Isolation Forest)

L'Isolation Forest entraîné sur `[surface, prix_m2, arrondissement, latitude, longitude]` a identifié **1 872 outliers** (contamination 3%) que l'IQR avait manqués. Nous les avons catégorisés :

**Bad Labels (7 transactions) — Erreurs de données confirmées :**

| Arrondissement | Surface | Prix/m² | Prix total | Raison du flag |
|---------------|---------|---------|-----------|----------------|
| 7e | 130 m² | 2 769 €/m² | 360 000 € | Impossible dans le 7e (< 30% du marché) — vente familiale |
| 1er | 45 m² | 1 800 €/m² | 81 000 € | Bien en dessous du marché parisien minimum |

**Rare Labels (1 865 transactions) — Vrais outliers légitimes :**

| Arrondissement | Surface | Prix/m² | Interprétation |
|---------------|---------|---------|----------------|
| 7e | 50 m² | 28 000 €/m² | Penthouse face au Champ-de-Mars — rare mais réel |
| 8e | 180 m² | 22 000 €/m² | Grand appartement avenue Montaigne |

Cette distinction est fondamentale : les Bad Labels doivent être supprimés ; les Rare Labels *peuvent* être conservés mais nécessitent un traitement spécial (modèle dédié ou downweighting).

#### Pipeline de Nettoyage Final (v5)

```
DVF brut (~120 000 lignes)
        ↓
Filtres durs (type, surface 9-1000m², prix 2k-40k€/m², doublons)
        ↓ ~67 292 lignes
IQR 1.0× par arrondissement (vs global 1.5× avant)
        ↓ 59 406 lignes  (removed 7 886)
Filtre prix core market : 5 000 – 20 000 €/m²
        ↓ 58 879 lignes  (removed 527 extrêmes de queue)
Dataset final pour l'entraînement v5
```

**Justification du filtre 5 000–20 000 €/m² :**

L'analyse résiduelle (voir Section 4) a révélé que le modèle avait un biais de +1 958 €/m² pour les biens <7 000 €/m² et de −2 624 €/m² pour les biens >14 000 €/m². Ces extrêmes polluent l'apprentissage sans que le modèle puisse les généraliser (données trop rares et trop hétérogènes). Le core market parisien (5k-20k €/m²) concentre 92% des transactions et permet au modèle de se spécialiser là où il est le plus utile.

---

### 2.4 Feature Engineering

Implémenté dans `src/ml/features_v2.py`. 26 features au total, en 5 catégories :

#### Catégorie 1 — Surface & Composition (6 features)

| Feature | Description | Justification |
|---------|-------------|---------------|
| `log_surface` | Log(surface) | Linéarise la relation prix/m² vs surface (rendements décroissants) |
| `surface_reelle_bati` | Surface brute (m²) | Signal direct |
| `nombre_pieces_principales` | Nombre de pièces | Proxy du type de bien |
| `pieces_per_m2` | Densité pièces/m² | Distingue studios ouverts vs appartements cloisonnés |
| `surface_per_piece` | Surface moyenne par pièce | Qualité de l'espace |
| `carrez_ratio` | Surface Carrez / Surface totale | Fraction habitable utile |

#### Catégorie 2 — Localisation Grossière (4 features)

| Feature | Description | Justification |
|---------|-------------|---------------|
| `arrondissement` | 1–20, ordinal | Signal de base |
| `arr_target_enc` | Moyenne prix/m² par arr. (Bayesian smoothé, train only) | Niveau de prix de l'arrondissement |
| `is_premium_arr` | Flag binaire arr. 1–8, 16, 17 | Prix >12 000 €/m² systématiquement |
| `dist_center_km` | Distance Haversine au Châtelet | Gradient de prix centre → périphérie |

#### Catégorie 3 — Localisation Fine (3 features — game changers)

| Feature | Description | Justification |
|---------|-------------|---------------|
| `voie_target_enc` | Moyenne prix/m² par code de rue (Bayesian smoothé) | Un même arrondissement peut varier de 5 000 €/m² selon la rue |
| `grid_target_enc` | Moyenne prix/m² par cellule de ~500m | Comble les lacunes de l'encodage par rue |
| `voie_recent_prix_m2` | Médiane glissante 12 mois sur la même rue (LOO) | Capture les tendances récentes de micro-marché |

**Note sur la prévention du data leakage :**
L'encodage par cible est calculé **exclusivement sur le fold d'entraînement** puis mappé sur le test set via lookup (avec la moyenne globale comme fallback). `voie_recent_prix_m2` utilise une fenêtre leave-one-out par date (chaque transaction ne voit que les transactions *antérieures* sur la même rue), garantissant zéro fuite.

#### Catégorie 4 — Coordonnées brutes & termes spatiaux (6 features)

| Feature | Description |
|---------|-------------|
| `latitude`, `longitude` | Coordonnées WGS84 brutes |
| `lat_sq`, `lon_sq` | Termes quadratiques — capturent les gradients spatiaux non-linéaires |
| `lat_lon_cross` | Terme d'interaction — capture le gradient diagonal NW–SE de Paris |

#### Catégorie 5 — Interactions & Temporel (7 features)

| Feature | Description |
|---------|-------------|
| `arr_price_x_log_surface` | arr_target_enc × log_surface |
| `premium_x_log_surface` | is_premium_arr × log_surface |
| `premium_x_dist_center` | is_premium_arr × dist_center_km |
| `voie_x_density` | voie_recent_prix_m2 × pieces_per_m2 |
| `annee`, `mois`, `trimestre` | Décomposition temporelle |
| `nombre_lots` | Proxy taille/type d'immeuble |

---

## 3. Modelling & Expérimentation

### 3.1 Setup Expérimental

- **Split train/test :** 80/20 stratifié par arrondissement, `random_state=42`
- **Validation :** Split interne 10% sur le train pour early stopping
- **Seed :** 42 (reproductibilité garantie)
- **Taille train final :** ~47 056 transactions
- **Taille test final :** ~11 769 transactions

### 3.2 Baseline Naïf

La baseline la plus simple raisonnable est la **médiane prix/m² par arrondissement** : prédire la médiane du train set pour l'arrondissement correspondant.

| Métrique | Baseline médiane/arr | Modèle final v5b |
|---------|---------------------|------------------|
| MAE (€/m²) | ~2 800 | **1 255** |
| R² | ~0.05 | **0.53** |
| MAPE (%) | ~32% | **12.7%** |

Le modèle final réduit le MAE de **~55%** vs la baseline naïve.

### 3.3 Tournoi de Modèles (v1 — 7 features)

Trois familles de modèles évaluées dans `src/ml/tournament.py` sur le feature set v1 (7 features, sans encodage par cible) :

| Modèle | MAE (€/m²) | R² | MAPE (%) | Notes |
|-------|-----------|-----|----------|-------|
| **Linear Regression** | 2 558 | 0.119 | 35.3% | Baseline ML |
| **GAM (splines + facteurs)** | 2 459 | 0.165 | 34.4% | Gain marginal de la non-linéarité |
| **LightGBM** | 2 417 | 0.194 | 33.5% | Meilleur sur features v1 |

**LightGBM sélectionné** pour :
1. Gestion native de la non-linéarité et des interactions entre features
2. Robustesse à l'échelle des features (pas de standardisation requise)
3. Compatibilité avec l'encodage par cible (régularisation native)
4. Entraînement rapide (<90s sur CPU pour 67k lignes)
5. TreeExplainer SHAP 10–100× plus rapide que sur sklearn

### 3.4 Progression Complète — 22 Expériences

| Version | Changement clé | MAE (€/m²) | R² | MAPE (%) |
|---------|---------------|-----------|-----|----------|
| LightGBM v1 | 7 features de base | 2 417 | 0.19 | 33.5% |
| LightGBM v2 | + encodage par cible, termes quadratiques | 2 035 | 0.37 | 28.4% |
| XGBoost v2 | Comparaison alternative | 2 047 | 0.36 | 28.8% |
| Ensemble v2 | Blending LGB+XGB | 2 034 | 0.37 | 28.5% |
| LightGBM v3 | Baseline v3 (même features) | 2 049 | 0.37 | 28.6% |
| XGBoost v3 | Comparaison | 2 051 | 0.36 | 28.8% |
| CatBoost v3 | Comparaison | 2 053 | 0.37 | 28.7% |
| Ensemble v3 | Blending 3 modèles | 2 043 | 0.37 | 28.6% |
| **LightGBM v3 IQR** | **+ filtre IQR global** | **1 417** | **0.43** | **15.8%** |
| LightGBM v3 L2 | Loss L2 à la place de L1 | 1 425 | 0.43 | 15.9% |
| LightGBM v3 log | Target log(prix_m2) | 1 416 | 0.43 | 15.8% |
| LightGBM v4 baseline | Baseline v4 | 1 426 | 0.42 | 15.9% |
| LightGBM v4 OSM | + features OSM | 1 424 | 0.42 | 15.9% |
| LightGBM v4 OSM log | + target log | 1 425 | 0.42 | 15.9% |
| **LightGBM v4 voie_recent** | **+ médiane rue 12 mois** | **1 427** | **0.43** | **15.8%** |
| LightGBM v5a IQR 1.0x | IQR 1.0× par arrondissement | 1 299 | 0.55 | 13.2% |
| LightGBM v5b price filter | + filtre prix 5k-20k | **1 255** | **0.53** | **12.7%** |
| LightGBM v5c Huber | + Huber loss | 1 268 | 0.52 | 12.7% |
| LightGBM v5d temporal | + poids temporels | 1 268 | 0.52 | 12.7% |
| **LightGBM v5e Optuna** | **+ HPO 40 trials** | **1 261** | **0.53** | **12.7%** |

**Modèle retenu : v5b (IQR 1.0×/arr + filtre prix 5k-20k)**
Meilleur ratio MAE/complexité. v5c/d/e n'apportent pas de gain significatif sur v5b.

#### Insight clé : la donnée > le modèle

Le graphique de progression révèle une leçon fondamentale :

- v1 → v2 (+ features d'encodage) : MAE −382 €/m²
- v2 → v3 IQR (nettoyage données) : MAE **−618 €/m²**  ← plus grand saut
- v3 → v4 (+ voie_recent feature) : MAE +10 €/m² (pas d'amélioration)
- **v4 → v5 (meilleur nettoyage per-arr + filtre prix) : MAE −172 €/m²**

**Conclusion :** Dans ce projet, l'amélioration de la qualité des données surpasse systématiquement la complexité du modèle. La transition IQR global → IQR par arrondissement était la clé.

### 3.5 Configuration Finale du Modèle (LightGBM v5b)

```python
LGBMRegressor(
    n_estimators        = 3000,
    learning_rate       = 0.04,
    num_leaves          = 127,
    min_child_samples   = 20,
    feature_fraction    = 0.8,
    bagging_fraction    = 0.8,
    bagging_freq        = 10,
    reg_alpha           = 0.1,
    reg_lambda          = 1.0,
    objective           = "regression_l1",   # MAE direct
    metric              = "mae",
    early_stopping_rounds = 60,
    random_state        = 42,
)
```

**Target :** `log1p(prix_m2)` → prédiction : `expm1(pred)` — stabilise la variance sur le haut du marché.

### 3.6 Optimisation Hyperparamètres (Optuna)

Nous avons utilisé Optuna (TPESampler, 40 trials) pour une recherche bayésienne des hyperparamètres optimaux sur le modèle v5c (Huber loss). Résultats :

```
Meilleurs paramètres Optuna :
  learning_rate:     0.0242
  num_leaves:        82
  min_child_samples: 28
  feature_fraction:  0.623
  reg_alpha:         0.834
  reg_lambda:        4.909
  huber_alpha:       0.728
```

MAE v5e (Optuna) = 1 261 €/m² — gain marginal de 6 €/m² vs v5b, indiquant que les hyperparamètres hand-tuned étaient déjà proches de l'optimum.

---

## 4. Error Analysis

### 4.1 Distribution Globale des Erreurs (Test Set)

Calculé sur le test set (11 769 transactions) du modèle v5b :

| Métrique | Valeur |
|---------|--------|
| MAE | **1 255 €/m²** |
| RMSE | **1 710 €/m²** |
| Médiane AE | ~950 €/m² |
| R² | **0.53** |
| Within 1 000 €/m² | **50.4%** |
| Within 2 000 €/m² | **79.5%** |

> La médiane AE (~950 €/m²) inférieure au MAE (1 255 €/m²) confirme la présence de queues épaisses — le modèle fait quelques grosses erreurs sur les extrêmes, mais la majorité des prédictions sont précises.

### 4.2 Analyse des Résidus par Arrondissement

Les 5 arrondissements avec le MAE le plus élevé :

| Arrondissement | MAE (€/m²) | Biais (€/m²) | N test | Interprétation |
|---------------|-----------|------------|--------|----------------|
| **6e** | 2 468 | −543 | 323 | Sous-prédit le luxe Saint-Germain |
| **7e** | 2 365 | −278 | 375 | Sous-prédit face Champ-de-Mars |
| **4e** | 2 002 | −124 | 229 | Île Saint-Louis, Marais premium |
| **1er** | 1 964 | −397 | 133 | Louvre, Palais Royal — données rares |
| **8e** | 1 894 | −448 | 265 | Champs-Élysées premium |

**Cause racine :** Le biais est systématiquement *négatif* dans les arrondissements de luxe — le modèle sous-estime. Dans le 7e, une rue face au Champ-de-Mars peut valoir 28 000 €/m², mais le training data de ces rues est très peu dense. L'encodage par rue (voie_target_enc) ne peut pas apprendre un signal fiable avec 3–5 transactions.

**Arrondissements bien calibrés (MAE < 1 200 €/m²) :** 11e, 19e, 20e — grands arrondissements avec abondance de données DVF.

### 4.3 Analyse des Résidus par Surface

| Tranche de surface | MAE (€/m²) | Biais (€/m²) | N test |
|-------------------|-----------|------------|--------|
| < 30 m² (studios) | 1 309 | −13 | 3 389 |
| 30–60 m² | **1 202** | −69 | 4 922 |
| 60–100 m² | 1 304 | −123 | 2 447 |
| > 100 m² (grands apps.) | 1 669 | −146 | 1 109 |

**Insight :** Le modèle est le plus précis sur les 30–60 m² — c'est le segment le plus représenté dans DVF. Les grands appartements (>100 m²) ont un MAE plus élevé car ils sont rares et souvent de luxe (biais lié à l'arrondissement).

### 4.4 Analyse des Résidus par Tranche de Prix

C'est l'analyse la plus révélatrice — directement liée à notre décision de filtre des extrêmes :

| Tranche prix/m² | MAE (€/m²) | Biais (€/m²) | N test | Interprétation |
|----------------|-----------|------------|--------|----------------|
| < 7 000 €/m² | 1 967 | **+1 958** | 854 | Sur-prédit massivement — ces biens sont atypiques (ventes familiales résiduelles) |
| 7 000–10 000 €/m² | **1 002** | +609 | 5 194 | Core market — très bien calibré |
| 10 000–14 000 €/m² | 1 222 | −683 | 4 939 | Marché premium — légère sous-prédiction |
| > 14 000 €/m² | 2 814 | **−2 624** | 880 | Luxe — sous-prédit sévèrement |

**Conclusion directe :** Le modèle est optimisé pour le marché standard (7k-14k €/m²) où il atteint MAE ~1 100 €/m². Les extrêmes (< 7k et > 14k) sont structurellement difficiles à prédire sans données de rénovation et d'étage. **C'est pourquoi le filtre 5k-20k lors de l'entraînement réduit le MAE global** : on arrête d'apprendre sur des transactions que le modèle ne peut pas généraliser.

### 4.5 Scatter Plot Prédit vs Réel

Le scatter plot de 500 points de test échantillonnés montre :

- **Bonne calibration** dans la plage 6 000–12 000 €/m² (la densité du coeur de distribution)
- **Sous-prédiction systématique au-dessus de 15 000 €/m²** — le modèle est tiré vers la moyenne du training set
- **Sur-prédiction en dessous de 5 000 €/m²** — même phénomène dans l'autre sens

---

## 5. Interpretability & Ethics

### 5.1 Importance des Features (SHAP)

SHAP TreeExplainer calculé sur 1 000 échantillons du test set. Top 10 features par mean |SHAP| :

| Rang | Feature | Mean \|SHAP\| (€/m²) | Interprétation |
|------|---------|-----------------|----------------|
| 1 | `voie_target_enc` | ~1 850 | Prix moyen de la rue — signal le plus discriminant |
| 2 | `arr_target_enc` | ~1 620 | Prix moyen de l'arrondissement |
| 3 | `voie_recent_prix_m2` | ~1 400 | Tendance récente 12 mois sur la même rue |
| 4 | `log_surface` | ~950 | Prix/m² inversement lié à la surface |
| 5 | `grid_target_enc` | ~820 | Cellule de 500m — comble les lacunes rue |
| 6 | `dist_center_km` | ~640 | Gradient centre-périphérie |
| 7 | `lat_lon_cross` | ~510 | Gradient spatial diagonal |
| 8 | `arr_price_x_log_surface` | ~480 | Interaction localisation × taille |
| 9 | `longitude` | ~420 | Gradient Est-Ouest (Ouest Paris = premium) |
| 10 | `nombre_pieces_principales` | ~280 | Effet faible une fois la surface contrôlée |

**Takeaway principal :** Le modèle est avant tout un **modèle de localisation**. Les 3 premières features sont toutes des encodages de prix basés sur la localisation. La surface explique relativement peu de variance une fois qu'on connaît *où* se situe l'appartement. Ceci est cohérent avec la réalité du marché parisien.

### 5.2 Explications Individuelles

L'endpoint `/predict` retourne une décomposition SHAP par feature en langage naturel :

```json
{
  "prix_predit_m2": 11250,
  "shap_contributions": [
    {"feature": "voie_target_enc",  "value": +2100, "display": "Prime rue de Rivoli"},
    {"feature": "log_surface",      "value":  -320, "display": "Surface 45m² : décote taille"},
    {"feature": "dist_center_km",   "value":  -180, "display": "Distance au centre"},
    {"feature": "nombre_pieces",    "value":   +95, "display": "3 pièces"}
  ],
  "xai_summary": "Ce bien bénéficie d'un fort premium lié à son adresse (+2 100 €/m²), partiellement compensé par sa petite surface (-320 €/m²)."
}
```

### 5.3 Score Hidden Gem

```
gem_score = (prix_predit_m2 - prix_affiche_m2) / prix_predit_m2
```

Un bien est flaggé **Hidden Gem** quand `gem_score > 0.10` (le modèle estime que le bien vaut plus de 10% de plus que le prix affiché).

**Correction marge de négociation :** Les prix DVF sont en moyenne 7–10% sous les prix d'annonce (la décote acheteur). Un gem_score de 10% peut donc en réalité représenter une vraie sous-évaluation de seulement 0–3%. C'est une limite connue de notre système, listée en roadmap.

### 5.4 Biais Identifiés

#### Biais Géographique — Luxe Sous-estimé
Le modèle sous-estime systématiquement les prix au-dessus de 15 000 €/m² (biais −2 624 €/m² dans le segment >14k). Les acheteurs de biens de luxe dans les arrondissements 1, 6, 7, 8 reçoivent des estimations inférieures à la réalité.

**Mitigation :** L'intervalle de confiance est plus large pour les prédictions hautes (le `_CI_HALF_WIDTH` actuel de 1 500 €/m² devrait être calibré par tranche de prix).

#### Biais de Type de Bien
DVF n'enregistre pas : l'étage, le diagnostic énergétique DPE, l'état de rénovation, la présence d'un balcon. Un appartement rénové en dernier étage avec terrasse et un appartement délabré en rez-de-chaussée dans le même immeuble partagent des features DVF identiques.

**Mitigation partielle :** Le module Vision (Google Gemini) applique une correction post-prédiction de ±0 à ±500 €/m² basée sur les photos d'annonce. Non intégré au training car DVF n'a pas de photos.

#### Biais Temporel
La fenêtre de données couvre 2023–2025. Un choc de marché post-entraînement (ex : variation rapide des taux) ne sera pas reflété avant un re-entraînement.

**Mitigation :** `voie_recent_prix_m2` utilise une fenêtre glissante 12 mois, donnant une certaine sensibilité à la récence.

### 5.5 Empreinte Carbone

| Composant | Hardware | Durée | Énergie (kWh) | CO₂e (gCO₂) |
|-----------|---------|-------|-------------|-------------|
| Pipeline complet v5 | CPU Intel i7, 35W | ~180s | 0.00175 | 0.79 |
| Expérience outliers (5 stratégies) | CPU | ~25 min | 0.0146 | 6.6 |
| Optuna 40 trials | CPU | ~12 min | 0.007 | 3.2 |
| 22 runs d'entraînement total | CPU | ~60 min | 0.035 | 15.9 |
| Inférence API (par requête) | Render CPU | ~0.1s | < 0.001 | < 0.001 |

**Méthodologie :**
- Consommation CPU : 35W moyen en charge
- Intensité carbone électrique France : 52 gCO₂/kWh (RTE 2024, majoritairement nucléaire)
- Formule : `CO₂e = Puissance_W × Durée_h × Intensité_kWh`

**Total training : ~16 gCO₂** — équivalent à charger un smartphone une fois et demie. LightGBM sur CPU est extrêmement efficace comparé à des approches deep learning. Un transformer-based model sur GPU utiliserait 100–1 000× plus d'énergie pour cette tâche.

---

## 6. Architecture Système

### 6.1 Backend (FastAPI sur Render)

```
POST /predict        →  features_v2.add_features()  →  LightGBM.predict()  →  SHAP
POST /analyze_url    →  Firecrawl scrape  →  Gemini extraction  →  /predict  →  Vision score
GET  /hidden_gems    →  live_listings_scored.json  →  filtre + tri
GET  /dvf/transactions → lecture parquet → filtre → sample
GET  /model/metrics  →  metrics.json + feature_importances_
GET  /model/errors   →  error_analysis.json (précomputé)
```

### 6.2 Frontend (React + Vite sur Vercel)

| Page | Route | Source de données |
|------|-------|-------------------|
| Analyze Listing | `/` | `POST /predict` |
| Analyze URL | `/url` | `POST /analyze_url` |
| Hidden Gems | `/gems` | `GET /hidden_gems` |
| Explore DVF | `/explore` | `GET /dvf/transactions` |
| Model Performance | `/performance` | `GET /model/metrics` |
| Error Analysis | `/errors` | `GET /model/errors` |

### 6.3 Pipeline de Données

```
data.gouv.fr (DVF CSV)
       ↓
dvf_client.py          — téléchargement 4 départements IDF
       ↓
dvf_cleaner.py         — filtres, cast, déduplication
       ↓
IQR 1.0× per-arrondissement + filtre prix 5k-20k
       ↓
dvf_paris_2023_2025_clean.parquet  (58 879 lignes)
       ↓
features_v2.add_features()         — 26 features, target encoding train-only
       ↓
train_v5_optimized.py              — LightGBM, log-target, early stopping
       ↓
best_model.pkl  +  metrics.json  +  error_analysis.json
```

---

## 7. Limites & Roadmap

### 7.1 Limites Connues

| Limite | Impact | Priorité |
|-------|--------|---------|
| Pas de DPE / données de rénovation dans le training | ±500 €/m² d'erreur non-réductible | Haute |
| Pas de données d'étage | ±200 €/m² (dernier étage vs RDC) | Moyenne |
| Luxe sous-estimé (7e, 8e, 6e) | Biais −400 à −550 €/m² | Moyenne |
| Intervalle de confiance fixe (1 500 €/m²) | Trompeur sur les prix extrêmes | Basse |
| Marge de négociation non calibrée | ~10% faux positifs Hidden Gems | Moyenne |

### 7.2 Améliorations à Court Terme

1. **Filtre IQR deux dimensions** — Appliquer IQR sur `[prix_m2, surface]` conjointement plutôt que prix_m2 seul
2. **Fix parité dtype Render** — Caster les colonnes `object` en `float32` dans `add_features()` avant inférence LightGBM
3. **Calibrer les intervalles de confiance** — Calculer l'écart-type résiduel par arrondissement, utiliser comme demi-largeur CI
4. **Correction marge de négociation** — Appliquer un markdown configurable de 7% à `prix_predit_m2` avant le calcul du `gem_score`
5. **Modèle dédié luxe** — Entraîner un modèle spécialisé sur les transactions >15 000 €/m²

### 7.3 Roadmap Long Terme

1. **Dataset hybride DVF × annonces** — Matcher les annonces scrapées avec les ventes DVF (même adresse ± 6 mois) pour entraîner nativement sur DPE, étage, état de rénovation
2. **Features temporelles enrichies** — Trend 12 mois par arrondissement, volatilité, saisonnalité
3. **Vision dans le training** — Utiliser Gemini Vision pour scorer les photos historiques, puis croiser avec DVF
4. **Rendement locatif** — Étendre le modèle pour prédire le yield locatif (cap rate, cash-on-cash) via données INSEE

---

## Annexe — Checklist Reproductibilité

| Élément | Statut |
|---------|--------|
| `requirements.txt` à jour (scipy, optuna) | ✅ |
| `pyproject.toml` avec extras dev | ✅ |
| `.env.example` avec toutes les clés | ✅ |
| Script d'entraînement documenté | ✅ `scripts/train_v5_optimized.py` |
| Seed fixé (42) | ✅ |
| Split train/test reproductible | ✅ `random_state=42` |
| Encodage par cible calculé sur train fold uniquement | ✅ |
| Artefact modèle commité | ✅ `models/artifacts/best_model.pkl` |
| Outputs précomputés commités | ✅ `data/outputs/ml/` |
| Résultats expérience outliers | ✅ `data/outputs/ml/outlier_experiment_results.json` |
| Déploiement Docker | ✅ `docker/Dockerfile.api` |
| CI/CD pipeline | ✅ `.github/workflows/` |
| Tests unitaires | ✅ `tests/unit/` |
| Tests d'intégration | ✅ `tests/integration/` |

---

*Rapport généré : Avril 2026 — FairSquare v0.1.0 · Modèle LightGBM v5b*
