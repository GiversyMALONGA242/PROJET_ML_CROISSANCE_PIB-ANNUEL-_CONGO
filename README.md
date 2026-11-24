# 🤖📊📈Prévision de la croissance du PIB réel — République du Congo

Ce dépôt présente une démarche complète de **Machine Learning appliqué aux séries temporelles** pour analyser, valider et projeter la **croissance annuelle du PIB réel de la République du Congo** jusqu'en **2030**.

Le projet est organisé en deux étapes :

* **Premier modèle (`v1` — Actuel) :** modèle autorégressif (`AR(2)`) avec régression `Ridge` (`StandardScaler` + `Ridge`), validation chronologique via `TimeSeriesSplit`, ajustement des hyperparamètres (`MAX_LAG = 2`, `alpha = 5.0`), prévision récursive jusqu'en **2030** et sauvegarde du modèle (`modele_ar_pib_congo_v1.joblib`).

* **Deuxième modèle (`v2` — À venir) :** extension vers un modèle enrichi par des **variables macroéconomiques exogènes** (`ARX` / multivarié) afin de mieux capter les chocs économiques réels.

## 1. Structure du dépôt

```text
prevision-pib-congo-ml/
├── notebook_and_script/                         # Notebook et scripts du projet
│   └── ML_PIB_COG.ipynb                         # Notebook principal (extraction, nettoyage, ADF, lags, ML, prévision 2030)
├── modele_ar_pib_congo_v1.joblib                # Modèle v1 figé et sauvegardé (Ridge, lag=2, alpha=5.0)
├── dataset/                                     # Données principales et extraites
│   ├── dataset_principal.csv                    # Jeu de données initial complet
│   └── data_cog.csv                             # Données extraites pour la République du Congo
├── assets/                                      # Graphiques PNG intégrés au README
│   ├── stationnarite_visuelle_pib_cog.png       # Visualisation initiale avant modélisation (stationnarité visuelle)
│   ├── serie_observee_vs_estimee_v0.png         # Série observée vs série estimée (configuration initiale)
│   └── serie_observee_vs_estimee_ar2_ridge.png  # Série observée vs série estimée (modèle AR(2) Ridge ajusté)
└── README.md                                    # Documentation complète du projet
```



## 2. Chargement, extraction, nettoyage et stationnarité

### Source et variable cible

* **Source :** indicateur macroéconomique de croissance annuelle du PIB réel (`NY.GDP.MKTP.KD.ZG`, World Development Indicators / Banque mondiale).

* **Pays ciblé :** République du Congo (`COG`).

### Étapes de chargement et d'extraction

1. **Chargement du dataset principal :** importation du fichier CSV principal dans un `DataFrame` `pandas` (`dataset/API_NY.GDP.MKTP.KD.ZG_DS2_fr_csv_v2_122.csv`) et vérification de la structure des colonnes et des pays présents.

2. **Extraction du sous-ensemble Congo (`data_cog`) :** filtrage sur la République du Congo pour isoler la chronologie d'intérêt, puis enregistrement de l'extrait dédié (`data_cog.csv` dans `dataset/`) afin de disposer d'une base claire pour la modélisation.

3. **Tri chronologique (`Year`) :** ordonnancement strict des observations par année croissante (`Year`) afin de garantir l'ordre temporel avant la génération des variables retardées (*lags*).

### Nettoyage et vérification de la stationnarité (visuelle + test ADF, sans KPSS)

* **Traitement des valeurs manquantes :** prise en charge des valeurs manquantes (notamment l'imputation par la moyenne de l'observation manquante en `1960` lors du prétraitement initial). *(Point d'attention méthodologique : réaliser une imputation globale avant le découpage temporel introduit une légère fuite d'information rétrospective à surveiller.)*

* **Appréciation visuelle de la stationnarité :** dès le tracé initial de la série de croissance du PIB congolais, la courbe paraît déjà stationnaire : elle oscille autour d'un niveau moyen sans tendance divergente sur l'ensemble de la période, malgré des chocs ponctuels marqués.

* **Test statistique ADF (et non KPSS) :** pour confirmer formellement cette intuition visuelle, seul le test **ADF (Augmented Dickey-Fuller)** a été réalisé (le test **KPSS** n'a pas été appliqué). Le test ADF rejette la racine unitaire (`p-value = 0.000`), confirmant la stationnarité de la série et permettant de modéliser directement `Croissance_PIB` sans différenciation supplémentaire.

## 3. Ce qui a été fait après le test ADF : lags, validation et modélisation

Une fois la stationnarité validée par le test ADF, le travail s'est déroulé selon les étapes suivantes :

1. **Création des variables retardées (*lags*) :** transformation de la série stationnaire en problème supervisé en générant des retards (`lag_1`, `lag_2`, ..., `lag_k`) sur `Croissance_PIB`, suivis d'un `.dropna()` sur les premières lignes incomplètes.

2. **Validation temporelle sans fuite (`TimeSeriesSplit`) :** évaluation chronologique des modèles face à une *baseline* naïve (`lag_1`) et à d'autres algorithmes (`ElasticNet`, `RandomForest`). Sur la configuration initiale de référence, la régression `Ridge` s'est démarquée avec un `MAE` moyen d'environ **5.115**.

3. **Ajustement des hyperparamètres (`MAX_LAG` et `alpha`) :** recherche ciblée sur la profondeur de retard (`MAX_LAG`) et la régularisation `Ridge` (`alpha`), retenant **`MAX_LAG = 2`** et **`alpha = 5.0`** (`MAE_moyen = 4.740`, `RMSE_moyen = 5.778`).

4. **Entraînement final, projection récursive jusqu'en 2030 et sauvegarde :** entraînement du `Pipeline` (`StandardScaler` + `Ridge(alpha=5.0)`) sur l'historique complet, génération de la prévision récursive jusqu'en **2030** (trajectoire douce et convergente d'environ **3.19 %** à **~3.38 %**), et exportation dans **`modele_ar_pib_congo_v1.joblib`**.

5. **Inspection graphique et préparation de la v2 :** comparaison visuelle *série observée vs série estimée* pour vérifier le comportement du modèle ajusté et poser les bases du **deuxième modèle (`v2`)** avec variables macroéconomiques exogènes.

## 4. Premier modèle (`v1`) : Modèle Autorégressif `AR(2)` avec `Ridge`

### Grille d'ajustement (`MAX_LAG` et `alpha`)

Une grille d'ajustement sur `MAX_LAG` (`1` à `4`) et `alpha` (`1.0` à `50.0`) évaluée via `TimeSeriesSplit` a permis d'affiner les performances :

| Rang | `MAX_LAG` | `alpha` | `MAE_moyen` | `RMSE_moyen` | Observation |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **2** | **5.0** | **4.740** | **5.778** | **Configuration retenue (`lag_optimal = 2`, `alpha_optimal = 5.0`)** |
| 2 | 2 | 10.0 | 4.742 | 5.820 | Très proche, régularisation légèrement plus marquée |
| 3 | 1 | 1.0 | 4.760 | 5.772 | Modèle `AR(1)` sobre, `RMSE` très légèrement inférieur |
| 4 | 2 | 1.0 | 4.775 | 5.753 | Moins de lissage sur les erreurs quadratiques |

Passer à **`MAX_LAG = 2`** et **`alpha = 5.0`** abaisse le **`MAE` moyen de `5.115` à `4.740`** et le **`RMSE` moyen à `5.778`**. Les deux retards annuels (`lag_1`, `lag_2`) concentrent l'information utile à court terme sans ajouter de bruit avec des retards plus anciens.

### Entraînement final, prévision jusqu'en 2030 et sauvegarde

* Le pipeline final (`StandardScaler` + `Ridge(alpha=5.0)`) est entraîné sur l'ensemble de l'historique disponible et enregistré dans **`modele_ar_pib_congo_v1.joblib`**.

* La **prévision récursive jusqu'en 2030** réinjecte chaque prévision annuelle comme prochain retard (`lag_1`, `lag_2`), produisant une trajectoire tendancielle douce et convergente (démarrant autour de **3.19 %** et se stabilisant progressivement vers **~3.38 %** en **2030**).

## 5. Visualisations et interprétation

> Assurez-vous que les trois fichiers PNG ci-dessous portent exactement ces noms dans le dossier `assets/` afin qu'ils s'affichent correctement sur GitHub.

### 5.1. Visualisation initiale — Stationnarité visuelle de la croissance du PIB (Congo)

![Stationnarité visuelle de la croissance du PIB du Congo](assets/stationnarite_visuelle_pib_cog.png)

* **Observation :** la croissance annuelle du PIB de la République du Congo paraît déjà stationnaire visuellement : elle oscille autour d'un niveau moyen avec des phases d'expansion et de contraction, sans dérive continue.

* **Interprétation :** cette appréciation visuelle, confirmée par le test **ADF** (`p-value = 0.000`, sans recours au test KPSS), valide la stationnarité de la série et permet de travailler directement sur `Croissance_PIB`.

### 5.2. Série observée vs Série estimée — Configuration initiale

![Série observée vs Série estimée — configuration initiale](assets/serie_observee_vs_estimee_v0.png)

* **Observation :** la série estimée suit le mouvement moyen de la série observée mais reste nettement plus lissée lors des variations rapides.

* **Interprétation :** la régularisation `Ridge` réduit la variance au prix d'un amortissement des pics élevés (au-delà de `15 %–20 %`) et des chutes marquées (sous `-5 %`).

### 5.3. Série observée vs Série estimée — Modèle `AR(2)` `Ridge` ajusté (`lag=2`, `alpha=5`)

![Série observée vs Série estimée — Modèle AR(2) Ridge ajusté](assets/serie_observee_vs_estimee_ar2_ridge.png)

* **Observation :** l'axe `Year` est gradué proprement de `1962` à la fin de l'historique, et la courbe orange (`Ridge, lag=2, alpha=5`) épouse plus souplement les évolutions récentes tout en restant plus douce que les chocs extrêmes.

* **Interprétation globale de la v1 :** parce qu'il s'appuie uniquement sur le passé de la variable cible (`lag_1`, `lag_2`), le modèle `AR(2)` constitue un **repère tendanciel (`baseline`)** utile jusqu'en **2030** (`~3.19 %` à `~3.38 %`), mais il ne peut pas anticiper à lui seul les chocs exogènes (variations pétrolières, chocs budgétaires ou crises externes).

## 6. Deuxième modèle (`v2`) : Extension avec variables macroéconomiques

Pour dépasser les limites d'un modèle purement autorégressif, la **v2** du projet étendra l'approche vers un modèle enrichi par des **variables macroéconomiques** (`ARX` / multivarié) :

1. **Intégration de variables explicatives macroéconomiques :** ajout d'indicateurs clés liés à l'économie congolaise (ex. : secteur pétrolier / matières premières, inflation, commerce extérieur, investissement ou finances publiques), alignés sur `Year`.

2. **Contrôle de la fuite temporelle et scénarios jusqu'en 2030 :** encadrement des transformations dans le `Pipeline`, usage de variables exogènes retardées ou de scénarios macroéconomiques explicites (prudent, central, optimiste) pour la projection jusqu'en **2030**.

3. **Comparaison formelle (`v1` vs `v2`) :** évaluation sur les mêmes découpages `TimeSeriesSplit` (`MAE`, `RMSE`) afin d'objectiver l'apport des variables macroéconomiques par rapport à la baseline `AR(2)` `Ridge`.
