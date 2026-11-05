# Differential Evolution parallèle avec CUDA

Projet réalisé dans le cadre du **Master 2 Ingénierie Mathématique et Data Science** à l’Université de Haute-Alsace.

L’objectif du projet est d’étudier l’accélération d’un algorithme d’optimisation métaheuristique, **Differential Evolution (DE)**, grâce au calcul parallèle sur GPU avec **NVIDIA CUDA**.

Nous avons développé une version séquentielle CPU et une version parallèle GPU, puis comparé leurs performances sur plusieurs fonctions d’optimisation.

## Objectifs du projet

- Implémenter l’algorithme Differential Evolution en C++
- Paralléliser les principales étapes de l’algorithme avec CUDA
- Comparer les performances CPU et GPU
- Étudier l’influence de la dimension et de la taille de population
- Vérifier statistiquement la qualité des solutions obtenues
- Analyser le compromis entre temps de calcul et convergence

## Algorithme étudié

L’implémentation utilise la stratégie :

`DE/rand/1/bin`

Chaque génération repose sur quatre étapes principales :

1. Mutation
2. Crossover binomial
3. Évaluation de la fonction objectif
4. Sélection

Dans la version GPU, les individus de la population sont traités en parallèle à l’aide de threads CUDA.

## Fonctions de benchmark

Les expérimentations ont été réalisées sur quatre fonctions classiques d’optimisation :

- Sphere
- Rosenbrock
- Griewank
- Rastrigin

Les configurations testées utilisent :

- Dimensions : `10`, `50`, `100`
- Populations : `50`, `100`, `500`
- Budget d’évaluation : `MaxFES = 10 000 × Dimension`
- 10 exécutions indépendantes pour chaque configuration

Au total, plusieurs centaines d’exécutions CPU/GPU ont été réalisées afin d’obtenir des résultats statistiquement exploitables.

## Technologies et compétences mobilisées

- C++
- NVIDIA CUDA
- cuRAND
- Calcul parallèle sur GPU
- Optimisation numérique
- Algorithmes évolutionnaires
- Analyse expérimentale
- Analyse statistique
- Wilcoxon signed-rank test
- Visualisation et interprétation des performances

## Résultats principaux

Les expérimentations montrent que l’accélération GPU devient particulièrement intéressante lorsque la taille de population augmente.

Le meilleur speedup observé sur les configurations testées est d’environ :

**16,9×**

Sur la fonction Rosenbrock, par exemple, avec :

`Dimension = 10`  
`Population = 500`

le speedup obtenu est d’environ :

**15×**

Les résultats montrent également que l’augmentation de la population améliore le potentiel de parallélisation, mais peut réduire le nombre de générations disponibles lorsque le budget `MaxFES` reste fixe.

## Comparaison de la qualité CPU / GPU

La comparaison statistique des résultats CPU et GPU a été réalisée à l’aide du test de Wilcoxon.

Sur la fonction Rosenbrock, une seule configuration sur neuf présente une différence statistiquement significative au seuil de 5 %.

Cela indique que les deux implémentations présentent globalement une qualité d’optimisation comparable, malgré des trajectoires aléatoires différentes.

## Ma contribution

Dans ce projet réalisé en équipe, mes principales contributions ont été :

- développement de la version séquentielle CPU de Differential Evolution ;
- implémentation du crossover parallèle avec CUDA ;
- gestion de la probabilité de crossover `CR` ;
- utilisation de `j_rand` afin de garantir au moins une composante issue du mutant ;
- gestion des contraintes avec clamping ;
- étude et analyse de la fonction Rosenbrock ;
- comparaison des temps d’exécution CPU / GPU ;
- analyse du speedup ;
- interprétation du test statistique de Wilcoxon ;
- contribution à l’architecture CPU/GPU et à la documentation du projet.

## Architecture du projet

```text
Rendu_Final/
│
├── de/
│   ├── main.cpp
│   ├── kernel.cpp
│   ├── kernel.cu
│   ├── kernel.h
│   ├── CudaDE.ipynb
│   ├── résultats CSV
│   └── visualisations
│
├── pso/
│   ├── implémentation CPU/GPU
│   ├── notebook d’analyse
│   └── résultats expérimentaux
│
├── rapport_final.pdf
│
└── Soutenance_DE_CUDA_final.pdf
```

## Exemple d’architecture CUDA

Dans le crossover GPU, chaque thread CUDA traite un individu de la population.

Le thread parcourt ensuite les dimensions de cet individu afin de construire le vecteur d’essai selon la probabilité de crossover.

Cette approche permet de paralléliser les opérations coûteuses lorsque la population devient importante.

## Ce que ce projet m’a apporté

Ce projet m’a permis de développer des compétences à l’intersection de la **Data Science, de l’optimisation numérique et du calcul haute performance**.

J’ai notamment approfondi :

- la conception d’expériences reproductibles ;
- la comparaison statistique d’algorithmes ;
- l’optimisation de performances ;
- la programmation parallèle ;
- l’analyse de compromis entre précision, convergence et temps de calcul.

Ces compétences sont directement transférables à des problématiques de Data Science nécessitant des traitements intensifs, de l’optimisation ou de grandes campagnes expérimentales.
