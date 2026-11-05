# Differential Evolution

Ce dossier contient l’implémentation CPU et GPU de l’algorithme Differential Evolution.

- `main.cpp` : gestion des paramètres, initialisation et lancement des expériences
- `kernel.cpp` : implémentation séquentielle CPU
- `kernel.cu` : implémentation parallèle avec CUDA
- `kernel.h` : déclarations et paramètres communs
- `CudaDE.ipynb` : exécution, expérimentation et analyse des résultats

La version GPU parallélise les principales étapes de l’algorithme afin d’étudier le gain de performance par rapport à la version séquentielle CPU.
