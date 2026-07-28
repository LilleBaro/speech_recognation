**Résumé des actions après l'utilisation de Git LFS**

- **Problème initial**: Le chargement du checkpoint produisait l'erreur `invalid load key, 'v'` — constaté que `models/mon_checkpoint.pth` était un pointeur Git LFS (fichier texte commençant par `version https://git-lfs.github.com/spec/v1`).

- **Diagnostic**: Vérification du contenu du fichier `models/mon_checkpoint.pth` et confirmation qu'il s'agissait d'un objet LFS, pas du véritable poids PyTorch.

- **Actions locales réalisées**:
  - **Ajout d'un fallback HF**: modification de [models/pipeline.py](models/pipeline.py) pour:
    - détecter un pointeur LFS ou l'absence du fichier local,
    - télécharger automatiquement le vrai checkpoint depuis Hugging Face via `huggingface_hub.hf_hub_download` (repo par défaut `LilleBaro/fr_sentiment_analysis` ou via `HF_REPO_ID`).
  - **Tokeniser**: bascule du chargement du tokenizer pour utiliser la source HF si disponible.
  - **Dépendances**: ajout de `huggingface_hub` dans `pyproject.toml` pour permettre le téléchargement automatique.
  - **Gestion du checkpoint**: maintien de la logique existante de `torch.load(..., weights_only=False)` et du fallback sur `checkpoint["model"]` / `checkpoint["state_dict"]`.
  - **Todo list mise à jour**: tâches de fallback et dépendance marquées comme complétées; test end-to-end restant.

- **Pourquoi cette approche**:
  - éviter de garder de gros fichiers binaires dans le dépôt Git en cas d'usage local incomplet (LFS pointer),
  - permettre un chargement fiable du modèle soit localement (si présent), soit via HF quand l'objet LFS n'est pas récupéré.

- **Étapes recommandées (à exécuter localement)**:

```powershell
# 1) Installer Git LFS (une seule fois)
git lfs install

# 2) Récupérer les objets LFS du dépôt (si vous voulez le vrai .pth localement)
git lfs pull

# 3) Installer la dépendance pour le fallback HF
python -m pip install huggingface_hub

# 4) (Optionnel) exporter le repo HF si vous utilisez un autre repo
# sous Windows PowerShell:
$env:HF_REPO_ID = "VotreUser/votre-repo"
```

- **Validation / tests**:
  - importer `models.pipeline` et appeler la route `/predict` (ou exécuter un petit script) pour vérifier que le checkpoint est bien chargé soit depuis `models/mon_checkpoint.pth`, soit téléchargé depuis Hugging Face.

- **Prochaines actions que je peux faire pour vous**:
  - exécuter `git lfs pull` depuis le dépôt (si vous me donnez l'autorisation),
  - lancer un test end-to-end local et vous transmettre les logs d'erreur le cas échéant.


Si vous voulez que je lance le `git lfs pull` ici, dites-le et je l'exécute (ou exécutez les commandes ci-dessus localement).