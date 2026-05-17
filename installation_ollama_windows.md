# 🦙 Installer et Activer Ollama sur Windows

L'erreur ollama: command not found indique que le programme Ollama n'est pas encore installé ou reconnu par votre terminal Windows (Git Bash / MINGW64).

## Étape 1 : Téléchargement et Installation

Allez sur le site officiel : ollama.com/download

Cliquez sur "Download for Windows" pour télécharger l'installateur OllamaSetup.exe.

Double-cliquez sur le fichier téléchargé et suivez les instructions rapides d'installation.

Une fois installé, l'icône d'Ollama (une tête de lama) doit apparaître dans votre barre des tâches Windows (en bas à droite, près de l'horloge).

## Étape 2 : Activer Ollama dans Git Bash

⚠️ IMPORTANT : Vous devez fermer complètement votre terminal Git Bash actuel et en ouvrir un nouveau. Si vous gardez le même terminal, il ne saura pas qu'Ollama a été installé car les variables système (PATH) n'ont pas été mises à jour.

Fermez votre terminal Git Bash actuel.

Ouvrez un nouveau terminal Git Bash.

Activez à nouveau votre environnement virtuel :

```bash
source venv/Scripts/activate
```

## Étape 3 : Télécharger les Modèles

Dans votre nouveau terminal, exécutez ces deux commandes l'une après l'autre :

### 1. Télécharger le LLM pour le code et le chat

```bash
ollama pull llama3
```

### 2. Télécharger le modèle d'embeddings pour l'onglet Gold / Visualisation
```bash
ollama pull nomic-embed-text
```

Chaque commande va télécharger quelques gigaoctets de données directement sur votre disque dur (le téléchargement peut prendre quelques minutes selon votre connexion internet).

### Étape 4 : Relancer l'application

Une fois les téléchargements terminés, relancez simplement votre application Streamlit :

```bash
streamlit run app.py
```

## 🛠️ Dépannage : Erreur d'allocation de mémoire (unable to allocate CPU buffer)

Si vous rencontrez le message d'erreur :
ollama._types.ResponseError: llama runner process has terminated: error loading model: unable to allocate CPU buffer panic: (status code: 500)

Cela signifie que votre ordinateur manque de RAM physique libre pour charger le modèle llama3 (8B). Voici comment résoudre ce problème étape par étape :

### Solution 1 : Libérer de la mémoire RAM

Les navigateurs web (Chrome, Edge) et les applications comme Slack, Teams ou Docker Desktop consomment énormément de mémoire vive.

Fermez tous vos onglets inutiles et quittez les applications lourdes en arrière-plan.

Essayez de recharger le modèle.

### Solution 2 : Redémarrer le service Ollama

Ollama peut parfois garder des anciens modèles en cache ou souffrir de fuites de mémoire temporaires.

Allez dans votre barre des tâches Windows (en bas à droite).

Faites un clic droit sur l'icône Ollama (la tête de lama) et cliquez sur Quit Ollama.

Relancez Ollama depuis votre menu Démarrer de Windows.

### Solution 3 : Utiliser un modèle plus léger (Recommandé pour les configurations < 16 Go RAM)

Le modèle llama3 par défaut fait 8 milliards de paramètres (~4.7 Go). Si votre PC possède 8 Go ou 16 Go de RAM globale, il est fortement conseillé d'utiliser la version llama3.2 qui est plus récente, beaucoup plus légère (3 milliards de paramètres, ~2.0 Go) et s'exécute parfaitement sur de petites configurations.

Téléchargez la version légère :

```bash
ollama pull llama3.2
```

Dans la barre latérale de l'application Streamlit, sous "Modèle LLM (Chat/Pandas)", sélectionnez simplement llama3.2:latest au lieu de llama3:latest.

Note : Pour les configurations extrêmement limitées (ex: 8 Go de RAM très saturée), vous pouvez même utiliser la version ultra-légère de 1 milliard de paramètres : ollama pull llama3.2:1b.

### Solution 4 : Augmenter le fichier d'échange (Virtual Memory Pagefile) de Windows

Si vous souhaitez absolument faire tourner le modèle de 8B sur un système limite, vous pouvez augmenter la mémoire virtuelle de Windows :

Ouvrez le menu démarrer et tapez "Ajuster l'apparence et les performances de Windows".

Allez dans l'onglet Avancé, puis dans la section Mémoire virtuelle, cliquez sur Modifier....

Décochez "Gestion automatique", sélectionnez votre disque principal (C:), choisissez Taille personnalisée :

Taille initiale : 8192 Mo (8 Go)

Taille maximale : 16384 Mo (16 Go)

Cliquez sur Définir, puis sur OK et redémarrez votre ordinateur.