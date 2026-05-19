# 🏗️ Studio Medallion Lakehouse & RAG 
Ce projet propose une implémentation interactive de l'architecture de données Medallion (Bronze ➔ Silver ➔ Gold) couplée à un moteur d'inférence RAG (Retrieval-Augmented Generation).

# 📸 Aperçu de l'Application

Voici à quoi ressemble l'interface utilisateur interactive développée avec Streamlit et alimentée par l'écosystème OpenAI (GPT-4o-mini & Text Embeddings).

📥 1. Ingestion de données & Nettoyage par Copilote (Bronze ➔ Silver)

![Bronze ➔ Silver ](img/demo1.png)

Cette capture montre l'espace de travail de l'Onglet 1 : profilage automatique des données (lignes, colonnes, types, valeurs manquantes, doublons), génération de code de nettoyage Pandas par l'IA et comparaison Avant/Après en temps réel.

📈 2. Agrégations Décisionnelles & Tables Métiers (Silver ➔ Gold)

![Silver ➔ Gold ](img/demo2.png)

Cette capture montre l'Onglet 2 : chargement des données nettoyées de la couche Silver, génération de requêtes complexes d'agrégation métier (ex: "Moyenne du prix par catégorie") via le copilote et sauvegarde directe dans la couche Gold au format CSV.

# 📁 Structure du Projet
architecture-medallion/
├── app.py                     # Application principale Streamlit (Interface Studio)
├── pipeline_medallion.py      # Script de démonstration autonome (sans interface)
├── requirements.txt           # Dépendances Python
├── README.md                  # Ce guide d'accueil et d'explications
├── test_connection.py         # Guide de résolution des erreurs de connexion API
├── installation_ollama_windows.md # Guide d'installation d'Ollama pour l'usage local
└── local_medallion_lakehouse/ # Dossier de base créé automatiquement (exclu de Git)
    ├── bronze/                # Fichiers sources bruts (CSV, TXT)
    ├── silver/                # Données nettoyées, structurées et découpées (JSON, CSV)
    └── gold/                  # Index FAISS, projections d'embeddings et tables agrégées

## 🔑 Étape 1 : Configuration de la Clé API OpenAI

Pour fonctionner, l'application a besoin de votre clé API OpenAI. Vous devez la configurer sous forme de variable d'environnement sur votre système.

Sous Windows (Git Bash) :
```bash
export OPENAI_API_KEY="votre-cle-api-ici"
```
Sous Windows (PowerShell) :

```bash
$env:OPENAI_API_KEY="votre-cle-api-ici"
```
Sous macOS / Linux :

```bash
export OPENAI_API_KEY="votre-cle-api-ici"
```
## 🚀 Étape 2 : Lancement de l'Application

Activez votre environnement virtuel :

```bash
# Sous Windows (Git Bash)
source venv/Scripts/activate

# Sous macOS/Linux
source venv/bin/activate
```
Installez les dépendances requises :

```bash
pip install -r requirements.txt
```

Lancez le Studio Streamlit :

```bash
streamlit run app.py
```
L'application démarrera et s'ouvrira automatiquement à l'adresse http://localhost:8501.

# 🛠️ Guide d'Utilisation des Onglets du Studio

## 📥 1. Ingestion & Copilote (Bronze ➔ Silver)
Données Brutes (Bronze) : Déposez un fichier .csv ou .txt (ou cliquez sur le bouton dans la sidebar pour injecter des fichiers de démonstration).

Copilote Pandas Intelligent : Donnez une instruction en français (ex: "Dédoublonne le fichier et filtre les prix supérieurs à 150"). OpenAI va générer le code Pandas optimal pour exécuter la tâche.

Itération & Validation : Vous pouvez éditer le code généré, tester la transformation en temps réel ("Avant/Après"), puis la sauvegarder physiquement dans la couche Silver.

## 📈 2. Consolidation Métier (Silver ➔ Gold)
Tables Métier (CSV) : Appliquez des regroupements ou des calculs d'indicateurs complexes (ex: "Calcule la somme des ventes par catégorie"), prévisualisez-les et poussez-les dans la couche Gold.

Espace Sémantique (TXT) : Extrayez les fragments textuels nettoyés à l'étape précédente et vectorisez-les avec le modèle ultra-performant text-embedding-3-small pour mettre à jour l'index vectoriel local FAISS.

## 📊 3. Exploration, Profiling & Visuels (Gold)
Décisionnel : Analysez vos indicateurs Gold à l'aide de graphiques interactifs (Plotly).

Projection PCA : Visualisez les embeddings de vos textes projetés en 2D ou 3D (Analyse en Composantes Principales). Identifiez d'un coup d'œil la proximité sémantique de vos documents !

## 💬 4. Requêtage & RAG (Inférence)
Analyse de tables : Posez une question sur vos ventes et l'IA génèrera et exécutera le calcul à votre place sur vos tables Gold.

RAG Documentaire (FAQ Interne) : Interrogez votre base de connaissances. L'application cherche les morceaux de textes pertinents dans FAISS et sollicite GPT-4o-mini pour formuler une réponse fiable et sourcée.

Lignage (Lineage) : L'application affiche pour chaque réponse la traçabilité complète de l'information (de la réponse Gold ➔ vers le chunk Silver ➔ jusqu'au fichier Bronze d'origine).


# 🤖 Alternatives à OpenAI (Modèles Locaux & Autres APIs)

Bien que ce PoC soit configuré par défaut pour OpenAI afin d'offrir les meilleures performances de raisonnement et de génération de code sans configuration complexe, l'architecture est 100 % modulaire.

Grâce à l'intégration de LangChain, vous pouvez facilement remplacer OpenAI par d'autres moteurs :

LLM Locaux et Gratuits (Ollama) : Vous pouvez faire tourner vos modèles entièrement en local (sans clé API et de manière 100 % confidentielle) en utilisant Ollama avec des modèles légers et performants comme llama3.2 (pour le texte/code) et nomic-embed-text (pour les embeddings).

👉 Consultez le guide complet : installation_ollama_windows.md pour installer et configurer Ollama.

Note de développement : Dans le code, il suffit de remplacer ChatOpenAI par ChatOllama et OpenAIEmbeddings par OllamaEmbeddings depuis le package langchain_community.

Autres API Cloud (Anthropic, Mistral, Cohere) : Si vous préférez utiliser Claude d'Anthropic ou les modèles de Mistral AI, LangChain propose des connecteurs natifs prêts à l'emploi (ex: ChatAnthropic, MistralAIEmbeddings). Vous n'aurez qu'à adapter les imports et configurer les clés d'API correspondantes.

# 👤 Auteur
 Diaraye BARRY 
 Senior Data Scientist & Machine Learning Engineer
 Expertise en Architectures LLM, MLOps et Stratégies Data.























# 🚀 Démarrage Rapide

1. Prérequis

Assurez-vous d'avoir Python 3.10 ou supérieur installé sur votre machine.

2. Installation

Clonez le dépôt et installez les dépendances requises :
```bash
# Activation de votre environnement virtuel
source venv/bin/activate  # Sur Mac/Linux/Git Bash
# ou
venv\Scripts\activate     # Sur Windows (PowerShell)

# Installation des paquets requis
pip install -r requirements.txt
```

3. Configuration de la Clé API OpenAI

Vous pouvez configurer votre clé de deux façons :

En variable d'environnement (Recommandé) :

export OPENAI_API_KEY="votre-cle-api-ici"


Directement dans l'interface : Saisissez simplement votre clé dans le champ sécurisé de la barre latérale gauche lors du lancement de l'application.

4. Lancement

Exécutez l'application Streamlit locale :

```bash
streamlit run app.py


# 🤖 Alternatives à OpenAI (Modèles Locaux & Autres APIs)

L'architecture de ce projet repose sur LangChain, ce qui la rend entièrement modulaire. Si vous ne souhaitez pas utiliser les APIs d'OpenAI, vous pouvez facilement adapter le code pour utiliser :

Un modèle local gratuit avec Ollama (ex: llama3 pour le raisonnement et nomic-embed-text pour l'index vectoriel).

D'autres Clouds tels qu'Anthropic (Claude), Mistral AI ou Cohere.

(Pour en savoir plus sur l'installation locale d'Ollama, consultez notre guide interne installation_ollama_windows.md).

# 👤 Auteur

Diaraye Barry - Data Scientist 

Développé dans le cadre d'un projet de démonstration de l'architecture Medallion appliquée aux pipelines de données IA (RAG).