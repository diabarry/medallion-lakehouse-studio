import os
import json
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
import plotly.express as px
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION DE L'ARCHITECTURE MEDALLION LOCALE ---
BASE_DIR = "local_medallion_lakehouse"
BRONZE_DIR = os.path.join(BASE_DIR, "bronze")
SILVER_DIR = os.path.join(BASE_DIR, "silver")
GOLD_DIR = os.path.join(BASE_DIR, "gold")

# Création automatique de l'arborescence locale
for folder in [BRONZE_DIR, SILVER_DIR, GOLD_DIR]:
    os.makedirs(folder, exist_ok=True)

EMBEDDINGS_ARCHIVE_PATH = os.path.join(GOLD_DIR, "embeddings_archive.json")
HISTORY_LOG_PATH = os.path.join(BASE_DIR, "modification_history.json")

# --- INITIALISATION DE L'INTERFACE STREAMLIT ---
st.set_page_config(
    page_title="Medallion Studio - Diaraye Barry", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS Premium adaptable (Sombre/Clair)
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION DE L'AUTHENTIFICATION OPENAI ---
with st.sidebar:
    st.title("🏗️ Studio OpenAI Cloud")
    # Ajout du profil dans la barre latérale
    st.markdown("👤 **Développé par :**")
    st.markdown("🚀 **Diaraye Barry**\n*Data Scientist*")
    st.markdown("---")
    
    # Récupération de la clé API
    api_key_env = os.environ.get("OPENAI_API_KEY", "")
    
    st.subheader("🔑 Authentification OpenAI")
    if api_key_env:
        st.success("✅ Clé API détectée dans le système !")
        openai_key = api_key_env
    else:
        st.warning("⚠️ Clé API non configurée en variable d'environnement.")
        openai_key = st.text_input(
            "Entrez votre clé API OpenAI (sk-...) :",
            type="password",
            help="Votre clé restera active uniquement pour cette session en mémoire."
        )
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            st.success("Clé API temporaire enregistrée en mémoire !")
            st.rerun()
        else:
            st.info("💡 Ajoutez la variable d'environnement `OPENAI_API_KEY` pour ne plus avoir à la saisir.")
            st.stop()

    # Sélection des modèles de production cloud
    st.subheader("🤖 Sélection des Modèles")
    selected_chat_model = st.selectbox(
        "Modèle de Raisonnement / Code :",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
        help="gpt-4o-mini est ultra-rapide et extrêmement économique."
    )
    
    selected_embed_model = st.selectbox(
        "Modèle d'Embeddings :",
        ["text-embedding-3-small", "text-embedding-3-large"],
        index=0,
        help="text-embedding-3-small produit d'excellents vecteurs à prix minime."
    )
    
    st.markdown("---")
    
    # Injection des données d'exemples
    st.subheader("💡 Mise en route")
    if st.button("🚀 Injecter des fichiers de démonstration"):
        # 1. Dataset structuré d'exemple (CSV brut)
        sample_df = pd.DataFrame({
            "Transaction_ID": [1001, 1002, 1003, 1003, 1004, 1005, 1006, 1007, 1008, 1008],
            "Date": ["2026-05-10", "2026-05-11", "2026-05-12", "2026-05-12", "2026-05-13", None, "2026-05-15", "2026-05-16", "2026-05-17", "2026-05-17"],
            "Client_ID": ["C_902", "C_112", "C_430", "C_430", "C_881", "C_202", "C_112", "C_506", "C_701", "C_701"],
            "Produit_Categorie": ["Électronique", "Mode", "Maison", "Maison", "Électronique", "Sport", None, "Maison", "Mode", "Mode"],
            "Quantite": [1, 2, 5, 5, 1, 3, 2, 1, 4, 4],
            "Prix_Unitaire": [299.99, 45.00, 12.50, 12.50, 899.00, 24.99, 150.00, None, 19.99, 19.99],
            "Statut_Retour": ["Non", "Non", "Oui", "Oui", "Non", "Non", "Non", "Non", "Non", "Non"]
        })
        sample_df.to_csv(os.path.join(BRONZE_DIR, "demo_transactions_brutes.csv"), index=False)
        
        # 2. Document textuel d'exemple
        sample_txt = """POLITIQUE DE SÉCURITÉ ET DE TÉLÉTRAVAIL - TECHVITALITY SAS
        Article 1 - Éligibilité au Télétravail : Chaque collaborateur en contrat à durée indéterminée (CDI) ayant validé sa période d'essai est éligible à un maximum de 3 jours de télétravail par semaine.
        Article 2 - Horaires de Disponibilité : Pendant les journées de télétravail, les salariés doivent être joignables de 9h30 à 12h00, et de 14h00 à 17h00.
        Article 3 - Remboursement des Frais : Une indemnité forfaitaire de 2.50 euros par jour de télétravail est allouée, dans la limite de 30 euros par mois."""
        
        with open(os.path.join(BRONZE_DIR, "demo_charte_teletravail.txt"), "w", encoding="utf-8") as f:
            f.write(sample_txt)
            
        st.success("Données de test injectées en Bronze !")
        st.rerun()

# --- INITIALISATION DE L'HISTORIQUE D'AUDIT ---
if "active_df" not in st.session_state:
    st.session_state.active_df = None
if "active_filename" not in st.session_state:
    st.session_state.active_filename = ""
if "active_stage" not in st.session_state:
    st.session_state.active_stage = ""
if "history" not in st.session_state:
    if os.path.exists(HISTORY_LOG_PATH):
        try:
            with open(HISTORY_LOG_PATH, "r", encoding="utf-8") as h_f:
                st.session_state.history = json.load(h_f)
        except Exception:
            st.session_state.history = []
    else:
        st.session_state.history = []

def save_history_to_disk():
    try:
        with open(HISTORY_LOG_PATH, "w", encoding="utf-8") as h_f:
            json.dump(st.session_state.history, h_f, indent=4, ensure_ascii=False)
    except Exception as e:
        st.sidebar.error(f"Erreur historique : {e}")

def add_to_history(filename, stage, action, code):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "stage": stage,
        "action": action,
        "code": code
    }
    st.session_state.history.append(log_entry)
    save_history_to_disk()

def clean_generated_code(code_str: str) -> str:
    """Nettoie le code Markdown renvoyé par l'API OpenAI."""
    if "```python" in code_str:
        code_str = code_str.split("```python")[1].split("```")[0]
    elif "```" in code_str:
        code_str = code_str.split("```")[1].split("```")[0]
    return code_str.strip()

# --- RENDU DE LA STRUCTURE DU LAKEHOUSE DANS LA SIDEBAR ---
with st.sidebar:
    st.markdown("---")
    st.subheader("📁 Structure du Lakehouse")
    
    bronze_files = os.listdir(BRONZE_DIR)
    silver_files = os.listdir(SILVER_DIR)
    gold_files = os.listdir(GOLD_DIR)
    
    with st.expander("📦 Bronze Layer (Raw)", expanded=True):
        if not bronze_files: st.caption("Vide")
        for f in bronze_files: st.text(f"📄 {f}")
            
    with st.expander("🥈 Silver Layer (Clean)", expanded=True):
        if not silver_files: st.caption("Vide")
        for f in silver_files: st.text(f"⚙️ {f}")
            
    with st.expander("🥇 Gold Layer (KPI/Vectors)", expanded=True):
        if not gold_files: st.caption("Vide")
        for f in gold_files: st.text("🌌 FAISS Index" if f == "faiss_index" else f"📈 {f}")

    st.markdown("---")
    st.subheader("📜 Journal des modifications")
    if not st.session_state.history:
        st.caption("Aucune action enregistrée.")
    else:
        for idx, item in enumerate(reversed(st.session_state.history)):
            st.markdown(f"**{item['timestamp']}** | `{item['filename']}`")
            st.caption(f"✏️ {item['action']}")
            with st.expander("Code exécuté", expanded=False):
                st.code(item["code"], language="python")
            st.markdown("---")

# Titre principal avec la signature de Diaraye Barry
st.title("🏗 *Medallion Lakehouse Studio*")
st.caption("🚀 Conçu et développé par **Diaraye Barry** | *Data Scientist*")
st.markdown("---")

# --- VUES DU LIGNAGE DYNAMIQUE ---
st.markdown("### 📊 Lignage Dynamique du Lakehouse")
col_lin1, col_lin2, col_lin3 = st.columns(3)
with col_lin1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Couche BRONZE (Brut)</div><div class="metric-value">{len(bronze_files)} Fichiers</div></div>', unsafe_allow_html=True)
with col_lin2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Couche SILVER (Nettoyé)</div><div class="metric-value" style="color: #a78bfa;">{len(silver_files)} Fichiers</div></div>', unsafe_allow_html=True)
with col_lin3:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Couche GOLD (Décisionnel)</div><div class="metric-value" style="color: #34d399;">{len(gold_files)} Fichiers/Index</div></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📥 Ingestion & Copilote (Bronze ➔ Silver)", 
    "📈 Consolidation Métier (Silver ➔ Gold)", 
    "📊 Exploration, Profiling & Visuels (Gold)",
    "💬 Requêtage & RAG (Inférence)"
])

# =====================================================================
# TAB 1 : INGESTION & COPILOTE PANDAS OPENAI
# =====================================================================
with tab1:
    st.header("1. Ingestion de données & Nettoyage par Copilote OpenAI")
    
    load_type = st.radio(
        "Source de données :",
        ["Déposer ou sélectionner un fichier", "Recharger un fichier existant du Lakehouse"],
        horizontal=True
    )
    
    if load_type == "Déposer ou sélectionner un fichier":
        col_up, col_sel = st.columns([2, 1])
        with col_up:
            uploaded_file = st.file_uploader("Fichier (.csv ou .txt)", type=["csv", "txt"])
        with col_sel:
            bronze_available = [f for f in os.listdir(BRONZE_DIR) if f.endswith(('.csv', '.txt'))]
            selected_bronze_quick = st.selectbox("Ou sélectionnez un fichier Bronze :", ["---"] + bronze_available)
            
        file_name = None
        file_ext = None
        
        if uploaded_file:
            file_name = uploaded_file.name
            file_ext = os.path.splitext(file_name)[1].lower()
            bronze_path = os.path.join(BRONZE_DIR, file_name)
            uploaded_file.seek(0)
            
            if file_ext == ".csv":
                df_temp = pd.read_csv(uploaded_file)
                df_temp.to_csv(bronze_path, index=False)
                st.session_state.active_df = df_temp
                st.session_state.active_filename = file_name
                st.session_state.active_stage = "bronze"
            elif file_ext == ".txt":
                raw_content = uploaded_file.read().decode("utf-8")
                with open(bronze_path, "w", encoding="utf-8") as f:
                    f.write(raw_content)
                st.success(f"Document texte stocké en Bronze : `{file_name}`")
                
        elif selected_bronze_quick != "---":
            file_name = selected_bronze_quick
            file_ext = os.path.splitext(file_name)[1].lower()
            bronze_path = os.path.join(BRONZE_DIR, file_name)
            if file_ext == ".csv":
                st.session_state.active_df = pd.read_csv(bronze_path)
                st.session_state.active_filename = file_name
                st.session_state.active_stage = "bronze"
                
        if file_ext == ".txt" and file_name:
            bronze_path = os.path.join(BRONZE_DIR, file_name)
            with open(bronze_path, "r", encoding="utf-8") as f:
                raw_content = f.read()
            
            chunk_size = st.slider("Taille des chunks (caractères)", 100, 1000, 500, step=50)
            chunk_overlap = st.slider("Chevauchement (overlap)", 0, 200, 50, step=10)
            
            if st.button("⚙️ Lancer le pipeline textuel (Chunking)"):
                with st.spinner("Découpage en cours..."):
                    cleaned_content = " ".join(raw_content.split())
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    chunks = text_splitter.split_text(cleaned_content)
                    
                    silver_data = {
                        "source_file": file_name,
                        "chunks": [{"chunk_id": i, "text": chunk} for i, chunk in enumerate(chunks)]
                    }
                    silver_path = os.path.join(SILVER_DIR, f"{file_name}.json")
                    with open(silver_path, "w", encoding="utf-8") as f:
                        json.dump(silver_data, f, indent=4, ensure_ascii=False)
                    st.success(f"🥈 Chunks créés en Silver : `{silver_path}`")
                    add_to_history(file_name, "silver", f"Chunking : {len(chunks)} fragments", "RecursiveCharacterTextSplitter()")
                    st.rerun()
    else:
        col_sel_stage, col_sel_file = st.columns(2)
        with col_sel_stage:
            stage_choice = st.selectbox("Couche source :", ["bronze", "silver"])
        with col_sel_file:
            target_dir = BRONZE_DIR if stage_choice == "bronze" else SILVER_DIR
            available_files = [f for f in os.listdir(target_dir) if f.endswith(".csv")]
            selected_to_reload = st.selectbox("Sélectionnez le fichier :", available_files)
            
        if selected_to_reload:
            if st.button("🔄 Charger les données dans l'espace de travail"):
                reload_path = os.path.join(target_dir, selected_to_reload)
                st.session_state.active_df = pd.read_csv(reload_path)
                st.session_state.active_filename = selected_to_reload
                st.session_state.active_stage = stage_choice
                st.rerun()

    # Espace de travail CSV actif
    if st.session_state.active_df is not None:
        st.markdown("---")
        st.subheader(f"🛠️ Espace de travail actif : `{st.session_state.active_filename}` ({st.session_state.active_stage.upper()})")
        
        # Profiling
        with st.expander("🔍 Rapport de Qualité des Données (Data Profiling)", expanded=True):
            df_current = st.session_state.active_df
            total_rows = len(df_current)
            total_cols = len(df_current.columns)
            missing_vals = df_current.isnull().sum().sum()
            dup_rows = df_current.duplicated().sum()
            
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            p_col1.metric("Lignes", f"{total_rows}")
            p_col2.metric("Colonnes", f"{total_cols}")
            p_col3.metric("Cellules vides", f"{missing_vals}")
            p_col4.metric("Doublons", f"{dup_rows}")
            
            # Types de colonnes
            profiling_df = pd.DataFrame({
                "Type": df_current.dtypes.astype(str),
                "Manquant (%)": (df_current.isnull().sum() / total_rows * 100).round(1),
                "Uniques": df_current.nunique()
            })
            st.dataframe(profiling_df.T, use_container_width=True)

        st.markdown("#### 🧠 Consigne de modification (Ex: 'Dédoublonne les lignes', 'Filtre prix > 50')")
        user_instruction = st.text_input("Saisissez votre consigne de transformation :", key="inst_input")
        
        if user_instruction:
            if "last_instruction" not in st.session_state or st.session_state.last_instruction != user_instruction:
                with st.spinner("OpenAI génère le code de transformation optimal..."):
                    llm = ChatOpenAI(model=selected_chat_model, temperature=0)
                    schema_info = f"Colonnes : {list(st.session_state.active_df.columns)}\nTypes :\n{st.session_state.active_df.dtypes.to_string()}"
                    
                    prompt = (
                        f"Tu es un Data Engineer expert spécialisé dans Pandas.\n"
                        f"Tu devez écrire du code Python pour transformer un DataFrame nommé `df` selon l'instruction suivante.\n"
                        f"Instruction : \"{user_instruction}\"\n\n"
                        f"Schéma actuel :\n{schema_info}\n\n"
                        f"Consignes STRICTES :\n"
                        f"1. Renvoie UNIQUEMENT le code Python brut à exécuter, pas de texte de préambule, pas de blabla, juste le bloc de code.\n"
                        f"2. Ne réinitialise pas le dataframe avec pd.DataFrame().\n"
                        f"3. Modifie le dataframe directement (ex: df = df.drop_duplicates() ou df.fillna(0, inplace=True)).\n"
                    )
                    
                    response = llm.invoke(prompt)
                    st.session_state.suggested_code = clean_generated_code(response.content)
                    st.session_state.last_instruction = user_instruction
            
            st.markdown("#### 🐍 Éditeur de code (Ajustez si nécessaire)")
            edited_code = st.text_area("Code Python généré :", value=st.session_state.suggested_code, height=150)
            
            try:
                local_env = {"df": st.session_state.active_df.copy()}
                exec(edited_code, {}, local_env)
                df_after = local_env["df"]
                
                col_b, col_a = st.columns(2)
                with col_b:
                    st.markdown("**Avant**")
                    st.dataframe(st.session_state.active_df.head(5))
                with col_a:
                    st.markdown("**Après**")
                    st.dataframe(df_after.head(5))
                    
                col_act1, col_act2 = st.columns(2)
                with col_act1:
                    if st.button("🔥 Appliquer les modifications (En mémoire)", key="btn_apply_mem"):
                        st.session_state.active_df = df_after
                        add_to_history(st.session_state.active_filename, "mémoire", user_instruction, edited_code)
                        st.success("Modifications appliquées en mémoire !")
                        st.rerun()
                with col_act2:
                    if st.button("💾 Enregistrer dans la couche Silver", key="btn_save_silver"):
                        silver_name = f"{os.path.splitext(st.session_state.active_filename)[0]}_cleaned.csv"
                        silver_path = os.path.join(SILVER_DIR, silver_name)
                        df_after.to_csv(silver_path, index=False)
                        st.session_state.active_df = df_after
                        st.session_state.active_filename = silver_name
                        st.session_state.active_stage = "silver"
                        add_to_history(silver_name, "silver", f"Sauvegarde : {user_instruction}", edited_code)
                        st.success(f"Enregistré dans la couche Silver : `{silver_name}`")
                        st.rerun()
            except Exception as e:
                st.error(f"Erreur d'exécution : {e}")

# =====================================================================
# TAB 2 : CONSOLIDATION EN GOLD (CLOUD OPENAI)
# =====================================================================
with tab2:
    st.header("2. Agrégations Décisionnelles (Silver ➔ Gold)")
    
    silver_files_csv = [f for f in os.listdir(SILVER_DIR) if f.endswith(".csv")]
    silver_files_json = [f for f in os.listdir(SILVER_DIR) if f.endswith(".json")]
    all_silver_files = silver_files_csv + silver_files_json
    
    if not all_silver_files:
        st.warning("Veuillez d'abord préparer un fichier dans l'onglet 1.")
    else:
        selected_silver = st.selectbox("Sélectionnez le fichier Silver :", all_silver_files, key="gold_sel_box")
        selected_ext = os.path.splitext(selected_silver)[1].lower()
        
        if selected_ext == ".csv":
            df_silver = pd.read_csv(os.path.join(SILVER_DIR, selected_silver))
            st.dataframe(df_silver.head(5))
            
            gold_instruction = st.text_input("Saisissez l'instruction d'agrégation (ex: 'Somme de Quantite par Produit_Categorie') :")
            
            if gold_instruction:
                with st.spinner("Génération du code par OpenAI..."):
                    llm = ChatOpenAI(model=selected_chat_model, temperature=0)
                    schema_info = f"Colonnes : {list(df_silver.columns)}"
                    prompt = (
                        f"Tu es un Data Analyst expert en Pandas.\n"
                        f"Génère le code brut Python pour agréger un DataFrame nommé `df` selon la consigne : \"{gold_instruction}\"\n"
                        f"Schéma : {schema_info}\n\n"
                        f"Le code doit réassigner le résultat à `df` sous forme d'un nouveau DataFrame.\n"
                        f"Ne renvoie que le code Python brut, sans aucun balisage markdown textuel autre que le code."
                    )
                    response = llm.invoke(prompt)
                    cleaned_gold_code = clean_generated_code(response.content)
                    
                    edited_gold_code = st.text_area("Ajuster le code d'agrégation :", value=cleaned_gold_code, height=120)
                    
                    try:
                        local_env = {"df": df_silver.copy()}
                        exec(edited_gold_code, {}, local_env)
                        df_gold = local_env["df"]
                        
                        st.write("Aperçu de l'agrégation Gold :")
                        st.dataframe(df_gold)
                        
                        if st.button("🥇 Pousser dans la couche Gold (CSV)", key="btn_push_gold"):
                            gold_path = os.path.join(GOLD_DIR, f"{os.path.splitext(selected_silver)[0]}_gold.csv")
                            df_gold.to_csv(gold_path, index=False)
                            add_to_history(selected_silver, "gold", f"Agrégation : {gold_instruction}", edited_gold_code)
                            st.success(f"Table consolidée enregistrée dans `{gold_path}`")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Erreur : {e}")

        elif selected_ext == ".json":
            with open(os.path.join(SILVER_DIR, selected_silver), "r", encoding="utf-8") as f:
                silver_data = json.load(f)
                
            st.info(f"Ce document contient {len(silver_data['chunks'])} fragments textuels prêts pour l'indexation sémantique.")
            
            if st.button("🥇 Vectoriser et créer l'Index Vectoriel FAISS"):
                with st.spinner("Calcul des embeddings via OpenAI Cloud..."):
                    embeddings = OpenAIEmbeddings(model=selected_embed_model)
                    texts_to_embed = [c["text"] for c in silver_data["chunks"]]
                    
                    generated_vectors = embeddings.embed_documents(texts_to_embed)
                    metadatas = [{"source": silver_data["source_file"], "chunk_id": c["chunk_id"]} for c in silver_data["chunks"]]
                    
                    vector_store = FAISS.from_texts(texts=texts_to_embed, embedding=embeddings, metadatas=metadatas)
                    gold_index_path = os.path.join(GOLD_DIR, "faiss_index")
                    vector_store.save_local(gold_index_path)
                    
                    # Sauvegarde pour PCA 2D/3D
                    archive_data = []
                    if os.path.exists(EMBEDDINGS_ARCHIVE_PATH):
                        try:
                            with open(EMBEDDINGS_ARCHIVE_PATH, "r", encoding="utf-8") as arch_f:
                                archive_data = json.load(arch_f)
                        except Exception:
                            archive_data = []
                            
                    archive_data = [item for item in archive_data if item["source"] != silver_data["source_file"]]
                    
                    for idx, chunk in enumerate(silver_data["chunks"]):
                        archive_data.append({
                            "source": silver_data["source_file"],
                            "chunk_id": chunk["chunk_id"],
                            "text": chunk["text"],
                            "embedding": generated_vectors[idx]
                        })
                        
                    with open(EMBEDDINGS_ARCHIVE_PATH, "w", encoding="utf-8") as arch_f:
                        json.dump(archive_data, arch_f, indent=4, ensure_ascii=False)
                        
                    add_to_history(selected_silver, "gold", "Indexation vectorielle FAISS OpenAI", f"OpenAIEmbeddings({selected_embed_model})")
                    st.success("🥇 Index vectoriel FAISS généré avec succès via OpenAI !")
                    st.rerun()

# =====================================================================
# TAB 3 : EXPLORATION & PROFILING GOLD
# =====================================================================
with tab3:
    st.header("📊 Centre d'Exploration Analytique (Couche Gold)")
    col_struct, col_semantic = st.columns(2)
    
    with col_struct:
        st.subheader("📈 Décisionnel Structuré (Tables Métier)")
        csv_gold_files = [f for f in os.listdir(GOLD_DIR) if f.endswith(".csv")]
        if not csv_gold_files:
            st.info("Aucune table décisionnelle disponible en Gold.")
        else:
            selected_gold_csv = st.selectbox("Sélectionner la table Gold à visualiser :", csv_gold_files)
            df_g = pd.read_csv(os.path.join(GOLD_DIR, selected_gold_csv))
            st.dataframe(df_g)
            
            numeric_cols = df_g.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df_g.select_dtypes(include=[object, "category"]).columns.tolist()
            
            if numeric_cols and categorical_cols:
                x_axis = st.selectbox("Axe X (Catégorique) :", categorical_cols)
                y_axis = st.selectbox("Axe Y (Numérique) :", numeric_cols)
                chart_type = st.radio("Type de rendu :", ["Barres", "Lignes", "Nuage de points"], horizontal=True)
                
                if chart_type == "Barres":
                    fig = px.bar(df_g, x=x_axis, y=y_axis, template="plotly_dark", color=x_axis)
                elif chart_type == "Lignes":
                    fig = px.line(df_g, x=x_axis, y=y_axis, template="plotly_dark")
                else:
                    fig = px.scatter(df_g, x=x_axis, y=y_axis, template="plotly_dark", size=y_axis, color=x_axis)
                st.plotly_chart(fig, use_container_width=True)
                
    with col_semantic:
        st.subheader("🌌 Espace Vectoriel PCA (Sémantique)")
        if not os.path.exists(EMBEDDINGS_ARCHIVE_PATH):
            st.info("Aucun embedding textuel généré en couche Gold pour l'instant.")
        else:
            with open(EMBEDDINGS_ARCHIVE_PATH, "r", encoding="utf-8") as f:
                archive = json.load(f)
                
            if len(archive) < 2:
                st.warning("⚠️ Besoin d'au moins 2 fragments textuels indexés pour projeter l'espace sémantique.")
            else:
                embeddings_matrix = np.array([item["embedding"] for item in archive])
                sources = [item["source"] for item in archive]
                labels = [f"Chunk {item['chunk_id']}" for item in archive]
                raw_texts = [item["text"] for item in archive]
                hover_texts = [t[:120] + "..." if len(t) > 120 else t for t in raw_texts]
                
                projection_dim = st.radio("Mode de projection :", ["2D (PCA)", "3D (PCA)"], horizontal=True)
                n_components = 2 if "2D" in projection_dim else 3
                
                pca = PCA(n_components=n_components)
                reduced_vectors = pca.fit_transform(embeddings_matrix)
                
                if n_components == 2:
                    fig_pca = px.scatter(
                        x=reduced_vectors[:, 0],
                        y=reduced_vectors[:, 1],
                        color=sources,
                        hover_name=labels,
                        hover_data={"Texte": hover_texts},
                        labels={"x": "Dim 1", "y": "Dim 2"},
                        template="plotly_dark"
                    )
                else:
                    fig_pca = px.scatter_3d(
                        x=reduced_vectors[:, 0],
                        y=reduced_vectors[:, 1],
                        z=reduced_vectors[:, 2],
                        color=sources,
                        hover_name=labels,
                        hover_data={"Texte": hover_texts},
                        labels={"x": "Dim 1", "y": "Dim 2", "z": "Dim 3"},
                        template="plotly_dark"
                    )
                st.plotly_chart(fig_pca, use_container_width=True)

# =====================================================================
# TAB 4 : REQUÊTAGE & PIPELINES RAG OPENAI
# =====================================================================
with tab4:
    st.header("💬 Requêtes Sémantiques & Pipelines RAG d'Inférence")
    query_mode = st.radio("Type de données :", ["Tables Structurées (CSVs)", "Politique Interne (RAG Cloud)"], horizontal=True)
    
    if query_mode == "Tables Structurées (CSVs)":
        csv_gold_files = [f for f in os.listdir(GOLD_DIR) if f.endswith(".csv")]
        if not csv_gold_files:
            st.info("Aucune table Gold CSV disponible.")
        else:
            selected_file_q = st.selectbox("Choisir la table Gold :", csv_gold_files, key="gold_q_sel")
            df_q = pd.read_csv(os.path.join(GOLD_DIR, selected_file_q))
            st.dataframe(df_q.head(3))
            
            user_question = st.text_input("Posez votre question analytique (ex: 'Quel est le produit le plus vendu ?') :")
            
            if user_question:
                with st.spinner("OpenAI calcule la réponse..."):
                    llm = ChatOpenAI(model=selected_chat_model, temperature=0)
                    prompt = (
                        f"Tu es un analyste de données expert.\n"
                        f"Tu as accès à un DataFrame Pandas nommé `df`.\n"
                        f"Schéma : {df_q.dtypes.to_string()}\n\n"
                        f"Extrait :\n{df_q.head(5).to_string()}\n\n"
                        f"Consigne : Rédige UNE ligne de code Pandas pour répondre à la question suivante : \"{user_question}\"\n"
                        f"Enregistre IMPÉRATIVEMENT le résultat final dans une variable nommée `result`.\n"
                        f"Renvoie uniquement le code Python brut, sans explications."
                    )
                    code_resp = llm.invoke(prompt)
                    exec_code = clean_generated_code(code_resp.content)
                    
                    try:
                        local_context = {"df": df_q}
                        exec(exec_code, {}, local_context)
                        res = local_context.get("result", "Aucun résultat généré.")
                        
                        st.markdown("### 🤖 Analyse IA :")
                        st.info(f"Réponse : {res}")
                        st.code(exec_code, language="python")
                    except Exception as e:
                        st.error(f"Erreur d'évaluation : {e}")
                        
    else:
        gold_index_path = os.path.join(GOLD_DIR, "faiss_index")
        if not os.path.exists(gold_index_path):
            st.warning("Veuillez d'abord indexer un document textuel dans l'onglet 2.")
        else:
            embeddings = OpenAIEmbeddings(model=selected_embed_model)
            vector_store = FAISS.load_local(gold_index_path, embeddings, allow_dangerous_deserialization=True)
            
            col_search_param, col_search_q = st.columns([1, 3])
            with col_search_param:
                k_chunks = st.slider("Nombre de fragments (k)", 1, 5, 3)
            with col_search_q:
                user_text_query = st.text_input("Posez votre question sur les politiques de l'entreprise :")
                
            if user_text_query:
                retriever = vector_store.as_retriever(search_kwargs={"k": k_chunks})
                
                with st.spinner("Recherche dans l'index FAISS..."):
                    retrieved_docs = retriever.invoke(user_text_query)
                    context_content = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    llm = ChatOpenAI(model=selected_chat_model, temperature=0)
                    system_prompt = (
                        "Tu es un assistant RH et juridique d'entreprise. Réponds à la question de manière claire "
                        "en utilisant uniquement le contexte fourni ci-dessous.\n\n"
                        "Contexte :\n{context}"
                    )
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])
                    
                    chain = prompt_template | llm | StrOutputParser()
                    response = chain.invoke({"context": context_content, "input": user_text_query})
                    
                    st.markdown("### 🤖 Réponse RAG générée :")
                    st.success(response)
                    
                    # Lignage
                    st.markdown("---")
                    st.markdown("#### 🔍 Lignage & Traçabilité (Sémantique Gold ➔ Silver ➔ Bronze)")
                    for idx, doc in enumerate(retrieved_docs):
                        with st.expander(f"Fragment #{idx+1} | Source : {doc.metadata['source']}"):
                            st.info(doc.page_content)
                            st.caption(f"Fichier original (Bronze) : `{doc.metadata['source']}`")