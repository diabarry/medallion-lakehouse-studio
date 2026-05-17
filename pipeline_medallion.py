import os
import json
import uuid
from datetime import datetime
from typing import List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from openai import OpenAI

# --- INITIALISATION DU CLIENT OPENAI ---
# Le client récupère automatiquement la clé API stockée dans la variable d'environnement OPENAI_API_KEY
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError(
        "La variable d'environnement 'OPENAI_API_KEY' n'est pas configurée. "
        "Veuillez la définir (ex: export OPENAI_API_KEY='votre-cle') avant de lancer le script."
    )

client = OpenAI()

# --- MODÈLE STRUCTURÉ (SILVER) ---
class ContractMetadata(BaseModel):
    document_title: str = Field(description="Le titre ou type de document identifié (ex: Contrat de prestation, Accord de confidentialité, etc.)")
    parties: List[str] = Field(description="Liste exacte des entités ou entreprises citées signataires")
    effective_date: Optional[str] = Field(description="Date d'effet du document au format ISO YYYY-MM-DD")
    confidence_score: float = Field(description="Note d'auto-évaluation de la certitude de l'extraction, entre 0.0 et 1.0")


# --- BASES DE DONNÉES LOCALES (IN-MEMORY POUR LE POC) ---
bronze_storage: List[dict] = []
silver_chunks_storage: List[dict] = []
gold_vector_store: List[dict] = []


# =====================================================================
# ÉTAPE 1 : COUCHE BRONZE (Données Brutes)
# =====================================================================
def ingest_to_bronze(raw_content: str, file_name: str, source_type: str) -> str:
    """Ingère le document brut avec ses métadonnées système dans la couche Bronze."""
    bronze_id = str(uuid.uuid4())
    bronze_record = {
        "bronze_id": bronze_id,
        "raw_content": raw_content,
        "file_name": file_name,
        "source_type": source_type,
        "ingested_at": datetime.utcnow().isoformat()
    }
    bronze_storage.append(bronze_record)
    return bronze_id


# =====================================================================
# ÉTAPE 2 : COUCHE SILVER (Nettoyage, Structuration par LLM & Chunking)
# =====================================================================
def process_bronze_to_silver(bronze_id: str):
    """Transforme les données Bronze : nettoyage, extraction stricte par LLM et chunking."""
    # 1. Récupération depuis la couche Bronze
    record = next(item for item in bronze_storage if item["bronze_id"] == bronze_id)
    text = record["raw_content"]
    
    # 2. Nettoyage de base
    cleaned_text = " ".join(text.split())
    
    # 3. Extraction de métadonnées structurées via OpenAI GPT-4o-mini
    print("🤖 Appel à OpenAI (gpt-4o-mini) pour l'extraction structurée (Silver)...")
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": "Tu es un assistant juridique expert en extraction d'informations. Extrais les métadonnées requises au format structuré."
            },
            {"role": "user", "content": cleaned_text}
        ],
        response_format=ContractMetadata,
        temperature=0.0 # Température à 0 pour une extraction déterministe et précise
    )
    
    extracted_metadata = completion.choices[0].message.parsed
    
    # 4. Chunking logique simple (ici découpé sur les sections ou articles du texte)
    chunks = [chunk.strip() for chunk in cleaned_text.split("Article ") if chunk]
    
    # 5. Stockage en Silver
    for index, chunk_text in enumerate(chunks):
        if index == 0:
            # On conserve le préambule intact
            formatted_chunk = chunk_text
        else:
            formatted_chunk = "Article " + chunk_text
            
        silver_record = {
            "silver_id": str(uuid.uuid4()),
            "bronze_ref_id": bronze_id,
            "chunk_index": index,
            "chunk_content": formatted_chunk,
            "extracted_metadata": extracted_metadata.model_dump(),
            "processed_at": datetime.utcnow().isoformat()
        }
        silver_chunks_storage.append(silver_record)


# =====================================================================
# ÉTAPE 3 : COUCHE GOLD (Calcul des Vecteurs & Indexation)
# =====================================================================
def process_silver_to_gold():
    """Génère les embeddings réels d'OpenAI pour alimenter la couche Gold."""
    existing_gold_refs = {item["silver_ref_id"] for item in gold_vector_store}
    
    for silver_record in silver_chunks_storage:
        if silver_record["silver_id"] in existing_gold_refs:
            continue
            
        # 1. Génération de l'embedding réel (text-embedding-3-small)
        print(f"📡 Génération de l'embedding OpenAI pour le Chunk {silver_record['chunk_index']}...")
        response = client.embeddings.create(
            input=[silver_record["chunk_content"]],
            model="text-embedding-3-small"
        )
        embedding_vector = response.data[0].embedding
        
        # 2. Construction du payload Gold
        gold_record = {
            "gold_id": str(uuid.uuid4()),
            "silver_ref_id": silver_record["silver_id"],
            "vector": embedding_vector,
            "metadata_payload": {
                "text_content": silver_record["chunk_content"],
                "doc_title": silver_record["extracted_metadata"]["document_title"],
                "parties": silver_record["extracted_metadata"]["parties"],
                "bronze_origin_id": silver_record["bronze_ref_id"]
            }
        }
        gold_vector_store.append(gold_record)


# =====================================================================
# ÉTAPE D'INFÉRENCE : MOTEUR DE RECHERCHE & TRACABILITÉ
# =====================================================================
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calcule la similarité cosinus entre deux vecteurs."""
    arr_a = np.array(a)
    arr_b = np.array(b)
    return np.dot(arr_a, arr_b) / (np.linalg.norm(arr_a) * np.linalg.norm(arr_b))

def query_rag_pipeline(user_query: str):
    """Calcule l'embedding de la requête, cherche la similarité dans Gold et affiche le résultat."""
    # 1. Vectorisation de la requête utilisateur
    print(f"\n📡 Calcul de l'embedding pour la requête : '{user_query}'...")
    response = client.embeddings.create(
        input=[user_query],
        model="text-embedding-3-small"
    )
    query_vector = response.data[0].embedding
    
    # 2. Calcul des scores de similarité sur l'ensemble de la table Gold
    scores = []
    for item in gold_vector_store:
        sim = cosine_similarity(query_vector, item["vector"])
        scores.append((sim, item))
        
    # Tri par score décroissant
    scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_match = scores[0]
    
    payload = best_match["metadata_payload"]
    
    print("\n" + "="*50)
    print(f"🔍 REQUÊTE UTILISATEUR : '{user_query}'")
    print(f"📊 Score de similarité cosinus : {best_score:.4f}")
    print(f"💡 Chunk Gold retourné : {payload['text_content']}")
    print("\n🧬 LIGNAGE & TRAÇABILITÉ (Gouvernance) :")
    print(f" - Document d'origine (Bronze) : ID {payload['bronze_origin_id']}")
    print(f" - Titre du document (Silver Metadata) : {payload['doc_title']}")
    print(f" - Parties identifiées : {', '.join(payload['parties'])}")
    print("="*50 + "\n")


# =====================================================================
# EXÉCUTION DÉMO : SCÉNARIO COMPLET
# =====================================================================
if __name__ == "__main__":
    # Document de test
    raw_document_test = """CONTRAT DE PRESTATION DE SERVICE COMPTABLE. 
    Fait à Paris, le 20 mai 2026. 
    Entre la société Cabinet ChiffreClair (le Prestataire) et la startup TechVenture SAS (le Client).
    Article 1 - Objet : Le présent accord définit les conditions de tenue de la comptabilité générale de TechVenture SAS par le Cabinet ChiffreClair.
    Article 2 - Honoraires : Les prestations sont facturées sur une base forfaitaire mensuelle de 450 euros HT."""

    print("--- DÉBUT DU PIPELINE MEDALLION + OPENAI ---")
    
    # 1. Ingestion en Bronze
    print("\n📥 Étape 1 : Ingestion dans la couche Bronze...")
    bronze_id = ingest_to_bronze(
        raw_content=raw_document_test, 
        file_name="contrat_compta_2026.txt", 
        source_type="api_upload"
    )
    df_bronze = pd.DataFrame(bronze_storage)
    print(f"✅ Couche Bronze alimentée ! Fichiers bruts stockés : {len(df_bronze)}")
    print(df_bronze[["bronze_id", "file_name", "ingested_at"]].to_string(index=False))

    # 2. Transformation vers Silver
    print("\n🥈 Étape 2 : Extraction et Chunking vers la couche Silver...")
    process_bronze_to_silver(bronze_id)
    df_silver = pd.DataFrame(silver_chunks_storage)
    print(f"✅ Couche Silver alimentée ! Chunks structurés générés : {len(df_silver)}")
    print(df_silver[["silver_id", "chunk_index", "chunk_content"]].to_string(index=False))

    # 3. Vectorisation vers Gold
    print("\n🥇 Étape 3 : Embeddings et stockage dans la couche Gold...")
    process_silver_to_gold()
    df_gold = pd.DataFrame(gold_vector_store)
    print(f"✅ Couche Gold alimentée ! Vecteurs prêts : {len(df_gold)}")
    print(df_gold[["gold_id", "silver_ref_id"]].to_string(index=False))

    # 4. Inférence (RAG sur la Gold)
    print("\n💬 Étape 4 : Test de recherche d'informations sur la Gold (Inférence)...")
    query_rag_pipeline("Quels sont les tarifs ou honoraires mensuels du Cabinet ChiffreClair ?")