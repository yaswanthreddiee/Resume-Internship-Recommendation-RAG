"""
Module 4: ChromaDB Vector Database Implementation
Manages persistent vector storage, cosine distance indexing,
idempotent record ingestion, deduplication, and metadata serialization.
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional, Union, Tuple
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

from src.internship_processing import (
    load_and_preprocess_internships,
    prepare_internship_embedding_text
)
from src.embedding_generator import (
    get_embedding_manager,
    generate_internship_embeddings
)

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")
DEFAULT_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "internship_collection")


def get_chroma_client(persist_directory: str = DEFAULT_PERSIST_DIR) -> chromadb.ClientAPI:
    """
    Initializes a persistent ChromaDB client using the specified directory.
    Guarantees persistence across application restarts.
    """
    os.makedirs(persist_directory, exist_ok=True)
    return chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )


def serialize_metadata_for_chroma(record: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
    """
    ChromaDB metadata supports only primitive types (str, int, float, bool).
    Converts list fields (e.g. required_skills, preferred_skills) to structured JSON/delimited strings.
    """
    meta: Dict[str, Union[str, int, float, bool]] = {}

    # String fields
    for key in [
        "internship_id", "title", "company_name", "category",
        "education_requirements", "experience_requirements",
        "location", "work_mode", "duration", "other_requirements",
        "provenance"
    ]:
        meta[key] = str(record.get(key, "") or "")

    # Serialized lists
    req_skills = record.get("required_skills", [])
    if isinstance(req_skills, list):
        meta["required_skills_csv"] = ", ".join(str(s) for s in req_skills if s)
        meta["required_skills_json"] = json.dumps(req_skills)
    else:
        meta["required_skills_csv"] = str(req_skills)
        meta["required_skills_json"] = json.dumps([])

    pref_skills = record.get("preferred_skills", [])
    if isinstance(pref_skills, list):
        meta["preferred_skills_csv"] = ", ".join(str(s) for s in pref_skills if s)
        meta["preferred_skills_json"] = json.dumps(pref_skills)
    else:
        meta["preferred_skills_csv"] = str(pref_skills)
        meta["preferred_skills_json"] = json.dumps([])

    # Boolean flags
    meta["is_synthetic"] = bool(record.get("is_synthetic", True))

    return meta


def deserialize_metadata_from_chroma(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstructs native Python dictionary from ChromaDB metadata."""
    res = dict(meta)

    # Reconstruct required_skills
    if "required_skills_json" in res:
        try:
            res["required_skills"] = json.loads(res["required_skills_json"])
        except Exception:
            res["required_skills"] = [s.strip() for s in res.get("required_skills_csv", "").split(",") if s.strip()]
    elif "required_skills_csv" in res:
        res["required_skills"] = [s.strip() for s in res["required_skills_csv"].split(",") if s.strip()]
    else:
        res["required_skills"] = []

    # Reconstruct preferred_skills
    if "preferred_skills_json" in res:
        try:
            res["preferred_skills"] = json.loads(res["preferred_skills_json"])
        except Exception:
            res["preferred_skills"] = [s.strip() for s in res.get("preferred_skills_csv", "").split(",") if s.strip()]
    elif "preferred_skills_csv" in res:
        res["preferred_skills"] = [s.strip() for s in res["preferred_skills_csv"].split(",") if s.strip()]
    else:
        res["preferred_skills"] = []

    return res


def get_or_create_collection(
    client: Optional[chromadb.ClientAPI] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIR
) -> chromadb.Collection:
    """
    Retrieves or initializes the ChromaDB collection configured with cosine distance.
    Cosine similarity metric is set explicitly via metadata={"hnsw:space": "cosine"}.
    """
    if client is None:
        client = get_chroma_client(persist_directory)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def upsert_internship_records(
    records: List[Dict[str, Any]],
    embeddings: Optional[List[List[float]]] = None,
    collection: Optional[chromadb.Collection] = None,
    client: Optional[chromadb.ClientAPI] = None
) -> int:
    """
    Inserts or updates internship records into ChromaDB.
    Avoids duplicate records by using unique internship IDs.
    """
    if not records:
        logger.warning("No records to upsert into ChromaDB.")
        return 0

    if collection is None:
        collection = get_or_create_collection(client=client)

    ids = [str(r["internship_id"]) for r in records]
    documents = [r.get("embedding_text") or prepare_internship_embedding_text(r) for r in records]
    metadatas = [serialize_metadata_for_chroma(r) for r in records]

    # If embeddings are not precomputed, generate them in batch
    if embeddings is None or len(embeddings) != len(records):
        manager = get_embedding_manager()
        embeddings = manager.encode_batch(documents, batch_size=32, normalize=True)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    logger.info(f"Upserted {len(records)} records into collection '{collection.name}'.")
    return len(records)


def init_vector_database(
    json_filepath: str = "data/internships.json",
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    force_reindex: bool = False
) -> Tuple[chromadb.Collection, int]:
    """
    Full initialization function:
    1. Connects to persistent ChromaDB.
    2. Inspects existing records in collection.
    3. Ingests data only if collection is empty or force_reindex=True.
    4. Guarantees persistence across restarts without re-embedding.
    """
    client = get_chroma_client(persist_directory)
    collection = get_or_create_collection(client, collection_name, persist_directory)

    existing_count = collection.count()
    logger.info(f"Existing count in collection '{collection_name}': {existing_count}")

    if existing_count > 0 and not force_reindex:
        logger.info(f"Vector database already populated with {existing_count} records. Reusing existing index.")
        return collection, existing_count

    if force_reindex and existing_count > 0:
        logger.info("Force reindex requested. Clearing existing records...")
        # Delete and recreate
        client.delete_collection(collection_name)
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    records = load_and_preprocess_internships(json_filepath)
    if not records:
        logger.error(f"No records loaded from {json_filepath}.")
        return collection, 0

    upserted = upsert_internship_records(records, collection=collection, client=client)
    return collection, upserted
