from pymongo import MongoClient
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=False)


def get_collection(db_name: str, collection_name: str):
    db = client[db_name]
    return db[collection_name]


def insert_data(collection, data):
    try:
        result = collection.insert_one(data)
        return str(result.inserted_id)  # return ID so UI can use it
    except Exception as e:
        print(f"Insert error: {e}")
        return None


def fetch_collation(collection):
    try:
        all_docs = list(collection.find())

        if not all_docs:
            return pd.DataFrame()  # empty safe return

        df = pd.json_normalize(all_docs, sep=".")
        df = df.drop(
            columns=["_id", "collection_name", "rag_preprocess"], errors="ignore"
        )

        return df

    except Exception as e:
        print(f"Fetch error: {e}")
        return pd.DataFrame()
