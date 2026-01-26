"""Rebuild the HNSW index as HNSW_SQ with FP16 quantization + FP32 refine."""
from pymilvus import connections, Collection, utility

connections.connect("default", host="localhost", port="19530")

collection_name = "cohere_medium_1m"
col = Collection(collection_name)

print("Releasing collection...")
col.release()

print("Dropping old index...")
col.drop_index()

print("Creating HNSW_SQ index with FP16 + FP32 refine...")
index_params = {
    "index_type": "HNSW_SQ",
    "metric_type": "IP",
    "params": {
        "M": 32,
        "efConstruction": 360,
        "sq_type": "FP16",
        "refine": True,
        "refine_type": "FP32",
    }
}
col.create_index("emb", index_params)
print("Index created. Loading collection...")
col.load()
print("Collection loaded.")

# Verify
print(f"Index: {col.indexes}")
print("Done.")
