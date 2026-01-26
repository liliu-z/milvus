from pymilvus import connections, Collection, utility
import time

connections.connect(host="localhost", port="19530")

collection_name = "cohere_medium_1m"
coll = Collection(collection_name)

# Release collection first
print("Releasing collection...")
coll.release()
time.sleep(2)

# Drop existing index
print("Dropping existing index...")
coll.drop_index()
time.sleep(2)

# Create HNSW_SQ index with BF16
print("Creating HNSW_SQ BF16 index with M=10...")
index_params = {
    "metric_type": "IP",
    "index_type": "HNSW_SQ",
    "params": {
        "M": 10,
        "efConstruction": 180,
        "sq_type": "BF16"
    }
}

coll.create_index("emb", index_params)
print("Index created. Waiting for build...")
utility.wait_for_index_building_complete(collection_name)
print("Index building complete.")

# Load collection
print("Loading collection...")
coll.load()
print("Collection loaded.")

connections.disconnect("default")
