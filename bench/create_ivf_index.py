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

# Create IVF_FLAT index
print("Creating IVF_FLAT index (nlist=1024)...")
index_params = {
    "metric_type": "IP",
    "index_type": "IVF_FLAT",
    "params": {
        "nlist": 1024
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
