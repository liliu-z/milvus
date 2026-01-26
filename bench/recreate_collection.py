"""Recreate the cohere_medium_1m collection from source data."""
import time
import pyarrow.parquet as pq
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

connections.connect("default", host="localhost", port="19530")

collection_name = "cohere_medium_1m"

# Drop if exists
if utility.has_collection(collection_name):
    Collection(collection_name).drop()
    print("Dropped existing collection")

# Create schema (id + emb only, no labels needed for benchmark)
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="emb", dtype=DataType.FLOAT_VECTOR, dim=768),
]
schema = CollectionSchema(fields, description="Cohere Medium 1M")
col = Collection(collection_name, schema)
print(f"Created collection: {collection_name}")

# Load data from parquet
print("Loading train.parquet...")
table = pq.read_table("/home/ubuntu/data/cohere_medium_1m/train.parquet")
print(f"Columns: {table.column_names}, Rows: {table.num_rows}")

# Insert in batches using pandas for efficiency
batch_size = 10000
print(f"Inserting {table.num_rows} vectors in batches of {batch_size}...")
for start in range(0, table.num_rows, batch_size):
    end = min(start + batch_size, table.num_rows)
    batch = table.slice(start, end - start)
    batch_ids = batch.column("id").to_pylist()
    batch_emb = [batch.column("emb")[i].as_py() for i in range(end - start)]
    col.insert([batch_ids, batch_emb])
    print(f"  Inserted {end}/{table.num_rows}")

# Flush
print("Flushing...")
col.flush()
print(f"Flush complete. Entity count: {col.num_entities}")

# Build HNSW index
print("Building HNSW index (M=32, efConstruction=360, IP)...")
index_params = {
    "index_type": "HNSW",
    "metric_type": "IP",
    "params": {"M": 32, "efConstruction": 360}
}
col.create_index("emb", index_params)

# Wait for index build
print("Waiting for index build...")
while True:
    progress = utility.index_building_progress(collection_name)
    if progress.get("indexed_rows", 0) >= table.num_rows:
        print(f"Index built: {progress}")
        break
    print(f"  Progress: {progress}")
    time.sleep(10)

# Load collection
print("Loading collection...")
col.load()
time.sleep(10)

# Verify
segs = utility.get_query_segment_info(collection_name)
print(f"Loaded {len(segs)} segments:")
for s in segs:
    print(f"  Segment {s.segmentID}: rows={s.num_rows}, index={s.index_name}")

print("Done!")
