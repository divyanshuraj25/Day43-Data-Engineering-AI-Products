# Persisting FAISS

Save a vector index to a dedicated local directory after successful ingestion. Reload it for later queries instead of embedding every existing chunk again. Store model and corpus metadata alongside the index to prevent incompatible updates.
