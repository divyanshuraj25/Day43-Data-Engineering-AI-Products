# Incremental Index Updates

An incremental update embeds only new or changed chunks and adds them to the existing FAISS index. It should not recreate vectors for unchanged documents. Track source identifiers and content hashes to avoid accidental duplicate additions.
