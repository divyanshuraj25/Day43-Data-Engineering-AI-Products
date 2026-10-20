# API Rate Limits

Embedding APIs can return rate-limit or temporary server errors. Use bounded exponential backoff with jitter, respect retry-after headers when available, and report failures clearly rather than silently skipping documents.
