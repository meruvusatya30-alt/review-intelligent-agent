import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

MEMORY_FILE = "memory/topic_memory.json"
SIM_THRESHOLD = 0.75

def load_memory():
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def assign_topic(embedding, review_text):
    memory = load_memory()

    if not memory:
        memory.append({
            "topic": review_text,
            "embedding": embedding
        })
        save_memory(memory)
        return review_text

    sims = [
        cosine_similarity([embedding], [m["embedding"]])[0][0]
        for m in memory
    ]

    max_sim = max(sims)

    if max_sim > SIM_THRESHOLD:
        return memory[sims.index(max_sim)]["topic"]
    else:
        memory.append({
            "topic": review_text,
            "embedding": embedding
        })
        save_memory(memory)
        return review_text
