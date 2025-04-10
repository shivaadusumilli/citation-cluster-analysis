import json
import os
import networkx as nx
from collections import defaultdict, Counter
from community import community_louvain
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------
# 1. Load Papers
# -----------------------------
DATA_DIR = "data"
json_files = [f for f in os.listdir(DATA_DIR) if f.startswith("dblp-ref") and f.endswith(".json")]
json_files.sort()

papers = []
print("\n🔄 Loading papers...")
for filename in tqdm(json_files, desc="Files"):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        for count, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                papers.append({
                    "id": record.get("id"),
                    "year": record.get("year"),
                    "authors": record.get("authors", []),
                    "references": record.get("references", [])
                })
            except json.JSONDecodeError:
                print(f"Skipping malformed line in {filename}")
                continue
    break

print(f"\n✅ Loaded {len(papers)} papers from {len(json_files)} files.")
print("📄 Sample paper record:", papers[0])

# -----------------------------
# 2. Build Author Graph
# -----------------------------
print("\n🔄 Building author citation graph...")
paper_authors = {}
author_papers = defaultdict(list)

for paper in tqdm(papers, desc="Mapping papers to authors"):
    pid = paper["id"]
    auth_list = []
    for author in paper.get("authors", []):
        if isinstance(author, dict):
            author_id = author.get("id", author.get("name", "unknown"))
        else:
            author_id = author
        auth_list.append(author_id)
    paper_authors[pid] = auth_list
    for auth in auth_list:
        author_papers[auth].append(pid)

G = nx.DiGraph()
for author_id in author_papers:
    G.add_node(author_id)

self_cite_count = defaultdict(int)

for paper in tqdm(papers, desc="Adding citation edges"):
    citing_authors = []
    for a in paper.get("authors", []):
        if isinstance(a, dict):
            citing_authors.append(a.get("id", a.get("name", "unknown")))
        else:
            citing_authors.append(a)
    for ref_id in paper.get("references", []):
        if ref_id not in paper_authors:
            continue
        cited_authors = paper_authors[ref_id]
        for a_citing in citing_authors:
            for a_cited in cited_authors:
                if a_citing == a_cited:
                    self_cite_count[a_citing] += 1
                else:
                    if G.has_edge(a_citing, a_cited):
                        G[a_citing][a_cited]["weight"] += 1
                    else:
                        G.add_edge(a_citing, a_cited, weight=1)

print("\n✅ Graph built")
print(f"Authors with >=1 self-citation: {sum(1 for c in self_cite_count.values() if c > 0)}")

# -----------------------------
# 3. Filter Graph by Edge Weight
# -----------------------------
print("\n🔄 Filtering strong edges...")
min_edge_weight = 25
G_strong = nx.Graph()
for u, v, data in tqdm(G.to_undirected().edges(data=True), desc="Edges"):
    if data.get("weight", 1) >= min_edge_weight:
        G_strong.add_edge(u, v, weight=data["weight"])

print(f"\n✅ Filtered graph: {G_strong.number_of_nodes()} nodes, {G_strong.number_of_edges()} strong edges (≥{min_edge_weight})")

# -----------------------------
# 4. Community Detection (Louvain)
# -----------------------------
print("\n🔄 Running community detection (Louvain)...")
partition = community_louvain.best_partition(G_strong, weight='weight')
community_sizes = Counter(partition.values())
min_community_size = 2
valid_communities = {cid for cid, size in community_sizes.items() if size >= min_community_size}

filtered_authors = [a for a in G_strong.nodes() if partition[a] in valid_communities]
print(f"Detected {len(valid_communities)} valid communities with size ≥ {min_community_size}")

print("Top 5 largest tightly coupled communities:")
for comm_id, size in community_sizes.most_common(5):
    print(f"Community {comm_id}: {size} authors")

# -----------------------------
# 5. Compute Citation Bias Scores
# -----------------------------
print("\n🔄 Computing citation bias scores...")
intra_cluster_count = defaultdict(int)
for u, v, data in tqdm(G.edges(data=True), desc="Intra-cluster edges"):
    if u in partition and v in partition and partition[u] == partition[v] and partition[u] in valid_communities:
        intra_cluster_count[u] += data.get("weight", 1)

citation_bias_score = {
    author: self_cite_count[author] + intra_cluster_count[author]
    for author in filtered_authors
}

avg_bias = sum(citation_bias_score.values()) / len(citation_bias_score)
max_bias = max(citation_bias_score.values())
top_author = max(citation_bias_score, key=citation_bias_score.get)
print(f"\nAverage bias score: {avg_bias:.2f}, Max bias score = {max_bias} (Author ID: {top_author})")

# -----------------------------
# 6. Compute h-index
# -----------------------------
print("\n🔄 Calculating h-index for authors...")
citation_count_per_paper = defaultdict(int)
for paper in tqdm(papers, desc="Counting citations"):
    for ref_id in paper.get("references", []):
        if ref_id in paper_authors:
            citation_count_per_paper[ref_id] += 1

def compute_h_index(citation_counts):
    citation_counts.sort(reverse=True)
    for i, count in enumerate(citation_counts, start=1):
        if count < i:
            return i - 1
    return len(citation_counts)

h_index = {}
for author in tqdm(filtered_authors, desc="Authors"):
    counts = [citation_count_per_paper[pid] for pid in author_papers[author]]
    h_index[author] = compute_h_index(counts)

# -----------------------------
# 7. Correlation Analysis
# -----------------------------
print("\n🔄 Performing correlation analysis...")
scores = np.array([citation_bias_score[a] for a in filtered_authors])
h_values = np.array([h_index[a] for a in filtered_authors])

corr, pval = pearsonr(scores, h_values)
print(f"\n📊 Pearson correlation = {corr:.3f}, p-value = {pval:.5e}")

plt.figure(figsize=(6,4))
plt.scatter(scores, h_values, alpha=0.6, edgecolors='k')
plt.xlabel("Self + Intra-cluster Citation Count (bias score)")
plt.ylabel("Author h-index")
plt.title("Correlation of Citation Bias with h-index")
plt.axhline(y=np.mean(h_values), color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=np.mean(scores), color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("bias_vs_hindex.png")
print("\n📈 Scatter plot saved as 'bias_vs_hindex.png'")