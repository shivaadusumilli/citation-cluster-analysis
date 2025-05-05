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
# 3A. Filter Graph by Self-Citation Ratio
# -----------------------------
print("\n🔄 Filtering graph by self-citation ratio...")
total_citations_by_author = defaultdict(int)
for u, v, data in G.edges(data=True):
    total_citations_by_author[u] += data.get("weight", 1)

# Compute average citations per paper for each author
avg_citations = {
    author: total_citations_by_author[author] / max(1, len(author_papers[author]))
    for author in G.nodes()
}

self_cite_ratio = {
    author: self_cite_count[author] / max(1, total_citations_by_author[author])
    for author in G.nodes()
}

min_self_cite_ratio = 0.25  # include more authors
G_strong = nx.Graph()
for u, v, data in G.to_undirected().edges(data=True):
    if self_cite_ratio.get(u, 0) >= min_self_cite_ratio or self_cite_ratio.get(v, 0) >= min_self_cite_ratio:
        G_strong.add_edge(u, v, weight=data["weight"])

print(f"\n✅ Filtered graph by ratio: {G_strong.number_of_nodes()} nodes, {G_strong.number_of_edges()} edges (self-cite ratio ≥ {min_self_cite_ratio})")

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

# -----------------------------------------
# 4B. Greedy Modularity Community Detection
# -----------------------------------------

# Reduce graph size for Greedy Modularity if too large
max_nodes_greedy = 10000
if G_strong.number_of_nodes() > max_nodes_greedy:
    top_nodes = sorted(G_strong.degree, key=lambda x: x[1], reverse=True)[:max_nodes_greedy]
    G_greedy = G_strong.subgraph([n for n, _ in top_nodes]).copy()
    print(f"\n⚠️ Reduced graph size for Greedy Modularity to {G_greedy.number_of_nodes()} nodes")
else:
    G_greedy = G_strong

print("\n🔄 Running community detection (Greedy Modularity)...")
greedy_communities = list(nx.algorithms.community.greedy_modularity_communities(G_greedy, weight='weight'))
greedy_partition = {author: i for i, com in enumerate(greedy_communities) for author in com}
filtered_authors_greedy = list(greedy_partition.keys())
greedy_sizes = [len(c) for c in greedy_communities]
print(f"Detected {len(greedy_communities)} communities")
print("Top 5 largest tightly coupled communities (Greedy Modularity):")
for i, size in enumerate(sorted(greedy_sizes, reverse=True)[:5]):
    print(f"Community {i+1}: {size} authors")

# -----------------------------------------
# 4C. Label Propagation Community Detection
# -----------------------------------------

print("\n🔄 Running community detection (Label Propagation)...")
label_communities = list(nx.algorithms.community.label_propagation_communities(G_strong))
label_partition = {author: i for i, com in enumerate(label_communities) for author in com}
filtered_authors_label = list(label_partition.keys())
label_sizes = [len(c) for c in label_communities]
print(f"Detected {len(label_communities)} communities")
print("Top 5 largest tightly coupled communities (Label Propagation):")
for i, size in enumerate(sorted(label_sizes, reverse=True)[:5]):
    print(f"Community {i+1}: {size} authors")

# -----------------------------
# 5. Compute Citation Bias Scores
# -----------------------------
print("\n🔄 Computing citation bias scores...")
intra_cluster_count = defaultdict(int)
for u, v, data in tqdm(G.edges(data=True), desc="Intra-cluster edges"):
    if u in partition and v in partition and partition[u] == partition[v] and partition[u] in valid_communities:
        intra_cluster_count[u] += data.get("weight", 1)

alpha = 0.8  # heavier weight on self-citation ratio
beta = 0.2  # lighter weight on intra-cluster citation ratio

bias_score = {
    author: alpha * self_cite_ratio.get(author, 0) + beta * (intra_cluster_count.get(author, 0) / max(1, avg_citations.get(author, 1)))
    for author in filtered_authors
}

avg_bias = sum(bias_score.values()) / len(bias_score)
max_bias = max(bias_score.values())
top_author = max(bias_score, key=bias_score.get)
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
scores = np.array([bias_score[a] for a in filtered_authors])
h_values = np.array([h_index[a] for a in filtered_authors])

corr, pval = pearsonr(scores, h_values)
print(f"\n📊 Pearson correlation = {corr:.3f}, p-value = {pval:.5e}")


# -----------------------------
# 7B. Community Evaluation Metrics
# -----------------------------
from networkx.algorithms.community.quality import modularity

def coverage(G, communities):
    intra_edges = 0
    total_edges = G.number_of_edges()
    node_community = {}
    for i, com in enumerate(communities):
        for node in com:
            node_community[node] = i
    for u, v in G.edges():
        if node_community.get(u) == node_community.get(v):
            intra_edges += 1
    return intra_edges / total_edges if total_edges > 0 else 0

def performance(G, communities):
    correct = 0
    total = 0
    node_community = {}
    for i, com in enumerate(communities):
        for node in com:
            node_community[node] = i
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            u, v = nodes[i], nodes[j]
            same_com = node_community.get(u) == node_community.get(v)
            connected = G.has_edge(u, v)
            if (same_com and connected) or (not same_com and not connected):
                correct += 1
            total += 1
    return correct / total if total > 0 else 0

# print("\nEvaluating community structure quality...")

# louvain_comms = [set([n for n in partition if partition[n] == cid]) for cid in set(partition.values())]
# mod_louvain = modularity(G_strong, louvain_comms)
# cov_louvain = coverage(G_strong, louvain_comms)
# perf_louvain = performance(G_strong, louvain_comms)
# print(f"Louvain: Modularity={mod_louvain:.4f}, Coverage={cov_louvain:.4f}, Performance={perf_louvain:.4f}")

# greedy_comms = [set(c) for c in greedy_communities]
# mod_greedy = modularity(G_strong, greedy_comms)
# cov_greedy = coverage(G_strong, greedy_comms)
# perf_greedy = performance(G_strong, greedy_comms)
# print(f"Greedy Modularity: Modularity={mod_greedy:.4f}, Coverage={cov_greedy:.4f}, Performance={perf_greedy:.4f}")

# label_comms = [set(c) for c in label_communities]
# mod_label = modularity(G_strong, label_comms)
# cov_label = coverage(G_strong, label_comms)
# perf_label = performance(G_strong, label_comms)
# print(f"Label Propagation: Modularity={mod_label:.4f}, Coverage={cov_label:.4f}, Performance={perf_label:.4f}")


# -----------------------------
# 8. Scatter Plots for All Algorithms
# -----------------------------
def plot_bias_vs_hindex(authors, bias_scores, h_indices, method_name):
    scores = np.array([bias_scores[a] for a in authors]) * 100  # scale up
    h_vals = np.array([h_indices[a] for a in authors])
    corr, pval = pearsonr(scores, h_vals)
    plt.figure(figsize=(6,4))
    plt.scatter(scores, h_vals, alpha=0.6, edgecolors='k')
    plt.xlabel("Self + Intra-cluster Citation Count (bias score)")
    plt.ylabel("Author h-index")
    plt.title(f"{method_name} (r={corr:.2f})")
    plt.axhline(y=np.mean(h_vals), color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=np.mean(scores), color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"bias_vs_hindex_{method_name.lower().replace(' ', '_')}.png")
    print(f"📈 Scatter plot saved as 'bias_vs_hindex_{method_name.lower().replace(' ', '_')}.png'")

# Plot for Louvain
plot_bias_vs_hindex(filtered_authors, bias_score, h_index, "Louvain")

# Plot for Greedy Modularity
# Compute Greedy intra-cluster citations
greedy_intra = defaultdict(int)
for u, v, data in G.edges(data=True):
    if u in greedy_partition and v in greedy_partition and greedy_partition[u] == greedy_partition[v]:
        greedy_intra[u] += data.get("weight", 1)
greedy_bias = {
    a: alpha * self_cite_ratio.get(a, 0) + beta * (greedy_intra.get(a, 0) / max(1, avg_citations.get(a, 1)))
    for a in filtered_authors_greedy
}
greedy_hindex = {}
for author in filtered_authors_greedy:
    counts = [citation_count_per_paper[pid] for pid in author_papers[author]]
    greedy_hindex[author] = compute_h_index(counts)
plot_bias_vs_hindex(filtered_authors_greedy, greedy_bias, greedy_hindex, "Greedy Modularity")

# Plot for Label Propagation
# Compute Label Propagation intra-cluster citations
label_intra = defaultdict(int)
for u, v, data in G.edges(data=True):
    if u in label_partition and v in label_partition and label_partition[u] == label_partition[v]:
        label_intra[u] += data.get("weight", 1)
label_bias = {
    a: alpha * self_cite_ratio.get(a, 0) + beta * (label_intra.get(a, 0) / max(1, avg_citations.get(a, 1)))
    for a in filtered_authors_label
}
label_hindex = {}
for author in filtered_authors_label:
    counts = [citation_count_per_paper[pid] for pid in author_papers[author]]
    label_hindex[author] = compute_h_index(counts)
plot_bias_vs_hindex(filtered_authors_label, label_bias, label_hindex, "Label Propagation")