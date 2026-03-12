# src/utils/report_visualizations.py
import base64
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Image, display

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.ml.feature import Word2VecModel, NGram, Tokenizer

def plot_centroid_neighbors(w2v_model: Word2VecModel, local_centroids: Dict[str, list], target_tags: List[str] = None) -> pd.DataFrame:
    """
    Qualitative Proof: Visualizes the nearest 'Culinary Synonyms' for specified centroids.
    """
    if not target_tags:
        target_tags = ["too_salty", "easy_quick", "moist_tender", "bland_lacks_flavor", "ingredient_issue"]
    
    neighbor_data = []
    print("--- Semantic Neighbor Audit ---")
    
    for tag in target_tags:
        if tag not in local_centroids:
            continue
            
        centroid_vec = local_centroids[tag]
        # findSynonyms validates the Subject-Modifier logic
        synonyms = w2v_model.findSynonyms(centroid_vec, 15).collect()
        words = [row['word'] for row in synonyms]
        
        neighbor_data.append({"Centroid Tag": tag, "Top 15 Neighbors": ", ".join(words)})
        print(f"{tag:<25} | {', '.join(words)}")

    return pd.DataFrame(neighbor_data)

def plot_threshold_distribution(metrics: dict) -> None:
    """
    Quantitative Proof: Visualizes the automated semantic floors derived from Algorithm 5.
    """
    thresholds = metrics.get('tag_thresholds', {})
    thresh_df = pd.DataFrame([
        {"tag": k, "threshold": v} for k, v in thresholds.items()
    ]).sort_values("threshold", ascending=False)

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    sns.barplot(data=thresh_df, x="threshold", y="tag", hue="tag", palette="viridis", legend=False)
    
    # Average line highlights the variance between specific and general tags
    mean_val = thresh_df['threshold'].mean()
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean Threshold ({mean_val:.2f})')

    plt.title("v7 Automated Semantic Thresholds (Algorithm 5)", fontsize=14)
    plt.xlabel("Cosine Similarity Score (Semantic Floor)", fontsize=12)
    plt.ylabel("Culinary Tag ID", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

architecture_graph = """
graph TD
    subgraph P1 [Phase 1: Ingestion - Bronze]
        A[RAW_recipes.csv] --> B(bronze_ingest.py)
        C[RAW_interactions.csv] --> B
        B --> D[(Bronze Parquet)]
    end

    subgraph P2 [Phase 2: Density Filter - Silver]
        D --> E1(interactions_spark.py)
        D --> E2(recipes_spark.py)
        E1 -->|15-120 Token Filter| F(text_cleaning.py)
        E2 --> F
        F --> G[(Silver Parquet)]
    end

    subgraph P3 [Phase 3: Semantic Structuring - Gold]
        G --> H(merge_spark.py)
        H -->|SHA-256 Key| I(zero_shot.py)
        I -->|65k Label Subset| J[(Gold Labeled Set)]
        H --> K(nlp_pipeline.py)
        K -->|Word2Vec Window-5| L[(Full Feature Set)]
        J & L --> M(semantic_space.py)
        M --> N[Calibrated Semantic Space]
        N --> O(inference.py)
    end

    %% Styling to prevent cutoff
    style P1 fill:#f9f9f9,stroke:#333,stroke-width:2px
    style P2 fill:#f9f9f9,stroke:#333,stroke-width:2px
    style P3 fill:#f9f9f9,stroke:#333,stroke-width:2px
"""

def render_mermaid(graph_code: str = architecture_graph) -> None:
    """
    Renders a Mermaid diagram string into a Jupyter notebook cell using the Mermaid.ink API.
    """
    graph_bytes = graph_code.encode("ascii")
    base64_bytes = base64.b64encode(graph_bytes)
    base64_string = base64_bytes.decode("ascii")
    display(Image(url="https://mermaid.ink/img/" + base64_string))


def plot_review_length_distribution(df: DataFrame) -> None:
    # Calculate token counts for cleaned reviews
    reviews_with_length = df.withColumn(
        "token_count", 
        F.size(F.split(F.col("review_clean"), " "))
    )

    # Aggregate for plotting
    length_dist = reviews_with_length.select("token_count").toPandas()

    plt.figure(figsize=(10, 6))
    sns.histplot(length_dist['token_count'], bins=50, kde=True, color='teal')
    plt.axvline(15, color='red', linestyle='--', label='Informative Threshold (15)')
    plt.title("Distribution of Review Length (Tokens)")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    print(f"Reviews meeting 15-token threshold: {reviews_with_length.filter((F.col('token_count') >= 15)).count():,}")

    # Calculate percentiles for review length
    stats = length_dist['token_count'].describe(percentiles=[.25, .5, .75, .90, .95, .99])
    print("Review Length Percentiles:\n", stats)

    # Filter out the top 5% longest reviews
    upper_limit = stats['95%']
    print(f"95th Percentile limit: {upper_limit:.2f} tokens")
    print(f"Reviews within 95th percentile: {reviews_with_length.filter(F.col('token_count') <= upper_limit).count():,}")

def plot_rating_distribution(df: DataFrame) -> None:
    rating_counts = df.groupBy("rating").count().orderBy("rating").toPandas()
    rating_counts['percent'] = (rating_counts['count'] / rating_counts['count'].sum()) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Absolute Counts
    sns.barplot(ax=ax1, x='rating', y='count', data=rating_counts, hue='rating', palette='viridis', legend=False)
    ax1.set_title("Absolute Distribution of Star Ratings")
    ax1.set_xlabel("Rating")
    ax1.set_ylabel("Review Count")

    # Percentages
    sns.barplot(ax=ax2, x='rating', y='percent', data=rating_counts, hue='rating', palette='viridis', legend=False)
    ax2.set_title("Percentage Distribution of Star Ratings")
    ax2.set_xlabel("Rating")
    ax2.set_ylabel("Percentage (%)")

    plt.tight_layout()
    plt.show()

    # Success Cluster Calculation
    success_pct = rating_counts[rating_counts['rating'] >= 4]['percent'].sum()
    print(f"Success Cluster (4-5 Stars): {success_pct:.2f}%")

def inspect_bigrams(df: DataFrame) -> None:
    # Tokenize the cleaned text
    tokenizer = Tokenizer(inputCol="review_clean", outputCol="words")
    words_df = tokenizer.transform(df)

    # Generate Bigrams
    ngram = NGram(n=2, inputCol="words", outputCol="bigrams")
    bigrams_df = ngram.transform(words_df)

    # Count top bigrams
    top_bigrams = bigrams_df.select(F.explode("bigrams").alias("bigram")) \
        .groupBy("bigram") \
        .count() \
        .orderBy(F.desc("count")) \
        .limit(10) \
        .toPandas()

    print("Top 10 Bigrams in Corpus:")
    print(top_bigrams)

def plot_tag_distribution(gold_labeled_final_df: DataFrame, tags: List[str]) -> None:
    tag_counts = []
    
    # Calculate the dynamic total count so percentages are always mathematically accurate
    total_reviews = gold_labeled_final_df.count() 
    
    for tag in tags:
        col_name = f"pred_{tag}"
        count = gold_labeled_final_df.filter(F.col(col_name) == 1).count()
        tag_counts.append({"Tag": tag, "Count": count, "Percentage": (count/total_reviews)*100})

    dist_df = pd.DataFrame(tag_counts).sort_values("Count", ascending=False)

    plt.figure(figsize=(12, 8))
    sns.set_style("white")

    ax = sns.barplot(data=dist_df, x="Count", y="Tag", palette="magma", hue="Tag", legend=False)
    plt.xscale('log') 

    plt.title("Section 6.2: Final Gold Label Distribution (Log Scale)", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Reviews (Log Scale)", fontsize=12)
    plt.ylabel("Taxonomy Tag", fontsize=12)

    for i, p in enumerate(ax.patches):
        pct = dist_df.iloc[i]['Percentage']
        ax.annotate(f'{pct:.2f}%', (p.get_width(), p.get_y() + p.get_height()/2), 
                    xytext=(5, 0), textcoords='offset points', va='center', fontsize=10)

    sns.despine()
    plt.tight_layout()
    plt.show()