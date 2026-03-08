import base64
from IPython.display import Image, display

architecture_graph = """
graph TD
    subgraph P1 [Phase 1: Ingestion - Bronze]
        A[RAW_recipes.csv] --> B(io_spark.py)
        C[RAW_interactions.csv] --> B
        B --> D[(Bronze Parquet)]
    end

    subgraph P2 [Phase 2: Density Filter - Silver]
        D --> E(interactions_spark.py)
        E -->|15-120 Token Filter| F(text_cleaning.py)
        F --> G[(Silver Interactions)]
    end

    subgraph P3 [Phase 3: Semantic Structuring - Gold]
        G --> H(merge_spark.py)
        H -->|SHA-256 Key| I(zero_shot.py)
        I -->|65k Label Subset| J[(Gold Labeled Set)]
        G --> K(build_features.py)
        K -->|Word2Vec Window-3| L[(Full Feature Set)]
        J & L --> M(thresholds.py)
        M --> N[Calibrated Semantic Space]
    end

    %% Styling to prevent cutoff
    style P1 fill:#f9f9f9,stroke:#333,stroke-width:2px
    style P2 fill:#f9f9f9,stroke:#333,stroke-width:2px
    style P3 fill:#f9f9f9,stroke:#333,stroke-width:2px
"""

def render_mermaid(graph_code: str = architecture_graph):
    """
    Renders a Mermaid diagram string into a Jupyter notebook cell using the Mermaid.ink API.
    """
    graph_bytes = graph_code.encode("ascii")
    base64_bytes = base64.b64encode(graph_bytes)
    base64_string = base64_bytes.decode("ascii")
    display(Image(url="https://mermaid.ink/img/" + base64_string))

