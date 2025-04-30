import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set wide layout for dashboard
st.set_page_config(layout="wide")

st.title("Welcome to SkillScope!")  # PAGE TITLE
image = Image.open("logo.png")  # OPEN IMAGE
st.image(image, width=400)  # DISPLAY IMAGE

# ---------- HEADER ----------
st.title("Job Skill Clustering & Labor Market Insights")

# ---------- LOAD FINAL DATA ----------
df_topic_pct = pd.read_csv("final_data.csv")

# ---------- TAB SETUP ----------
tab1, tab2, tab3 = st.tabs(["📄 Project Overview", "💻 Code Breakdown", "📊 Visualizations"])

# ---------- TAB 1: PROJECT OVERVIEW ----------
with tab1:
    st.header("Project Overview")

    st.markdown("""
This project presents a full-stack NLP pipeline designed to extract, cluster, and visualize skill demand signals from unstructured job posting data. By transforming raw text into structured, interpretable outputs, the system enables real-time labor market intelligence that can support strategic workforce planning, curriculum design, and policy development.

---

### 📌 Objectives

The pipeline is built to address a key business challenge: the lack of timely, high-resolution insight into emerging roles, competencies, and job market dynamics. Specifically, it:

1. **Automates Role and Skill Extraction**
   - **Text Preprocessing:** Job descriptions are cleaned to remove HTML, punctuation, and boilerplate content.
   - **Linguistic Feature Extraction:** Using `spaCy` and `textacy`, the system extracts meaningful language units, including lemmatized tokens, noun phrases, named entities, and part-of-speech–filtered terms.
   - **Feature Vectorization:** These features are encoded into a TF-IDF matrix that captures the relative importance of skills and concepts in each job posting.

2. **Identifies Latent Labor Market Structures**
   - **Topic Modeling:** Algorithms such as Non-Negative Matrix Factorization (NMF), Truncated Singular Value Decomposition (SVD), and Latent Dirichlet Allocation (LDA) are applied to uncover latent job clusters—repeating constellations of co-occurring skills and responsibilities.
   - **Dimensionality Reduction:** These techniques reduce noise and compress the data into a smaller set of interpretable topics that represent high-level job archetypes.

3. **Generates Interactive Insights**
   - **Topic Visualization:** The most significant terms for each topic are displayed, and the proportional distribution of job postings across topics is visualized using horizontal bar charts.
   - **Interactive Exploration:** The final outputs enable users to explore the landscape of hiring trends through skill-based filters and structured visual summaries.

---

### 📊 Impact and Applications

By converting unstructured job descriptions into structured topic distributions and skill taxonomies, this system provides actionable intelligence for:

- **HR and Talent Acquisition Teams**: To benchmark job postings, assess competitive positioning, and refine role definitions.
- **Educational Institutions and Training Providers**: To align learning pathways with real-time employer demand and emerging job categories.
- **Workforce Development Agencies and Policymakers**: To identify skill gaps, prioritize funding, and direct upskilling interventions to high-growth sectors.

---

### 🧠 Conclusion

This NLP-driven approach transforms static, free-text job data into a dynamic, decision-ready representation of market demand. It delivers a scalable, automated solution to understanding how labor market needs are evolving—closing the gap between workforce capabilities and employer expectations.
    """)

# ---------- TAB 2: CODE BREAKDOWN ----------
with tab2:
    st.header("Pipeline Overview")

    st.subheader("1. Preprocessing & Cleaning")
    st.code("""
def clean(text):
    # Removes HTML, markdown, punctuation, and extra whitespace
    ...
    return cleaned_text
    """)

    st.subheader("2. NLP Feature Extraction")
    st.code("""
def extract_nlp(doc):
    return {
        'lemmas': ...,
        'noun_phrases': ...,
        'entities': ...
    }
    """)

    st.subheader("3. Feature Engineering")
    st.code("df['text_for_tfidf'] = lemmas + noun_phrases")

    st.subheader("4. Topic Modeling")
    st.code("""
from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation
nmf_model = NMF(n_components=10)
W = nmf_model.fit_transform(tfidf_matrix)
H = nmf_model.components_
    """)

    st.subheader("5. Export Outputs")
    st.code("""
doc_topic_counts = np.argmax(W, axis=1)
doc_pct = np.bincount(doc_topic_counts) / len(doc_topic_counts)
df_topic_pct = pd.DataFrame({
    "topic": topic_labels,
    "percent": doc_pct * 100
})
df_topic_pct.to_csv("topic_distribution.csv", index=False)
    """)

# ---------- TAB 3: VISUALIZATIONS ----------
with tab3:
    st.header("NMF Topic Modeling Results")

    # Pie Chart
    st.subheader("Topic Share (Pie Chart)")
    fig1, ax1 = plt.subplots()
    ax1.pie(df_topic_pct['percent'], labels=df_topic_pct['topic'], autopct='%1.1f%%', startangle=140)
    ax1.set_title("Proportion of Job Postings by Topic")
    st.pyplot(fig1)

    # Horizontal Bar Chart
    st.subheader("Topic Distribution (Bar Chart)")
    fig2, ax2 = plt.subplots()
    ax2.barh(df_topic_pct['topic'][::-1], df_topic_pct['percent'][::-1], color='teal')
    ax2.set_xlabel("Percentage")
    ax2.set_title("Distribution of Job Postings Across Topics")
    st.pyplot(fig2)

    # Line Chart (Cumulative)
    st.subheader("Cumulative Coverage of Topics")
    sorted_df = df_topic_pct.sort_values(by="percent", ascending=False).reset_index(drop=True)
    sorted_df['cumulative'] = sorted_df['percent'].cumsum()
    fig3, ax3 = plt.subplots()
    ax3.plot(sorted_df['topic'], sorted_df['cumulative'], marker='o')
    ax3.set_ylabel("Cumulative %")
    ax3.set_title("Market Coverage by Top Topics")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    # Interactive Topic Explorer
    st.subheader("Interactive Topic Explorer")
    topic_choice = st.selectbox("Select a topic to view details:", df_topic_pct['topic'])
    topic_index = df_topic_pct[df_topic_pct['topic'] == topic_choice].index[0]
    st.markdown(f"**Top keywords for {topic_choice}:**")
    st.write(topic_keywords[topic_index])

    # Data Table
    st.subheader("Topic Breakdown Table")
    st.dataframe(df_topic_pct)
    st.markdown("Each topic reflects a cluster of similar job postings defined by key terms.")
