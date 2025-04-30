# NLP-Driven Labor Market Intelligence

Extracting Skill Demand from Unstructured Job Postings

![CS 370 - NLP Driven Labor Market Intelligence ](https://github.com/user-attachments/assets/9b45dd91-199c-4c3a-8f38-1f5295967c6a)

Final Presentation on Canva: [Canva](https://www.canva.com/design/DAGmIaz44Wc/E_3mHa37KiWmUPhxG3Fyuw/view?utm_content=DAGmIaz44Wc&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h6d80fdf21e)

## Quick Overview:

This project uses Natural Language Processing (NLP) and unsupervised machine learning to analyze job descriptions and extract skill-based market intelligence. Using TF-IDF and topic modeling (NMF, SVD, LDA), we identify latent job clusters and high-demand skill sets from over 1,000 postings. The result: a real-time system for HR teams, job seekers, educators, and policymakers to track workforce trends and align with emerging roles.

#### Problem
- Businesses, educators, and policymakers lack up-to-date visibility into emerging roles and skillsets.

- Job seekers don’t have clear guidance on how to align their resumes with real demand.

- Most job posting analysis today is manual or relies on rigid taxonomies that miss nuance.

#### Solution
Text Preprocessing: Cleaned HTML-heavy descriptions, removed noise, and standardized content.

- Linguistic Feature Extraction: Used spaCy + textacy to extract lemmata, noun phrases, and named entities.

- Vectorization + Topic Modeling: Converted text to TF-IDF matrix, then applied NMF, SVD, and LDA to discover latent job themes and skill clusters.

- Visualization: Delivered insights through Streamlit dashboards—cluster frequencies, top skills, and distribution across job categories.

#### Business Use Cases
HR & Recruiting Teams: Benchmark job ads, identify competitive role structures, and discover talent gaps.

- Training & Higher Ed: Adapt programs to match trending skills and new job archetypes.

- Policymakers: Use topic-based insights to prioritize upskilling funds and better target workforce development.

- Job Seekers: Optimize resumes with in-demand language and understand market-aligned pathways.

#### Key Takeaways
- Topic modeling provides a scalable, real-time solution for surfacing hidden job roles and competencies.

- This approach bridges the gap between what employers ask for and what talent pipelines are producing.

- Our final dashboard enables strategic, data-backed decision-making across sectors.
