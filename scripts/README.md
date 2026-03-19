# Scripts Overview

The v1 pipeline originally used `book_id` as the main unit for recommendation generation. This created duplication because the same underlying work could appear in multiple editions or duplicate records.

In the final pipeline, I switched to `work_id` so that multiple editions of the same book could be grouped together. I then selected a representative book for each work, typically the one with the highest `ratings_count`, to create cleaner recommendation outputs and more consistent Tableau visualizations.

## Final Pipeline
- `1_flatten_books_file.py`: flattens raw books data into a usable table
- `2_make_work_genres_from_shelves.py`: creates work-level genre/tag signals from shelves
- `3_build_work_books_with_tags.py`: merges work/book information with tags
- `4_preprocess_work_descriptions.py`: cleans work descriptions for NLP
- `4b_inspect_work_parquets.py`: checks intermediate parquet outputs
- `5_tfidf_cosine_neighbors_work.py`: computes TF-IDF cosine neighbors at the work level
- `6_train_lda_work.py`: trains the LDA topic model
- `7_train_nmf_work.py`: trains the NMF topic model
- `8_add_sim_lda_nmf_to_tfidf_pairs.py`: adds LDA and NMF similarity features to TF-IDF pairs
- `9_build_top50_variants.py`: builds recommendation ranking variants
- `10_build_duckdb_for_tableau_v2.py`: builds the final DuckDB database for Tableau

## v1 Pipeline
- `v1_2_make_genres_from_shelves.py`: earlier genre-building step
- `v1_3_merge_tags_into_books.py`: merges tag data into books
- `v1_4_preprocess_descriptions.py`: earlier text preprocessing
- `v1_5_cap_description_length.py`: caps long descriptions
- `v1_6_tfidf_cosine_neighbors.py`: TF-IDF cosine neighbors at the book level
- `v1_7_build_tableau_candidates_tfidf.py`: builds Tableau candidates from TF-IDF
- `v1_8_train_lda_fit_sample_transform_all.py`: earlier LDA workflow
- `v1_9_add_sim_lda_to_candidates.py`: adds LDA similarity to candidates
- `v1_10_train_nmf_topics.py`: trains NMF topics
- `v1_11_add_sim_nmf_to_candidates.py`: adds NMF similarity to candidates
- `v1_12_build_duckdb_for_tableau.py`: earlier DuckDB build for Tableau
- `v1_oldscript_build_duckdb_cdb.py`: older experimental DuckDB build
- `v1_oldscript_merge_top_genres_into_books.py`: older genre merge script
- `v1_util_check_parquet_counts.py`: utility script for parquet count checks