# bookworms-book-recommendation-system
Transparent book recommendation system using TF-IDF, LDA, NMF, DuckDB, and Tableau

# Bookworms Book Recommendation System

This repository contains my portfolio version of a book recommendation project focused on transparency, topic modeling, and interactive visualization. The goal of the project was to build a recommendation system that helps users understand why books are recommended while allowing them to explore recommendations based on ratings, themes, and text similarity.

A key design change in this project was shifting from `book_id` to `work_id` as the main unit of analysis. In the earlier version of the pipeline, recommendations were generated at the `book_id` level, which caused duplication because the same work could appear in multiple editions. In the final version, I grouped books by `work_id` and selected a representative book, typically the one with the highest `ratings_count`, to produce cleaner recommendations and more consistent Tableau outputs.

The project pipeline includes text extraction, preprocessing, TF-IDF similarity, LDA topic modeling, NMF topic modeling, recommendation generation, DuckDB export, and Tableau dashboard integration.

## Repository Structure

- `scripts/`: final pipeline scripts
- `scripts/v1/`: earlier pipeline versions and experiments
- `data/db/`: Tableau-related files
