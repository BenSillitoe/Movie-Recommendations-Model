# Movie Recommendation System

A collaborative filtering movie recommender built on the MovieLens 100K dataset, with a dark-themed desktop GUI for browsing recommendations and linking straight through to IMDb.

## Overview

This was my first machine learning project, built to understand collaborative filtering from the ground up — using real user rating data to recommend movies based on similarity between users' tastes, rather than relying on genre or metadata alone.

## How it works

1. **Data preparation** — the MovieLens `ratings`, `movies`, and `links` datasets are loaded and merged, with IMDb links reconstructed from `imdbId` so each recommendation can deep-link to its IMDb page.
2. **Average ratings** — a per-movie average rating is computed and merged back into the movie catalogue.
3. **User-movie matrix** — ratings are pivoted into a user × movie matrix (`userID` as rows, `movieID` as columns, ratings as values), with missing values filled using each user's own average rating.
4. **Collaborative filtering** — **cosine similarity** (`sklearn.metrics.pairwise`) is used to measure how similar users' rating patterns are to one another, which drives the recommendation logic: movies liked by similar users are surfaced as recommendations.
5. **GUI** — built with `tkinter`/`ttk`, styled as a dark, card-based interface. Selecting a movie opens its IMDb page directly via `webbrowser`.

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — 100,000 ratings from real users across thousands of movies, a standard benchmark dataset for recommendation system projects.

## Tech stack

Python · pandas · NumPy · scikit-learn (cosine similarity) · tkinter/ttk

## Running it

```bash
pip install pandas numpy scikit-learn
python main.py
```

(Rename `main.py` to whatever your actual entry-point script is called.)

Make sure `ratings.csv`, `movies.csv`, and `links.csv` from the MovieLens 100K dataset are in the same folder as the script.

## What I learned

This project was my introduction to collaborative filtering and similarity-based recommendation — the foundation that later informed more advanced portfolio projects. It also involved building a usable desktop interface around a model, rather than stopping at a notebook.

## Possible next steps

- Add item-based collaborative filtering alongside the existing user-based approach
- Experiment with matrix factorization (e.g. SVD) for comparison
- Add a search/filter bar to the GUI for larger datasets
