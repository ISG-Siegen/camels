# IMPORTANT NOTE: This is the prototypical version of CaMeLS discussed in the PERSPECTIVES workshop paper, released for reproducibility purposes. If you want to use CaMeLS, always pull the latest version from main.

---

# CaMeLS: Cooperative Meta-Learning Service for Recommender Systems

---

Usage:
1. Configuration
   1. Install the requirements or build the Docker container. 
   2. Optional: Configure the database with the database_identifier file. 
2. Run server.py with Flask.
3. Change the server-ip connection_settings.
4. Call functions in client.py. Supply parameters if necessary. There are examples in client.py.
   1. `c_populate_database()` to fill the database with basic tables and information.
   2. `c_evaluate_algorithms()` to upload data sets and evaluation scores.
   3. `c_train_meta_learner()` to manually train a selected meta-learner.
   4. `c_predict_with_meta_learner()` to predict the best algorithm using the meta-learner.
5. To distribute for clients, package the client folder with your database_identifier.py.

---

### List of supported algorithms.

---

*Lenskit* [[1]](#1) algorithms with descriptions from their documentation:

1. UserUser: User-user nearest-neighbor collaborative filtering
2. ItemItem: Item-item nearest-neighbor collaborative filtering
3. BiasedMF: Biased matrix factorization trained with alternating least squares
4. BiasedSVD: Biased matrix factorization for implicit feedback using SciKit-Learn’s SVD solver
5. FunkSVD: Algorithm class implementing FunkSVD matrix factorization
6. Bias: A user-item bias rating prediction algorithm


*Surprise* [[2]](#2) algorithms with descriptions from their documentation:

1. NormalPredictor: Algorithm predicting a random rating based on the distribution of the training set, which is assumed to be normal
2. Baseline: Algorithm predicting the baseline estimate for given user and item
3. KNNBasic: A basic collaborative filtering algorithm
4. KNNWithMeans: A basic collaborative filtering algorithm, taking into account the mean ratings of each user
5. KNNWithZScore: A basic collaborative filtering algorithm, taking into account the z-score normalization of each user
6. KNNBaseline: A basic collaborative filtering algorithm taking into account a baseline rating
7. SVD: The famous SVD algorithm, as popularized by Simon Funk during the Netflix Prize
8. NMF: A collaborative filtering algorithm based on Non-negative Matrix Factorization
9. SlopeOne: A simple yet accurate collaborative filtering algorithm
10. CoClustering: A collaborative filtering algorithm based on co-clustering

---

### List of supported data sets. Bold data sets are included in the default database.

---

*Movielens* https://grouplens.org/datasets/movielens/

1. **Movielens 100K**
2. **Movielens 1M**
3. Movielens 10M
4. Movielens 20M
5. **Movielens Latest Small**


*Amazon* https://nijianmo.github.io/amazon/index.html

1. **amazon-all-beauty**
2. **amazon-appliances**
3. **amazon-arts-crafts-and-sewing**
4. amazon-automotive
5. amazon-books
6. amazon-cds-and-vinyl
7. amazon-cell-phones-and-accessories
8. amazon-clothing-shoes-and-jewelry
9. **amazon-digital-music**
10. amazon-electronics
11. **amazon-fashion**
12. **amazon-gift-cards**
13. amazon-grocery-and-gourmet-food
14. **amazon-industrial-and-scientific**
15. amazon-home-and-kitchen
16. amazon-kindle-store
17. **amazon-luxury-beauty**
18. **amazon-magazine-subscriptions**
19. amazon-movies-and-tv
20. **amazon-musical-instruments**
21. amazon-office-products
22. amazon-patio-lawn-and-garden
23. amazon-pet-supplies
24. **amazon-prime-pantry**
25. **amazon-software**
26. amazon-sports-and-outdoors
27. amazon-tools-and-home-improvement
28. amazon-toys-and-games
29. **amazon-video-games**

***BookCrossing*** https://grouplens.org/datasets/book-crossing/

***EachMovie*** http://www.gatsby.ucl.ac.uk/~chuwei/data/EachMovie/eachmovie.html

***Jester*** http://eigentaste.berkeley.edu/dataset/

1. **Jester3** 
2. **Jester4**

---

### List of calculated complexity measures.

---

1. Number of users
2. Number of items
3. Minimum rating
4. Maximum rating
5. Mean rating
6. Normalized mean rating
7. Number of instances
8. Highest number of rating by a single user
9. Lowest number of ratings by a single user
10. Highest number of ratings on a single item
11. Lowest number of ratings on a single item
12. Mean number of ratings by a single user
13. Mean number of ratings on a single item
14. Rating skew
15. Rating kurtosis
16. Rating standard deviation
17. Rating variance

---

## References

<a id="1">[1]</a> 
Lenskit for python: Next-generation software for recommender systems experiments
Ekstrand, Michael D

<a id="1">[2]</a> 
Surprise: A Python library for recommender systems
Hug, Nicolas

