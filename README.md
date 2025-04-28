# Student Roommate Preferences Clustering - University of Michigan Ann Arbor

This project focuses on clustering University of Michigan student roommate preference survey data to better understand and classify different types of roommate profiles. By applying both traditional and deep learning-based clustering methods, we aim to improve roommate matching strategies based on student living habits, social preferences, and cleanliness standards.

## Project Overview

We implemented and compared two clustering methods:
- **K-Means Clustering**: A traditional baseline method for grouping students based on their survey responses.
- **Deep Clustering Network (DCN)**: A deep learning model combining autoencoders and clustering objectives to capture complex, non-linear relationships between student preferences.

## Methods

- **Data**: 12 survey features related to student preferences for studying habits, socializing, sleep schedule, and cleanliness.
- **Baseline**: K-Means clustering with 3 clusters, evaluated using PCA visualization.
- **Deep Clustering**: Deep Clustering Network (DCN) model trained with reconstruction loss and clustering loss.
- **Evaluation**: PCA visualization of latent spaces, feature importance analysis.

## Sources
- **Documentation**: torch.nn (https://pytorch.org/docs/stable/nn.html)
- **Github**: Deep-Clustering-Network (https://github.com/xuyxu/Deep-Clustering-Network)
