# Book Recommender System – IT3190 Project

This repository contains the implementation of a hybrid book recommender system, developed as the final project for **IT3190 – Introduction to Machine Learning and Data Mining** at Hanoi University of Science and Technology (SoICT) 

## Contributors

- Trần Nguyên Chiến – 20220018 – CTTN KHMT K67  
- Trần Vương Hoàng – 20224855 – CTTN KHMT K67  
- Phạm Quang Hưng – 20220030 – CTTN KHMT K67  
- Nguyễn Đức Triệu – 20224903 – CTTN KHMT K67

## Setup

1. Clone this repository.  
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ``` 

## Data Download

Download and extract the following datasets before running any scripts 

- **Goodbooks-10k** (10 000 books + ratings):  
  https://github.com/zygmuntz/goodbooks-10k/releases  
- **Amazon Books 2014 metadata** (`meta_Books.json`) and **ratings** (`ratings_Books.csv`):  
  https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html  
- **Books XML** (extended metadata):  
  https://github.com/zygmuntz/goodbooks-10k/tree/master/books_xml  



## Data Preparation

1. Run files in **Preprocessing**  
   Clean and merge raw Goodreads and Amazon datasets to produce:  
   - `ratings_combined`: unified ratings matrix (Goodreads + Amazon)  
   - `books_dataframe`: metadata (title, description, tags, shelves) for 10 000 books 

2. Run files in **Transformation**  
   - Compute TF-IDF vectors on `description`, `tags`, and `shelves` to form three high-dimensional matrices   
   - Concatenate and apply Truncated SVD for dimensionality reduction to obtain a compact `feature_matrix` 

## Methods

We evaluate both **non–matrix-factorization** and **matrix-factorization** approaches, you can use **GridSearch** folder to find the best hyper param for each algos:

- **Non–Matrix-Factorization**:  
  - Normal Predictor  
  - Baseline Only  
  - Slope One
  - Co-clustering  

- **Matrix-Factorization**:  
  - Funk-SVD (latent factors)  
  - Non-negative Matrix Factorization (NMF) 

- **Hybrid Recommendations**:  
  – Based on a single favorite book (item similarity)  
  – Based on user latent vectors (collaborative + content hybrid) 
---

### References

1. Zygmunt Zajac. *Goodreads data from zygmuntz/goodbooks-10k repository.*  
2. Julian McAuley. *Amazon product data 2014.*  
3. PGS.TS. Thân Quang Khoát. *Bài giảng Nhập môn Học máy và Khai phá dữ liệu.*  
4. Daniel Lemire & Anna Maclachlan. *Slope One predictors for collaborative filtering*, 2007.  
5. Thomas George & Srujana Merugu. *Co-clustering framework for collaborative filtering*, 2005.  
6. Yehuda Koren, Robert Bell & Chris Volinsky. *Matrix Factorization Techniques for Recommender Systems*, 2009.  
7. Nicolas Hug. *Surprise: A Python library for recommender systems*, JOSS 5(52):2174, 2020.  
8. Xin Luo et al. *NMF-based collaborative filtering*, 2014.  

*All information synthesized from ML_Project.pdf and rawReadme.pdf.*```

