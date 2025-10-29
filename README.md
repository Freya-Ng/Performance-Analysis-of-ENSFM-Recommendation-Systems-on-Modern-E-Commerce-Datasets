# Performance Analysis of ENSFM Recommendation Systems on Modern E-Commerce Datasets

**Author:** Nguyen Linh  
**Thesis Project:** Analysis and evaluation of Efficient Non-Sampling Factorization Machines (ENSFM) on large-scale e-commerce datasets

---

## 📋 Overview

This project implements and evaluates **ENSFM (Efficient Non-Sampling Factorization Machines)** for recommendation systems on modern, large-scale e-commerce datasets. The research focuses on assessing the scalability, efficiency, and accuracy of ENSFM when applied to real-world datasets from Amazon Books and Yelp, comparing its performance against baseline datasets.

### Key Objectives
- Implement ENSFM algorithm for implicit feedback recommendation
- Evaluate performance on large-scale e-commerce datasets (Amazon Books, Yelp)
- Compare results with baseline datasets (LastFM, Frappe, MovieLens-1M)
- Analyze scalability and efficiency on modern datasets
- Optimize hyperparameters for different dataset characteristics

---

## 🗂️ Repository Structure

```
.
├── ENSFM/                      # Core ENSFM implementation
│   ├── code/                   # Main algorithm code
│   │   ├── ENSFM.py           # ENSFM model implementation
│   │   ├── ENSFM_light.py     # Lightweight version
│   │   ├── LoadData.py        # Data loading utilities
│   │   └── run.sh             # Training script
│   ├── data/                   # Baseline datasets and results
│   │   ├── frappe/            # Frappe dataset
│   │   ├── lastfm/            # LastFM dataset
│   │   └── ml-1m/             # MovieLens-1M dataset
│   └── PPT/                    # Presentation materials
│
├── Preprocessing/              # Data preprocessing scripts
│   ├── amz_preprocess.py      # Amazon Books preprocessing
│   └── yelp_preprocess.py     # Yelp dataset preprocessing
│
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── ENSFM.ipynb            # Model experiments
│   ├── hyperturningENSFM.ipynb # Hyperparameter tuning
│   ├── preprocess_amz.ipynb   # Amazon data exploration
│   ├── Preprocessing_categories.ipynb
│   └── Yelp Dataset Preprocessing for ENSFM.ipynb
│
├── data/                       # Raw datasets
│   ├── amz_dataset.zip        # Amazon Books dataset
│   └── yelp_dataset.zip       # Yelp dataset
│
└── Documents/                  # Thesis documentation
    ├── Nguyen_Linh.pdf        # Full thesis document
    ├── Thesis_Research_Plan.pdf
    └── Thesis_Theoretical_Background.pptx
```

---

## 🚀 Features

### ENSFM Implementation
- **Efficient Training:** Non-sampling approach for implicit feedback
- **TensorFlow-based:** Optimized with GPU support
- **Flexible Architecture:** Supports multi-field categorical features
- **Hyperparameter Optimization:** Automated grid search for optimal parameters

### Data Processing Pipeline
- **Multi-category Support:** Processes Amazon Books (Arts & Photography, Genre Fiction, History, Children's Books)
- **Yelp Categories:** Food, Restaurants, Shopping
- **Feature Engineering:**
  - User features: rating history, review count, helpful votes, verified purchase ratio, elite status
  - Item features: average rating, review count, price, category, location
- **Outlier Removal:** Category-specific 75th percentile thresholding
- **Train/Test Splitting:** Temporal split (latest interaction as test)
- **Category Balancing:** Optional stratified sampling across categories

---

## 📊 Datasets

### Baseline Datasets
1. **Frappe:** Mobile app usage context-aware dataset
2. **LastFM:** Music listening history
3. **MovieLens-1M:** Movie ratings

### Modern E-Commerce Datasets
1. **Amazon Books (2023):**
   - Categories: Arts & Photography, Genre Fiction, History, Children's Books
   - Features: ratings, reviews, prices, verified purchases
   - Min thresholds: 59+ item reviews, 15+ user reviews

2. **Yelp (2024):**
   - Categories: Food, Restaurants, Shopping
   - Features: business stars, reviews, location, user elite status
   - Min thresholds: Configurable per category

---

## 🛠️ Installation

### Requirements
```bash
# Python 3.7+
pip install tensorflow numpy scipy pandas scikit-learn
```

### Setup
```bash
# Clone repository
git clone https://github.com/Freya-Ng/Performance-Analysis-of-ENSFM-Recommendation-Systems-on-Modern-E-Commerce-Datasets.git
cd Performance-Analysis-of-ENSFM-Recommendation-Systems-on-Modern-E-Commerce-Datasets

# Extract datasets
unzip data/amz_dataset.zip -d data/
unzip data/yelp_dataset.zip -d data/
```

---

## 📈 Usage

### 1. Data Preprocessing

#### Amazon Books
```bash
python Preprocessing/amz_preprocess.py \
    --meta_file meta_Books.jsonl/meta_Books_processed.jsonl \
    --user_file Books.jsonl/Books_processed.jsonl \
    --min_item_reviews 59 \
    --min_user_reviews 15 \
    --mode separate  # or 'unified' for combined categories
```

#### Yelp
```bash
python Preprocessing/yelp_preprocess.py \
    --user_file yelp_user.json \
    --business_file yelp_business.json \
    --review_file yelp_review.json \
    --min_user_reviews 5 \
    --subsample_ratio 0.5 \
    --mode unified \
    --balance_categories
```

### 2. Training ENSFM

```bash
cd ENSFM/code

# Train on Frappe dataset
python ENSFM.py \
    --dataset frappe \
    --batch_size 512 \
    --epochs 501 \
    --embed_size 64 \
    --lr 0.02 \
    --negative_weight 0.01

# Train on Amazon Books
python ENSFM.py \
    --dataset "Arts & Photography" \
    --batch_size 512 \
    --epochs 501 \
    --embed_size 64 \
    --lr 0.02
```

### 3. Hyperparameter Tuning

Use the `hyperturningENSFM.ipynb` notebook to perform grid search over:
- Embedding size: [32, 64, 128]
- Learning rate: [0.001, 0.01, 0.02, 0.05]
- Negative weight: [0.001, 0.01, 0.1]
- Batch size: [256, 512, 1024]

---

## 📊 Evaluation Metrics

The model is evaluated using:
- **Recall@K:** K ∈ {5, 10, 20}
- **NDCG@K:** Normalized Discounted Cumulative Gain at K ∈ {5, 10, 20}

### Results Format
```
Epoch 0:
Recall@5: 0.3699, NDCG@5: 0.2326
Recall@10: 0.3744, NDCG@10: 0.2492
Recall@20: 0.4033, NDCG@20: 0.2814
```

---

## 🎯 Experimental Results

### Performance Comparison Across Datasets

The comprehensive evaluation reveals significant performance differences between baseline datasets and modern e-commerce datasets. Both Hit Rate (HR) and NDCG scores consistently increase as the recommendation list length K grows from 5 to 20 across all datasets.

#### Complete Evaluation Results

| Dataset | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---------|------|-------|-------|--------|---------|---------|
| **Baseline Datasets** |
| Frappe | 0.6094 | **0.7118** | 0.7889 | 0.4771 | 0.5105 | 0.5301 |
| Last.fm | 0.3683 | **0.4729** | 0.5793 | 0.2744 | 0.3082 | 0.3352 |
| MovieLens-1M | 0.0601 | **0.1024** | 0.1690 | 0.0373 | 0.0508 | 0.0674 |
| **Yelp 2018 Datasets** |
| Multi-category | 0.0294 | **0.0491** | 0.0796 | 0.0186 | 0.0250 | 0.0326 |
| Food | 0.0500 | **0.0823** | 0.1323 | 0.0312 | 0.0416 | 0.0541 |
| Restaurants | 0.0378 | **0.0628** | 0.1015 | 0.0236 | 0.0317 | 0.0414 |
| Shopping | 0.0512 | **0.0814** | 0.1218 | 0.0339 | 0.0436 | 0.0538 |
| **Amazon Books 2023 Datasets** |
| Multi-category | 0.0240 | **0.0354** | 0.0508 | 0.0163 | 0.0199 | 0.0238 |
| Genre Fiction | 0.0412 | **0.0609** | 0.0849 | 0.0281 | 0.0345 | 0.0405 |
| Children's Books | 0.0448 | **0.0628** | 0.0846 | 0.0311 | 0.0369 | 0.0424 |
| Arts & Photography | 0.0549 | **0.0744** | 0.0982 | 0.0385 | 0.0447 | 0.0508 |

**Key Observations:**
- **Baseline datasets** achieve significantly higher performance (Frappe: HR@10 = 0.7118)
- **Single-category datasets** consistently outperform multi-category variants
- **Arts & Photography** shows the best performance among Amazon categories (HR@10 = 0.0744)
- **Multi-category datasets** exhibit the lowest scores across both platforms

### Training Time Analysis

Training time varies significantly across datasets, with modern e-commerce datasets requiring substantially longer training periods.

#### Detailed Training Time (GPU: Tesla P100-PCIE-16GB)

| Dataset | Training Time | Seconds |
|---------|--------------|---------|
| **Baseline Datasets** |
| Frappe | 5m 28.5s | 328.5 |
| Last.fm | 3m 12.1s | 192.1 |
| MovieLens-1M | 12m 29s | 749.5 |
| **Yelp 2018 Datasets** |
| Multi-category | **4h 17m 43s** | **15,463** |
| Food | 43m 8s | 2,588 |
| Restaurants | 1h 47m 32s | 6,452 |
| Shopping | 10m 33s | 633 |
| **Amazon Books 2023 Datasets** |
| Multi-category | **3h 17m 51s** | **11,871** |
| Genre Fiction | 36m 7.8s | 2,167.8 |
| Children's Books | 30m 27.8s | 1,827.8 |
| Arts & Photography | 3m 21.7s | 201.7 |

**Critical Findings:**
- Multi-category datasets require **40-80x longer** training time than baseline datasets
- Training time **does not correlate with performance** (longer training ≠ better results)
- Single-category datasets offer better **time-performance trade-offs**

### Key Factors Affecting Model Performance

Through comprehensive analysis, three primary factors were identified that significantly impact ENSFM's performance on modern e-commerce datasets:

#### 1. **Data Sparsity and Scale** 🔴 *Critical Impact*
- **Challenge:** Yelp and Amazon datasets exhibit much higher sparsity compared to baseline datasets
- **Impact:** Even after preprocessing, these datasets contain:
  - Significantly more users and items
  - Much lower interaction density per user/item
  - Insufficient data for reliable preference learning
- **Result:** Directly leads to lower HR and NDCG scores

#### 2. **Feature Complexity** 🟡 *Moderate Impact*
- **Challenge:** Modern datasets possess substantially more features:
  - Diverse item attributes (ratings, reviews, prices)
  - Rich user details (purchase history, verification status)
  - Contextual information (city IDs, categories, timestamps)
- **Impact:** 
  - Increased dimensionality makes learning harder
  - Model struggles to identify predictive feature interactions
  - Contributes significantly to longer training times
- **Insight:** More features ≠ better performance; **feature selection is crucial**

#### 3. **Category Heterogeneity** 🟢 *Observable Impact*
- **Finding:** Single-category datasets consistently outperform multi-category variants
- **Explanation:**
  - Single categories create more focused recommendation tasks
  - Less sparse interaction matrices within specific domains
  - More homogeneous user behavior patterns
  - Reduced complexity compared to balanced multi-category datasets
- **Recommendation:** ENSFM performs best on **domain-specific** recommendation tasks

### Performance-Efficiency Trade-off

| Dataset Type | Performance (HR@10) | Training Time | Recommendation |
|--------------|---------------------|---------------|----------------|
| Baseline | 0.47-0.71 | 3-12 min | ✅ Excellent for small-scale |
| Single-category (Yelp) | 0.06-0.08 | 10-107 min | ⚠️ Good for domain-specific |
| Single-category (Amazon) | 0.06-0.07 | 3-36 min | ⚠️ Good for domain-specific |
| Multi-category | 0.03-0.05 | 3-4 hours | ❌ Not recommended |

---

## 🔬 Key Findings

### Performance on Baseline Datasets
- **Frappe:** Strong performance with optimized hyperparameters (HR@10: 0.7118)
- **LastFM:** Effective handling of implicit feedback (HR@10: 0.4729)
- **MovieLens-1M:** Competitive results with traditional CF methods (HR@10: 0.1024)

### Modern E-Commerce Challenges
1. **Data Sparsity:** Addressed through feature engineering and non-sampling approach
2. **Cold Start:** Multi-field features help with new users/items
3. **Scalability:** Efficient matrix operations enable large-scale training
4. **Category Imbalance:** Stratified sampling and balancing strategies

### Technical Contributions
- Adapted ENSFM for multi-category e-commerce scenarios
- Developed robust preprocessing pipeline for real-world noisy data
- Implemented category-specific outlier detection
- Created temporal train/test splits preserving recommendation realism

---

## 🔧 Model Architecture

### ENSFM Components
1. **User Embeddings:** Multi-hot encoding of user features
2. **Item Embeddings:** Multi-hot encoding of item features
3. **Feature Interactions:** Second-order feature crossing
4. **Loss Function:** Weighted non-sampling loss
   - Positive samples: (1 - weight) × squared error
   - Negative samples: weight × global approximation
5. **Optimization:** Adagrad with configurable learning rate

### Key Parameters
```python
{
    'embed_size': 64,           # Embedding dimension
    'lr': 0.02,                 # Learning rate
    'negative_weight': 0.01,    # Weight for unobserved data
    'dropout': 1.0,             # Dropout keep probability
    'batch_size': 512,          # Training batch size
    'lambda_bilinear': [0, 0]   # L2 regularization
}
```

---

## 📝 Citation

If you use this code or datasets in your research, please cite:

```bibtex
@thesis{nguyen2024ensfm,
  author = {Nguyen, Linh},
  title = {Performance Analysis of ENSFM Recommendation Systems on Modern E-Commerce Datasets},
  year = {2025},
  school = {Vaasa University of Applied Sciences}},
  type = {Bachelor's Thesis}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Acknowledgments

- Original ENSFM paper: "Efficient Non-Sampling Factorization Machines for Optimal Context-Aware Recommendation" (WWW 2020)
- Amazon Product Data (2023): Provided by McAuley Lab, UCSD
- Yelp Open Dataset (2024): Provided by Yelp Inc.

---

## 📧 Contact

For questions or collaborations:
- **Author:** Nguyen Linh
- **GitHub:** [@Freya-Ng](https://github.com/Freya-Ng)
- **Email:** anphuonglinh02062003@gmail.com

---

## 🔄 Future Work

- [ ] Extend to sequential recommendation scenarios
- [ ] Incorporate deep learning components (DeepFM, xDeepFM)
- [ ] Evaluate on additional e-commerce platforms
- [ ] Real-time recommendation deployment
- [ ] A/B testing in production environment
