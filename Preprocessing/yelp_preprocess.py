#!/usr/bin/env python
import os
import json
import datetime
import csv
import argparse
import random
import numpy as np
from collections import defaultdict

# === Helper Functions ===

def to_str(feature_list):
    """Format each feature as an integer string and join with dashes."""
    return "-".join(str(int(x)) for x in feature_list)

def stratified_subsample_interactions(interactions_dict, subsample_ratio):
    """
    Groups users by the number of interactions and randomly samples a certain ratio 
    from each bin.
    """
    bins = defaultdict(list)
    for uid, interactions in interactions_dict.items():
        bins[len(interactions)].append(uid)
    sampled = {}
    for count, uid_list in bins.items():
        sample_size = max(1, int(len(uid_list) * subsample_ratio))
        sampled_uids = random.sample(uid_list, sample_size)
        for uid in sampled_uids:
            sampled[uid] = interactions_dict[uid]
    return sampled

# === Data Loading Functions ===

def load_users(user_file):
    """
    Loads Yelp user JSON and builds a reduced user feature vector.
    Final user feature vector:
      [ user_id, unique_useful, unique_avg_star, unique_review_count, elite_years ]
    """
    users = {}
    user_mapping = {}
    idx = 0
    print("👤 Loading user data...")
    
    with open(user_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            uid = data['user_id']

            # assign a compact index
            if uid not in user_mapping:
                user_mapping[uid] = idx
                idx += 1
            user_id_feat = user_mapping[uid] + 1

            # existing features
            avg_star = float(data.get('average_stars', 0.0))
            unique_avg_star   = int(round(avg_star * 10)) + 2000
            review_count_val  = int(data.get('review_count', 0))
            unique_review_count = review_count_val + 3000
            useful_val        = int(data.get('useful', 0))
            unique_useful     = useful_val + 1000

            # Count of elite years
            elite_list = data.get('elite') or []
            elite_years = len(elite_list) + 4000  # Add offset to elite years count
            
            users[uid] = [
                user_id_feat,
                unique_useful,
                unique_avg_star,
                unique_review_count,
                elite_years
            ]
            
    print(f"✅ Loaded {len(users)} users.")
    return users, user_mapping


def load_businesses(business_file, allowed_categories=None):
    """
    Loads Yelp business JSON and builds a reduced business feature vector.
    Final business feature vector:
      [ business_id, unique_avg_star, unique_review_count, city_id, is_open ]
    """
    if allowed_categories is None:
        allowed_categories = {"Food", "Restaurants", "Shopping"}
    
    allowed_categories_set = set(allowed_categories)
    businesses = {}
    business_mapping = {}
    city_mapping = {}
    business_categories = {}  # Track which category each business belongs to
    category_business_counts = {cat: 0 for cat in allowed_categories}
    idx = 0
    
    print("🏪 Loading business data...")
    
    # First pass to calculate thresholds
    print("📏 Calculating outlier thresholds...")
    stars_list = defaultdict(list)
    review_count_list = defaultdict(list)
    
    with open(business_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            cats = []
            if data.get('categories'):
                cats = [c.strip() for c in data['categories'].split(',')]
            
            # Find matching categories
            matching_cats = set(cats).intersection(allowed_categories_set)
            if not matching_cats:
                continue
            
            # Choose primary category (if business belongs to multiple categories)
            primary_cat = sorted(matching_cats)[0]
                
            stars_val = float(data.get('stars', 0.0))
            review_count_val = int(data.get('review_count', 0))
            
            stars_list[primary_cat].append(stars_val)
            review_count_list[primary_cat].append(review_count_val)
    
    # Calculate thresholds per category
    stars_thresholds = {}
    review_count_thresholds = {}
    
    for cat in allowed_categories:
        if stars_list[cat]:
            stars_thresholds[cat] = np.percentile(stars_list[cat], 75)
        else:
            stars_thresholds[cat] = float("inf")
            
        if review_count_list[cat]:
            review_count_thresholds[cat] = np.percentile(review_count_list[cat], 75)
        else:
            review_count_thresholds[cat] = float("inf")
            
        print(f"Category '{cat}' thresholds: stars <= {stars_thresholds[cat]:.2f}, review_count <= {review_count_thresholds[cat]}")
    
    # Second pass to build business features
    with open(business_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            bid = data['business_id']
            cats = []
            if data.get('categories'):
                cats = [c.strip() for c in data['categories'].split(',')]

            # Find matching categories
            matching_cats = set(cats).intersection(allowed_categories_set)
            if not matching_cats:
                continue

            # Choose primary category (if business belongs to multiple categories)
            primary_cat = sorted(matching_cats)[0]
            
            # Apply category-specific thresholds
            stars_val = float(data.get('stars', 0.0))
            review_count_val = int(data.get('review_count', 0))
            
            if (stars_val > stars_thresholds[primary_cat] or 
                review_count_val > review_count_thresholds[primary_cat]):
                continue
                
            # assign compact index
            if bid not in business_mapping:
                business_mapping[bid] = idx
                idx += 1
            business_id_feat = business_mapping[bid] + 1 + 4000

            # existing features
            unique_avg_star = int(round(stars_val * 10)) + 5000
            unique_review_count = review_count_val + 6000

            # city → compact ID
            city_raw = data.get('city', '').strip()
            if city_raw not in city_mapping:
                city_mapping[city_raw] = len(city_mapping) + 1
            city_id = city_mapping[city_raw] + 7000  # Add offset

            # Open flag
            is_open = int(data.get('is_open', 0)) + 8000  # Add offset
            
            businesses[bid] = [
                business_id_feat,
                unique_avg_star,
                unique_review_count,
                city_id,
                is_open
            ]
            
            business_categories[bid] = primary_cat
            category_business_counts[primary_cat] += 1
            
    # Log category business counts
    print(f"✅ Loaded {len(businesses)} businesses after filtering and outlier removal.")
    for cat, count in category_business_counts.items():
        if count > 0:
            print(f"   - '{cat}': {count} businesses")
            
    return businesses, business_mapping, business_categories


def load_reviews(review_file, users, businesses):
    """
    Loads reviews, keeping only those where both user and business are present.
    Returns a dictionary of user interactions sorted by date.
    """
    print("📝 Loading review data...")
    reviews_by_user = defaultdict(list)
    category_interaction_counts = defaultdict(int)
    
    with open(review_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            uid = data['user_id']
            bid = data['business_id']
            date_str = data.get('date', None)
            if not date_str:
                continue
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                timestamp = int(date_obj.timestamp())
            except Exception:
                continue
            if uid not in users or bid not in businesses:
                continue
            reviews_by_user[uid].append({'business_id': bid, 'date': date_obj, 'timestamp': timestamp})
    
    print(f"✅ Loaded {sum(len(reviews) for reviews in reviews_by_user.values())} review interactions for {len(reviews_by_user)} users.")
    return reviews_by_user


def frequency_filter_reviews(reviews_by_user, min_user_reviews):
    """
    Keep only users with at least min_user_reviews reviews.
    """
    filtered = {uid: reviews for uid, reviews in reviews_by_user.items() if len(reviews) >= min_user_reviews}
    print(f"✅ After filtering for min {min_user_reviews} reviews: {len(filtered)} users remain.")
    return filtered


# === Processing Functions ===

def process_category(user_file, business_file, review_file, target_category, min_user_reviews, subsample_ratio=1.0, use_random_test=False):
    """
    Process a single category dataset and create train/test splits.
    
    - Output paths for this category will be:
        <target_category>/train.csv and <target_category>/test.csv
    - If use_random_test is False, the latest interaction per user is held out as test.
      If True, one interaction is randomly selected from each user as test.
    """
    print(f"\n🔍 PROCESSING CATEGORY: '{target_category}'")
    
    # Set output paths for the category
    output_dir = target_category
    os.makedirs(output_dir, exist_ok=True)
    train_output = os.path.join(output_dir, "train.csv")
    test_output = os.path.join(output_dir, "test.csv")
    
    # Load users
    users, _ = load_users(user_file)
    
    # Load businesses filtered by target category
    businesses, _, _ = load_businesses(business_file, allowed_categories=[target_category])
    print(f"✅ Loaded {len(businesses)} businesses for category '{target_category}'.")
    
    # Load reviews for users and businesses
    reviews_by_user = load_reviews(review_file, users, businesses)
    reviews_by_user = frequency_filter_reviews(reviews_by_user, min_user_reviews)
    
    # Subsample if requested
    if subsample_ratio < 1.0:
        print(f"🔁 Subsampling user interactions (ratio = {subsample_ratio})...")
        before_subsample = len(reviews_by_user)
        reviews_by_user = stratified_subsample_interactions(reviews_by_user, subsample_ratio)
        print(f"✅ Subsampled users: {before_subsample} ➜ {len(reviews_by_user)}.")
    
    # Reindex Users
    print("🔄 Reindexing users with interactions...")
    new_user_mapping = {}
    new_users = {}
    new_user_index = 0
    for uid in reviews_by_user.keys():
        new_user_mapping[uid] = new_user_index
        old_feats = users[uid]
        # Update user ID feature with new index
        new_feats = [new_user_index + 1] + old_feats[1:]
        new_users[uid] = new_feats
        new_user_index += 1
    users_cat = new_users
    print(f"✅ Users reindexed: {len(users_cat)} users with interactions for category '{target_category}'.")

    # Reindex Businesses
    print("🔄 Reindexing businesses that appear in interactions...")
    used_business_ids = set()
    for review_list in reviews_by_user.values():
        for r in review_list:
            used_business_ids.add(r['business_id'])

    new_business_mapping = {}
    new_businesses = {}
    new_business_index = 0
    for bid in used_business_ids:
        new_business_mapping[bid] = new_business_index
        old_feats = businesses[bid]
        # Update business ID feature with new index
        new_feats = [new_business_index + 1 + 4000] + old_feats[1:] 
        new_businesses[bid] = new_feats
        new_business_index += 1
    businesses_cat = new_businesses
    print(f"✅ Businesses reindexed: {len(businesses_cat)} businesses with interactions.")

    # Split data for training and testing
    print("🔀 Splitting data for train/test...")
    train_interactions = []
    test_interactions = []
    
    for uid, review_list in reviews_by_user.items():
        if len(review_list) < 2:
            continue
        
        # Sort by timestamp
        review_list.sort(key=lambda x: x['timestamp'])
        
        # Choose test instance based on splitting rule
        if use_random_test:
            chosen_index = random.randint(0, len(review_list) - 1)
            test_review = review_list[chosen_index]
            train_reviews = [r for i, r in enumerate(review_list) if i != chosen_index]
        else:
            # Default: use the most recent (last after sorting) interaction as test
            test_review = review_list[-1]
            train_reviews = review_list[:-1]
        
        for r in train_reviews:
            train_interactions.append({'user_id': uid, 'business_id': r['business_id']})
        test_interactions.append({'user_id': uid, 'business_id': test_review['business_id']})
    
    print(f"✅ Split data: {len(train_interactions)} train interactions, {len(test_interactions)} test interactions.")

    # Write the train/test CSV files
    print("📤 Writing train/test CSV files...")
    with open(train_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for inter in train_interactions:
            uid = inter['user_id']
            bid = inter['business_id']
            writer.writerow([to_str(users_cat[uid]), to_str(businesses_cat[bid])])
    
    with open(test_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for inter in test_interactions:
            uid = inter['user_id']
            bid = inter['business_id']
            writer.writerow([to_str(users_cat[uid]), to_str(businesses_cat[bid])])
    
    print(f"💾 Train file written to: {train_output}")
    print(f"💾 Test file written to: {test_output}")
    print(f"✅ Dataset for '{target_category}' processed successfully.")


def process_unified(user_file, business_file, review_file, target_categories, min_user_reviews, output_dir, subsample_ratio=1.0, use_random_test=False, balance_categories=True):
    """
    Process a unified dataset from multiple target categories and create train/test splits.
    
    The output paths will be:
        <output_dir>/train.csv and <output_dir>/test.csv
        
    If use_random_test is False, the latest interaction per user is held out as test.
    If True, one interaction is randomly selected per user as test.
    
    If balance_categories is True, we'll attempt to balance the dataset across different sized categories.
    """
    print(f"\n🔍 PROCESSING UNIFIED CATEGORIES: {target_categories}")
    
    # Set output paths for the unified dataset
    os.makedirs(output_dir, exist_ok=True)
    train_output = os.path.join(output_dir, "train.csv")
    test_output = os.path.join(output_dir, "test.csv")
    
    # Load users
    users, _ = load_users(user_file)
    
    # Load businesses filtered by target categories
    businesses, _, business_categories = load_businesses(business_file, allowed_categories=target_categories)
    
    # Generate category ID mapping (for embedding)
    cat_to_id = {cat: idx+1 for idx, cat in enumerate(target_categories)}
    
    # Add category ID as a feature to business vectors
    for bid, features in businesses.items():
        if bid in business_categories:
            category_feat = cat_to_id[business_categories[bid]] + 9000  # Add offset
            businesses[bid] = features + [category_feat]  # Append category ID feature
    
    # Load reviews for users and businesses
    reviews_by_user = load_reviews(review_file, users, businesses)
    reviews_by_user = frequency_filter_reviews(reviews_by_user, min_user_reviews)
    
    # Track category interaction counts
    category_interaction_counts = defaultdict(int)
    for uid, reviews in reviews_by_user.items():
        for review in reviews:
            bid = review['business_id']
            if bid in business_categories:
                category_interaction_counts[business_categories[bid]] += 1
    
    print("Category interaction counts before processing:")
    for cat, count in category_interaction_counts.items():
        print(f"   - '{cat}': {count} interactions")
    
    # Balance dataset across categories if requested
    if balance_categories and len(target_categories) > 1:
        print("⚖️ Balancing interactions across categories...")
        # Calculate target interaction count per category
        # Strategy: Up-sample smaller categories
        max_category_count = max(category_interaction_counts.values())
        category_sampling_rates = {cat: max_category_count / count if count > 0 else 0 
                                for cat, count in category_interaction_counts.items()}
        
        print("Category sampling rates for balancing:")
        for cat, rate in category_sampling_rates.items():
            print(f"   - '{cat}': {rate:.2f}x")
        
        # Apply category-based sampling to user interactions
        balanced_interactions = defaultdict(list)
        for uid, reviews in reviews_by_user.items():
            for review in reviews:
                bid = review['business_id']
                if bid in business_categories:
                    cat = business_categories[bid]
                    # Probabilistic up-sampling for smaller categories
                    if random.random() < category_sampling_rates[cat]:
                        balanced_interactions[uid].append(review)
            
        # Replace original interactions with balanced ones
        # but only if the user still has interactions
        reviews_by_user = {uid: reviews for uid, reviews in balanced_interactions.items() 
                           if len(reviews) >= 2}
        
        # Recalculate category interaction counts after balancing
        new_category_counts = defaultdict(int)
        for uid, reviews in reviews_by_user.items():
            for review in reviews:
                bid = review['business_id']
                if bid in business_categories:
                    new_category_counts[business_categories[bid]] += 1
        
        print("Category counts after balancing:")
        for cat, count in new_category_counts.items():
            print(f"   - '{cat}': {count} interactions")
    
    # Apply general subsampling if requested
    if subsample_ratio < 1.0:
        print(f"🔁 Subsampling user interactions (ratio = {subsample_ratio})...")
        before_subsample = len(reviews_by_user)
        reviews_by_user = stratified_subsample_interactions(reviews_by_user, subsample_ratio)
        print(f"✅ Subsampled users: {before_subsample} ➜ {len(reviews_by_user)}.")

    # Global reindexing for users with interactions
    print("🔄 Reindexing users with interactions...")
    new_users = {}
    new_user_index = 0
    for uid in reviews_by_user.keys():
        old_feats = users[uid]
        # Replace original user ID with new global index (starting from 1)
        new_feats = [new_user_index + 1] + old_feats[1:]
        new_users[uid] = new_feats
        new_user_index += 1
    users = new_users
    print(f"✅ Users reindexed: {len(users)} users with interactions.")

    # Global reindexing for businesses that appear in interactions
    print("🔄 Reindexing businesses that appear in interactions...")
    used_business_ids = set()
    for reviews in reviews_by_user.values():
        for review in reviews:
            used_business_ids.add(review['business_id'])
    
    new_businesses = {}
    new_business_index = 0
    for bid in used_business_ids:
        old_feats = businesses[bid]
        # Update business ID with new global index (starting from 1)
        new_feats = [new_business_index + 1 + 4000] + old_feats[1:]
        new_businesses[bid] = new_feats
        new_business_index += 1
    businesses = new_businesses
    print(f"✅ Businesses reindexed: {len(businesses)} businesses used in interactions.")

    # Split data for training and testing
    print("🔀 Splitting data for train/test...")
    train_interactions = []
    test_interactions = []
    
    for uid, review_list in reviews_by_user.items():
        if len(review_list) < 2:
            continue
        
        # Sort by timestamp
        review_list.sort(key=lambda x: x['timestamp'])
        
        # Choose test instance based on splitting rule
        if use_random_test:
            chosen_index = random.randint(0, len(review_list) - 1)
            test_review = review_list[chosen_index]
            train_reviews = [r for i, r in enumerate(review_list) if i != chosen_index]
        else:
            # Default: use the most recent (last after sorting) interaction as test
            test_review = review_list[-1]
            train_reviews = review_list[:-1]
        
        for r in train_reviews:
            train_interactions.append({'user_id': uid, 'business_id': r['business_id']})
        test_interactions.append({'user_id': uid, 'business_id': test_review['business_id']})
    
    print(f"✅ Split data: {len(train_interactions)} train interactions, {len(test_interactions)} test interactions.")

    # Write the train/test CSV files
    print("📤 Writing train/test CSV files...")
    with open(train_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for inter in train_interactions:
            uid = inter['user_id']
            bid = inter['business_id']
            writer.writerow([to_str(users[uid]), to_str(businesses[bid])])
    
    with open(test_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for inter in test_interactions:
            uid = inter['user_id']
            bid = inter['business_id']
            writer.writerow([to_str(users[uid]), to_str(businesses[bid])])
    
    print(f"💾 Train file written to: {train_output}")
    print(f"💾 Test file written to: {test_output}")
    print("✅ Unified dataset processed successfully.")


# === Main Function ===

def main():
    parser = argparse.ArgumentParser(
        description="Generate train.csv and test.csv files with features for recommendation across multiple categories."
    )
    parser.add_argument("--user_file", type=str, default="yelp_user.json", help="Path to Yelp user JSON file")
    parser.add_argument("--business_file", type=str, default="yelp_business.json", help="Path to Yelp business JSON file")
    parser.add_argument("--review_file", type=str, default="yelp_review.json", help="Path to Yelp review JSON file")
    parser.add_argument("--min_user_reviews", type=int, default=5, help="Minimum reviews per user to keep")
    parser.add_argument("--subsample_ratio", type=float, default=0.5, help="Subsample ratio for users (0-1)")
    parser.add_argument("--mode", type=str, choices=['separate', 'unified'], default='separate',
                        help="Processing mode: 'separate' for individual categories or 'unified' for combined categories")
    parser.add_argument("--unified_output", type=str, default=r"d:\Project\CARS\Yelp JSON\yelp_dataset\F_yelp",
                        help="Output directory for unified dataset (when mode is 'unified')")
    parser.add_argument("--use_random_test", action="store_true",
                        help="If set, randomly select one test instance per user")
    parser.add_argument("--balance_categories", action="store_true",
                        help="Balance dataset across categories for unified mode")
    args = parser.parse_args()

    # Categories to process
    categories = ["Food", "Restaurants", "Shopping"]
    
    if args.mode == 'separate':
        # Process each category separately
        print("🔄 Processing categories separately...")
        for category in categories:
            process_category(
                user_file=args.user_file,
                business_file=args.business_file,
                review_file=args.review_file,
                target_category=category,
                min_user_reviews=args.min_user_reviews,
                subsample_ratio=args.subsample_ratio,
                use_random_test=args.use_random_test
            )
    else:
        # Process all categories together in a unified dataset
        print("🔄 Processing categories in unified mode...")
        process_unified(
            user_file=args.user_file,
            business_file=args.business_file,
            review_file=args.review_file,
            target_categories=categories,
            min_user_reviews=args.min_user_reviews,
            output_dir=args.unified_output,
            subsample_ratio=args.subsample_ratio,
            use_random_test=args.use_random_test,
            balance_categories=args.balance_categories
        )

if __name__ == "__main__":
    main()