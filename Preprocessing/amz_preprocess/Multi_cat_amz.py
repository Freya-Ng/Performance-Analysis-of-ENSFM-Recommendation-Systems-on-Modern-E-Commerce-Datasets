#!/usr/bin/env python
import os
import json
import csv
from collections import defaultdict
import numpy as np
import random
import argparse

# === Feature Binning Functions ===
def bin_avg_rating(avg):
    if avg < 3.5:
        return 63000
    elif avg < 4.2:
        return 63001
    elif avg < 4.5:
        return 63002
    elif avg < 4.8:
        return 63003
    else:
        return 3004

def bin_rating_count(cnt):
    if cnt < 10:
        return 63010
    elif cnt < 70:
        return 63011
    elif cnt < 500:
        return 63012
    elif cnt < 2000:
        return 63013
    else:
        return 3014

def bin_price(price):
    if price < 9:
        return 63020
    elif price < 15:
        return 63021
    elif price < 25:
        return 63022
    elif price < 70:
        return 63023
    else:
        return 63024

def bin_helpful_vote(v):
    if v == 0:
        return 61120
    elif v < 5:
        return 61121
    elif v < 20:
        return 61122
    else:
        return 61123

def bin_user_avg_rating(avg):
    if avg < 3.5:
        return 61100
    elif avg < 4.1:
        return 61101
    elif avg < 4.75:
        return 61102
    else:
        return 61103

def bin_user_total_reviews(cnt):
    if cnt < 5:
        return 61110
    elif cnt < 20:
        return 61111
    elif cnt < 100:
        return 61112
    elif cnt < 500:
        return 61113
    else:
        return 61114

def bin_user_verified_purchase_ratio(ratio):
    if ratio < 0.35:
        return 61130
    elif ratio < 0.75:
        return 61131
    else:
        return 61132

def to_str(x):
    """Format each feature as an integer string and join with dashes."""
    return '-'.join(str(int(i)) for i in x)

def stream_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def safe_parse_price(price):
    try:
        return float(price)
    except (ValueError, TypeError):
        return 0

def stratified_subsample_interactions(interactions_dict, subsample_ratio):
    """
    Groups users by the number of interactions and randomly samples a certain ratio 
    (subsample_ratio) from each bin.
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

def process_unified(meta_file, user_file, target_categories, min_item_reviews, min_user_reviews, output_dir, subsample_ratio=1.0, use_random_test=False):
    """Process a unified dataset from multiple target categories and create train/test splits.
    
    The output paths will be:
        <output_dir>/train.csv and <output_dir>/test.csv
        
    If use_random_test is False, the latest (most recent) interaction per user is held out as test
    (after sorting by timestamp). Otherwise, one interaction is randomly selected per user.
    """
    print(f"\n🔍 PROCESSING UNIFIED CATEGORIES: {target_categories}")
    
    # Set output paths for the unified dataset
    os.makedirs(output_dir, exist_ok=True)
    train_output = os.path.join(output_dir, "train.csv")
    test_output = os.path.join(output_dir, "test.csv")
    
    # Calculate outlier thresholds based on the meta_file
    print("📏 Calculating outlier thresholds...")
    rating_numbers_list = []
    prices_list = []
    target_categories = set(target_categories)  # ensure it's a set for membership tests
    for row in stream_jsonl(meta_file):
        raw_cats = row.get("categories", "")
        if not raw_cats:
            continue
        
        # Handle both list and comma-separated strings
        if isinstance(raw_cats, list):
            cats = [cat.strip() for cat in raw_cats if cat.strip()]
        else:
            cats = [cat.strip() for cat in raw_cats.split(',') if cat.strip()]
        
        # Check if any of the target categories is present
        if not set(cats).intersection(target_categories):
            continue
        
        try:
            rn = float(row.get("rating_number", 0))
            if rn < min_item_reviews:
                continue
            rating_numbers_list.append(rn)
        except:
            continue
        
        pr = safe_parse_price(row.get("price"))
        prices_list.append(pr)
    
    rating_number_threshold = np.percentile(rating_numbers_list, 75) if rating_numbers_list else float("inf")
    price_threshold = np.percentile(prices_list, 75) if prices_list else float("inf")
    print(f"Outlier thresholds: rating_number <= {rating_number_threshold:.2f}, price <= {price_threshold:.2f}")

    # Build item features for valid items in any of the target categories
    print("📦 Filtering items and creating item features...")
    item2id = {}
    item_feat_dict = {}
    item_id_base = 1
    valid_item_count = 0
    for row in stream_jsonl(meta_file):
        raw_cats = row.get("categories", "")
        if not raw_cats:
            continue
        
        if isinstance(raw_cats, list):
            cats = [cat.strip() for cat in raw_cats if cat.strip()]
        else:
            cats = [cat.strip() for cat in raw_cats.split(',') if cat.strip()]
        
        if not set(cats).intersection(target_categories):
            continue
        
        try:
            rn_val = float(row.get("rating_number", 0))
            if rn_val < min_item_reviews:
                continue
        except:
            continue
        
        price_val = safe_parse_price(row.get("price"))
        if rn_val > rating_number_threshold or price_val > price_threshold:
            continue
        
        pid = row['parent_asin']
        if pid not in item2id:
            item2id[pid] = item_id_base
            item_id_base += 1
        
        # Build item features: item id, binned average rating, rating count, and price
        features = [
            item2id[pid],
            bin_avg_rating(row['average_rating']),
            bin_rating_count(rn_val),
            bin_price(price_val)
        ]
        item_feat_dict[pid] = features
        valid_item_count += 1
    print(f"✅ {valid_item_count} items kept after filtering and outlier removal.")

    # Process user interactions and create user features
    print("👤 Processing user interactions and creating user features...")
    user2id = {}
    user_feat_dict = {}
    user_interactions = defaultdict(list)
    user_id_base = 1
    total_interactions = 0
    for row in stream_jsonl(user_file):
        uid, pid = row['user_id'], row['parent_asin']
        if pid not in item_feat_dict:
            continue
        try:
            user_reviews = int(row.get("user_total_reviews", 0))
        except:
            continue
        if user_reviews < min_user_reviews:
            continue
        if uid not in user2id:
            user2id[uid] = user_id_base
            user_id_base += 1
        if uid not in user_feat_dict:
            user_feat_dict[uid] = [
                user2id[uid],
                bin_helpful_vote(row['helpful_vote']),
                bin_user_avg_rating(row['user_average_rating']),
                bin_user_total_reviews(row['user_total_reviews']),
                bin_user_verified_purchase_ratio(row['user_verified_purchase_ratio'])
            ]
        # Record the timestamp along with the pid.
        timestamp = int(row.get("timestamp", 0))
        user_interactions[uid].append((timestamp, pid))
        total_interactions += 1
    print(f"✅ Loaded {len(user2id)} users with a total of {total_interactions} interactions.")

    # Sort each user's interactions by timestamp so that the latest is last
    for uid in user_interactions:
        user_interactions[uid].sort(key=lambda x: x[0])
    
    print(f"🔁 Subsampling user interactions (ratio = {subsample_ratio})...")
    if subsample_ratio < 1.0:
        before_subsample = len(user_interactions)
        user_interactions = stratified_subsample_interactions(user_interactions, subsample_ratio)
        print(f"✅ Subsampled users: {before_subsample} ➜ {len(user_interactions)}.")

    # Write out training and testing splits
    print("📤 Writing train/test CSV files...")
    train_cnt, test_cnt = 0, 0
    with open(train_output, "w", newline='', encoding='utf-8') as f_train, \
         open(test_output, "w", newline='', encoding='utf-8') as f_test:
        writer_train = csv.writer(f_train)
        writer_test = csv.writer(f_test)
        for uid, interactions in user_interactions.items():
            if len(interactions) < 2:
                continue
            
            # Choose test instance based on splitting rule
            if use_random_test:
                chosen_index = random.randint(0, len(interactions) - 1)
                test_pid = interactions[chosen_index][1]
                train_pids = [interaction[1] for idx, interaction in enumerate(interactions) if idx != chosen_index]
            else:
                # Default: use the most recent (last after sorting) interaction as test
                test_pid = interactions[-1][1]
                train_pids = [interaction[1] for interaction in interactions[:-1]]
            
            # Write training interactions
            for pid in train_pids:
                writer_train.writerow([
                    to_str(user_feat_dict[uid]),
                    to_str(item_feat_dict[pid])
                ])
                train_cnt += 1
            
            # Write the test interaction
            writer_test.writerow([
                to_str(user_feat_dict[uid]),
                to_str(item_feat_dict[test_pid])
            ])
            test_cnt += 1

    print(f"📂 Train interactions: {train_cnt}")
    print(f"📂 Test interactions: {test_cnt}")
    print(f"💾 Train file written to: {train_output}")
    print(f"💾 Test file written to: {test_output}")
    print("✅ Unified dataset processed successfully.")

def main():
    parser = argparse.ArgumentParser(
        description="Process Amazon Books datasets for a unified set across multiple target categories with custom train/test split rules"
    )
    parser.add_argument("--meta_file", type=str, default="meta_Books.jsonl/meta_Books_processed.jsonl",
                        help="Path to meta Books file")
    parser.add_argument("--user_file", type=str, default="Books.jsonl/Books_processed.jsonl",
                        help="Path to Books user interactions file")
    parser.add_argument("--min_item_reviews", type=int, default=59,
                        help="Minimum number of reviews an item must have to be kept")
    parser.add_argument("--min_user_reviews", type=int, default=15,
                        help="Minimum number of reviews a user must have to be kept")
    parser.add_argument("--subsample_ratio", type=float, default=1.0,
                        help="Subsample ratio for users (0-1)")
    parser.add_argument("--use_random_test", action="store_true",
                        help="If set, randomly select one test instance per user (for datasets without timestamp)")
    args = parser.parse_args()
    
    # Define the target categories to combine
    target_categories = ["Arts & Photography", "Genre Fiction", "History"]
    # Set the unified output directory (adjust the path as needed)
    unified_output_dir = r"D:\Project\CARS\AMZ\full"
    
    process_unified(
        meta_file=args.meta_file,
        user_file=args.user_file,
        target_categories=target_categories,
        min_item_reviews=args.min_item_reviews,
        min_user_reviews=args.min_user_reviews,
        output_dir=unified_output_dir,
        subsample_ratio=args.subsample_ratio,
        use_random_test=args.use_random_test
    )

if __name__ == "__main__":
    main()
