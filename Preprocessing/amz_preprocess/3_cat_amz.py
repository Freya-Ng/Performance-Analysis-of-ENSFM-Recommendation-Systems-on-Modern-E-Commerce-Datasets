#!/usr/bin/env python
import os
import json
import csv
from collections import defaultdict
import numpy as np
import random
import argparse

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

def process_category(meta_file, user_file, target_category, min_item_reviews, min_user_reviews, subsample_ratio=1.0, use_random_test=False):
    """Process a single category dataset and create train/test splits.
    
    - Output paths for this category will be:
        AMZ/<target_category>/train.csv and AMZ/<target_category>/test.csv
    - If use_random_test is False, the latest (most recent) interaction per user is held out (after sorting by timestamp).
      If True, one interaction is randomly selected from each user as test.
    """
    print(f"\n🔍 PROCESSING CATEGORY: '{target_category}'")
    
    # Set output paths for the category
    output_dir = f"AMZ/{target_category}"
    os.makedirs(output_dir, exist_ok=True)
    train_output = f"{output_dir}/train.csv"
    test_output = f"{output_dir}/test.csv"
    
    # Calculate outlier thresholds based on the meta_file
    print("📏 Calculating outlier thresholds...")
    rating_numbers_list = []
    prices_list = []
    for row in stream_jsonl(meta_file):
        raw_cats = row.get("categories", "")
        if not raw_cats:
            continue
        # Handle both list and comma-separated strings
        if isinstance(raw_cats, list):
            cats = [cat.strip() for cat in raw_cats if cat.strip()]
        else:
            cats = [cat.strip() for cat in raw_cats.split(',') if cat.strip()]
        
        if target_category not in cats:
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

    # Build item features for valid items in the target category
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
        
        if target_category not in cats:
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
        
        # Build item features with direct value assignment
        # Format: [item_id, avg_rating + offset, rating_count + offset, price + offset]
        avg_rating_feat = int(round(float(row['average_rating']) * 10)) + 5000  # Scale up and add offset
        rating_count_feat = int(rn_val) + 6000  # Add offset
        price_feat = int(price_val) + 7000  # Add offset
        
        features = [
            item2id[pid],
            avg_rating_feat,
            rating_count_feat,
            price_feat
        ]
        item_feat_dict[pid] = features
        valid_item_count += 1
    print(f"✅ {valid_item_count} items kept after filtering for '{target_category}' and outlier removal.")

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
            # Direct value assignment with offsets
            helpful_vote_feat = int(row['helpful_vote']) + 1000
            avg_rating_feat = int(round(float(row['user_average_rating']) * 10)) + 2000
            review_count_feat = int(row['user_total_reviews']) + 3000
            verified_ratio_feat = int(float(row['user_verified_purchase_ratio']) * 100) + 4000
            
            user_feat_dict[uid] = [
                user2id[uid],
                helpful_vote_feat,
                avg_rating_feat,
                review_count_feat,
                verified_ratio_feat
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
                # By default, use the most recent (last after sorting) interaction as test
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
    print(f"✅ Dataset for '{target_category}' processed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Process Amazon Books datasets for multiple categories with custom train/test split rules"
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
    
    categories = [
        "Arts & Photography",
        "Genre Fiction",
        "Children's Books"
    ]
    
    # Process each category separately
    for category in categories:
        process_category(
            meta_file=args.meta_file,
            user_file=args.user_file,
            target_category=category,
            min_item_reviews=args.min_item_reviews,
            min_user_reviews=args.min_user_reviews,
            subsample_ratio=args.subsample_ratio,
            use_random_test=args.use_random_test
        )

if __name__ == "__main__":
    main()