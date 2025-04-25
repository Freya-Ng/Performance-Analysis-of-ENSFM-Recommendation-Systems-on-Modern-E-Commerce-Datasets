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

def process_unified(meta_file, user_file, target_categories, min_item_reviews, min_user_reviews, output_dir, subsample_ratio=1.0, use_random_test=False, balance_categories=True):
    """Process a unified dataset from multiple target categories and create train/test splits.
    
    The output paths will be:
        <output_dir>/train.csv and <output_dir>/test.csv
        
    If use_random_test is False, the latest (most recent) interaction per user is held out as test
    (after sorting by timestamp). Otherwise, one interaction is randomly selected per user.
    
    If balance_categories is True, we'll attempt to balance the dataset across different sized categories.
    """
    print(f"\n🔍 PROCESSING UNIFIED CATEGORIES: {target_categories}")
    
    # Set output paths for the unified dataset
    os.makedirs(output_dir, exist_ok=True)
    train_output = os.path.join(output_dir, "train.csv")
    test_output = os.path.join(output_dir, "test.csv")
    
    # Calculate outlier thresholds PER CATEGORY based on the meta_file
    print("📏 Calculating outlier thresholds per category...")
    category_rating_numbers = {cat: [] for cat in target_categories}
    category_prices = {cat: [] for cat in target_categories}
    
    # Ensure target_categories is a set for membership tests
    target_categories_set = set(target_categories)  
    
    for row in stream_jsonl(meta_file):
        raw_cats = row.get("categories", "")
        if not raw_cats:
            continue
        
        # Handle both list and comma-separated strings
        if isinstance(raw_cats, list):
            cats = [cat.strip() for cat in raw_cats if cat.strip()]
        else:
            cats = [cat.strip() for cat in raw_cats.split(',') if cat.strip()]
        
        # Find intersection with target categories
        matching_cats = set(cats).intersection(target_categories_set)
        if not matching_cats:
            continue
        
        try:
            rn = float(row.get("rating_number", 0))
            if rn < min_item_reviews:
                continue
            
            # Add to each matching category's data
            for cat in matching_cats:
                if cat in category_rating_numbers:  # Safety check
                    category_rating_numbers[cat].append(rn)
        except:
            continue
        
        pr = safe_parse_price(row.get("price"))
        for cat in matching_cats:
            if cat in category_prices:  # Safety check
                category_prices[cat].append(pr)
    
    # Calculate thresholds for each category
    category_rating_thresholds = {}
    category_price_thresholds = {}
    
    for cat in target_categories:
        if category_rating_numbers[cat]:
            category_rating_thresholds[cat] = np.percentile(category_rating_numbers[cat], 75)
        else:
            category_rating_thresholds[cat] = float("inf")
            
        if category_prices[cat]:
            category_price_thresholds[cat] = np.percentile(category_prices[cat], 75)
        else:
            category_price_thresholds[cat] = float("inf")
            
        print(f"Category '{cat}' thresholds: rating_number <= {category_rating_thresholds[cat]:.2f}, price <= {category_price_thresholds[cat]:.2f}")

    # Generate category ID mapping (for embedding)
    cat_to_id = {cat: idx+1 for idx, cat in enumerate(target_categories)}
    
    # Build item features for valid items in any of the target categories
    print("📦 Filtering items and creating item features...")
    item2id = {}
    item_feat_dict = {}  # Will now include category ID as a feature
    item_categories = {}  # Track which category each item belongs to
    item_id_base = 1
    valid_item_count = 0
    category_item_counts = {cat: 0 for cat in target_categories}
    
    for row in stream_jsonl(meta_file):
        raw_cats = row.get("categories", "")
        if not raw_cats:
            continue
        
        if isinstance(raw_cats, list):
            cats = [cat.strip() for cat in raw_cats if cat.strip()]
        else:
            cats = [cat.strip() for cat in raw_cats.split(',') if cat.strip()]
        
        # Find matching categories
        matching_cats = set(cats).intersection(target_categories_set)
        if not matching_cats:
            continue
        
        # Choose primary category (if item belongs to multiple target categories)
        # For simplicity, choose the first one alphabetically
        primary_cat = sorted(matching_cats)[0]
        
        try:
            rn_val = float(row.get("rating_number", 0))
            if rn_val < min_item_reviews:
                continue
        except:
            continue
        
        price_val = safe_parse_price(row.get("price"))
        
        # Apply category-specific thresholds
        if (rn_val > category_rating_thresholds[primary_cat] or 
            price_val > category_price_thresholds[primary_cat]):
            continue
        
        pid = row['parent_asin']
        if pid not in item2id:
            item2id[pid] = item_id_base
            item_id_base += 1
        
        # Build item features with direct value assignment
        # Format: [item_id, category_id + 8000, avg_rating + offset, rating_count + offset, price + offset]
        category_feat = cat_to_id[primary_cat] + 8000  # Add category embedding with offset
        avg_rating_feat = int(round(float(row['average_rating']) * 10)) + 5000  # Scale up and add offset
        rating_count_feat = int(rn_val) + 6000  # Add offset
        price_feat = int(price_val) + 7000  # Add offset
        
        features = [
            item2id[pid],
            category_feat,  # Add category ID as a feature
            avg_rating_feat,
            rating_count_feat,
            price_feat
        ]
        item_feat_dict[pid] = features
        item_categories[pid] = primary_cat  # Track which category this item belongs to
        category_item_counts[primary_cat] += 1
        valid_item_count += 1
    
    # Log category item counts
    print(f"✅ {valid_item_count} items kept after filtering and outlier removal.")
    for cat, count in category_item_counts.items():
        print(f"   - '{cat}': {count} items")

    # Process user interactions and create user features
    print("👤 Processing user interactions and creating user features...")
    user2id = {}
    user_feat_dict = {}
    user_interactions = defaultdict(list)
    user_id_base = 1
    total_interactions = 0
    category_interaction_counts = {cat: 0 for cat in target_categories}
    
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
        
        # Track category interaction counts
        if pid in item_categories:
            category_interaction_counts[item_categories[pid]] += 1
        
        total_interactions += 1
    
    print(f"✅ Loaded {len(user2id)} users with a total of {total_interactions} interactions.")
    for cat, count in category_interaction_counts.items():
        print(f"   - '{cat}': {count} interactions")

    # Sort each user's interactions by timestamp so that the latest is last
    for uid in user_interactions:
        user_interactions[uid].sort(key=lambda x: x[0])
    
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
        for uid, interactions in user_interactions.items():
            for timestamp, pid in interactions:
                if pid in item_categories:
                    cat = item_categories[pid]
                    # Probabilistic up-sampling for smaller categories
                    if random.random() < category_sampling_rates[cat]:
                        balanced_interactions[uid].append((timestamp, pid))
            
        # Replace original interactions with balanced ones
        # but only if the user still has interactions
        user_interactions = {uid: interactions for uid, interactions in balanced_interactions.items() 
                           if len(interactions) >= 2}
        
        # Recalculate category interaction counts after balancing
        new_category_counts = {cat: 0 for cat in target_categories}
        for uid, interactions in user_interactions.items():
            for _, pid in interactions:
                if pid in item_categories:
                    new_category_counts[item_categories[pid]] += 1
        
        print("Category counts after balancing:")
        for cat, count in new_category_counts.items():
            print(f"   - '{cat}': {count} interactions")
    
    # Apply general subsampling if requested
    print(f"🔁 Subsampling user interactions (ratio = {subsample_ratio})...")
    if subsample_ratio < 1.0:
        before_subsample = len(user_interactions)
        user_interactions = stratified_subsample_interactions(user_interactions, subsample_ratio)
        print(f"✅ Subsampled users: {before_subsample} ➜ {len(user_interactions)}.")

    # Global reindexing for users with interactions
    print("🔄 Reindexing users with interactions...")
    new_users = {}
    new_user_index = 0
    for uid in user_interactions.keys():
        old_feats = user_feat_dict[uid]
        # Replace original user ID with new global index (starting from 1)
        new_feats = [new_user_index + 1] + old_feats[1:]
        new_users[uid] = new_feats
        new_user_index += 1
    user_feat_dict = new_users
    print(f"✅ Users reindexed: {len(user_feat_dict)} users with interactions.")

    # Global reindexing for items that appear in interactions
    print("🔄 Reindexing items that appear in interactions...")
    used_item_ids = set()
    for interactions in user_interactions.values():
        for _, pid in interactions:
            used_item_ids.add(pid)
    
    new_items = {}
    new_item_index = 0
    for pid in used_item_ids:
        old_feats = item_feat_dict[pid]
        # Update item ID with new global index (starting from 1)
        new_feats = [new_item_index + 1] + old_feats[1:]
        new_items[pid] = new_feats
        new_item_index += 1
    item_feat_dict = new_items
    print(f"✅ Items reindexed: {len(item_feat_dict)} items used in interactions.")

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
    parser.add_argument("--mode", type=str, choices=['separate', 'unified'], default='unified',
                        help="Processing mode: 'separate' for individual categories or 'unified' for combined categories")
    parser.add_argument("--unified_output", type=str, default="AMZ/full",
                        help="Output directory for unified dataset (when mode is 'unified')")
    parser.add_argument("--balance_categories", action="store_true", 
                        help="Balance dataset across categories for unified mode")
    args = parser.parse_args()
    
    # Categories to process
    categories = [
        "Arts & Photography",
        "Genre Fiction", 
        "Children's Books"
    ]
    
    # Categories for unified processing (can be different from the individual categories)
    unified_categories = ["Arts & Photography", "Genre Fiction", "History"]
    
    if args.mode == 'separate':
        # Process each category separately
        print("🔄 Processing categories separately...")
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
    else:
        # Process all categories together in a unified dataset
        print("🔄 Processing categories in unified mode...")
        process_unified(
            meta_file=args.meta_file,
            user_file=args.user_file,
            target_categories=unified_categories,
            min_item_reviews=args.min_item_reviews,
            min_user_reviews=args.min_user_reviews,
            output_dir=args.unified_output,
            subsample_ratio=args.subsample_ratio,
            use_random_test=args.use_random_test,
            balance_categories=args.balance_categories
        )

if __name__ == "__main__":
    main()