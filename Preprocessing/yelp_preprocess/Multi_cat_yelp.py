#!/usr/bin/env python
"""
Version 4.0

This script generates synthetic train/test CSV files for a recommendation system
using Yelp data. It processes user, business, and review JSON files to generate
a reduced set of features.

Features Used:
    For Users:
        - Reindexed User ID
        - Useful Count (binned)
        - Average Star Rating (binned)
        - Review Count (binned)

    For Business:
        - Reindexed Business ID (with offset +2000)
        - Average Star Rating (binned)
        - Review Count (binned)
        - City (text field)

Referenced JSON Keys in Business JSON:
    business_id, categories, stars, review_count, is_open, city

Referenced JSON Keys in User JSON:
    user_id, elite, average_stars, review_count, useful

Referenced JSON Keys in Review JSON:
    user_id, business_id, date
"""

import os
import json
import datetime
import csv
import argparse
import random
from collections import defaultdict

# === Binning Functions ===

def bin_avg_star(star):
    if star < 2.5:
        return 50000
    elif star < 3.5:
        return 50001
    elif star < 4.0:
        return 50002
    elif star < 4.5:
        return 50003
    else:
        return 50004

def bin_review_count_user(count):
    if count < 10:
        return 51010
    elif count < 50:
        return 51011
    elif count < 200:
        return 51012
    else:
        return 51013

def bin_review_count_business(rc):
    if rc < 10:
        return 53010
    elif rc < 50:
        return 53011
    elif rc < 200:
        return 53012
    else:
        return 53013

def bin_generic(val, base):
    if val == 0:
        return base
    elif val < 5:
        return base + 1
    elif val < 20:
        return base + 2
    else:
        return base + 3

# === Data Loading Functions ===

def load_users(user_file):
    """
    Loads Yelp user JSON and builds a reduced user feature vector.
    Final user feature vector:
      [ user_id, bin_useful, bin_avg_star, bin_review_count ]
    Only non-elite users are considered.
    """
    users = {}
    user_mapping = {}
    idx = 0
    with open(user_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            uid = data['user_id']
            # Skip elite users.
            if data.get('elite'):
                continue
            if uid not in user_mapping:
                user_mapping[uid] = idx
                idx += 1
            user_id_feat = user_mapping[uid] + 1
            avg_star = float(data.get('average_stars', 0.0))
            review_count_val = int(data.get('review_count', 0))
            useful_val = int(data.get('useful', 0))
            user_avg_star_bin = bin_avg_star(avg_star) + 1000
            user_review_count_bin = bin_review_count_user(review_count_val)
            user_useful_bin = bin_generic(useful_val, 51100)
            users[uid] = [user_id_feat, user_useful_bin, user_avg_star_bin, user_review_count_bin]
    return users, user_mapping

def load_businesses(business_file, allowed_categories=None):
    """
    Loads Yelp business JSON and builds a reduced business feature vector.
    Final business feature vector:
      [ business_id, bin_avg_star, bin_review_count, city ]
    Only open businesses whose categories intersect with allowed_categories are kept.
    
    Parameters:
      business_file (str): Path to the business JSON file.
      allowed_categories (set or None): A set of category names to filter by.
                                        If None, defaults to {"Restaurants"}.
                                        
    Returns:
      businesses (dict): Mapping from business ID to its feature vector.
      business_mapping (dict): Mapping from original business ID to new index.
    """
    if allowed_categories is None:
        allowed_categories = {"Restaurants"}
        
    businesses = {}
    business_mapping = {}
    idx = 0

    with open(business_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            bid = data['business_id']
            raw_cats = data.get('categories')
            cats = []
            if raw_cats:
                cats = [cat.strip() for cat in raw_cats.split(',') if cat.strip()]
            # Check if the business belongs to at least one allowed category.
            if not set(cats).intersection(allowed_categories):
                continue
            # Only include open businesses.
            if data.get("is_open", 0) != 1:
                continue
            if bid not in business_mapping:
                business_mapping[bid] = idx
                idx += 1
            # Assign unique feature ID with an offset (+2000).
            business_id_feat = business_mapping[bid] + 1 + 2000
            stars_val = float(data.get('stars', 0.0))
            stars_bin = bin_avg_star(stars_val) + 3000
            rc = int(data.get('review_count', 0))
            rc_bin = bin_review_count_business(rc)
            # Retrieve the city (text) from the business JSON.
            city_val = data.get('city', '')
            businesses[bid] = [business_id_feat, stars_bin, rc_bin, city_val]
    return businesses, business_mapping

def load_reviews(review_file, users, businesses):
    """
    Loads reviews, keeping only those where both user and business are present.
    """
    reviews_by_user = defaultdict(list)
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
            except Exception:
                continue
            if uid not in users or bid not in businesses:
                continue
            reviews_by_user[uid].append({'business_id': bid, 'date': date_obj})
    return reviews_by_user

def frequency_filter_reviews(reviews_by_user, min_user_reviews):
    """
    Filters out users with fewer than min_user_reviews.
    """
    return {uid: reviews for uid, reviews in reviews_by_user.items() if len(reviews) >= min_user_reviews}

def stratified_subsample_reviews(reviews_by_user, subsample_ratio):
    """
    Stratified subsampling based on the number of reviews per user.
    """
    bins = defaultdict(list)
    for uid, reviews in reviews_by_user.items():
        bins[len(reviews)].append(uid)
    sampled_reviews = {}
    for count, uids in bins.items():
        sample_size = max(1, int(len(uids) * subsample_ratio))
        sampled_uids = random.sample(uids, sample_size)
        for uid in sampled_uids:
            sampled_reviews[uid] = reviews_by_user[uid]
    return sampled_reviews

def split_train_test(reviews_by_user):
    """
    Implements leave-one-out: for each user, the most recent review is test,
    and the remainder is used for training.
    """
    train_reviews = []
    test_reviews = []
    for uid, review_list in reviews_by_user.items():
        review_list.sort(key=lambda x: x['date'])
        test_review = review_list[-1]
        for r in review_list[:-1]:
            train_reviews.append({'user_id': uid, 'business_id': r['business_id']})
        test_reviews.append({'user_id': uid, 'business_id': test_review['business_id']})
    return train_reviews, test_reviews

def features_to_string(feature_list):
    """
    Converts a feature list into a dash-separated string.
    """
    return "-".join(str(x) for x in feature_list)

def write_csv(filename, interactions, users, businesses):
    """
    Writes CSV rows, each containing a user feature string and a business feature string.
    """
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for inter in interactions:
            uid = inter['user_id']
            bid = inter['business_id']
            user_feat_str = features_to_string(users[uid])
            business_feat_str = features_to_string(businesses[bid])
            writer.writerow([user_feat_str, business_feat_str])

# === Main Function ===

def main():
    parser = argparse.ArgumentParser(
        description="Create a synthetic train/test set for the union of Food, Home Services, and Shopping categories."
    )
    parser.add_argument("--user_file", type=str, default="yelp_user.json", help="Path to the Yelp user JSON file")
    parser.add_argument("--business_file", type=str, default="yelp_business.json", help="Path to the Yelp business JSON file")
    parser.add_argument("--review_file", type=str, default="yelp_review.json", help="Path to the Yelp review JSON file")
    parser.add_argument("--min_user_reviews", type=int, default=5, help="Minimum reviews per user to keep.")
    parser.add_argument("--subsample_ratio", type=float, default=1.0, help="Subsample ratio for users (0-1)")
    parser.add_argument("--output_dir", type=str, default=r"D:\Project\CARS\Yelp JSON\yelp_dataset\F_yelp", help="Output folder for synthetic train/test CSV files")
    args = parser.parse_args()

    # The allowed categories are now the union of Food, Home Services, and Shopping.
    allowed_categories = {"Food", "Home Services", "Shopping"}

    # Ensure the output directory exists.
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading user data (non-elite only)...")
    users, _ = load_users(args.user_file)
    print(f"Loaded {len(users)} users.")

    print("Loading business data for allowed categories...")
    businesses, _ = load_businesses(args.business_file, allowed_categories=allowed_categories)
    print(f"Loaded {len(businesses)} businesses from allowed categories: {allowed_categories}.")

    print("Loading review data...")
    reviews_by_user = load_reviews(args.review_file, users, businesses)
    reviews_by_user = frequency_filter_reviews(reviews_by_user, args.min_user_reviews)
    print(f"After filtering, {len(reviews_by_user)} users remain (min {args.min_user_reviews} reviews).")

    if args.subsample_ratio < 1.0:
        reviews_by_user = stratified_subsample_reviews(reviews_by_user, args.subsample_ratio)
        print(f"After subsampling, {len(reviews_by_user)} users remain.")

    # ===== Global Reindexing of Users =====
    new_user_mapping = {}
    new_users = {}
    new_user_index = 0
    for uid in reviews_by_user.keys():
        new_user_mapping[uid] = new_user_index
        # Replace the original user feature ID with the new global index (starting from 1).
        old_feats = users[uid]
        new_feats = [new_user_index + 1] + old_feats[1:]
        new_users[uid] = new_feats
        new_user_index += 1
    users_global = new_users
    print(f"Users reindexed: {len(users_global)} users with interactions.")

    # ===== Global Reindexing of Businesses =====
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
        # Update the business ID with the new global index (starting from 1, preserving offset +2000).
        new_feats = [new_business_index + 1 + 2000] + old_feats[1:]
        new_businesses[bid] = new_feats
        new_business_index += 1
    businesses_global = new_businesses
    print(f"Businesses reindexed: {len(businesses_global)} businesses used in reviews.")

    print("Splitting data using the leave-one-out protocol...")
    train_reviews, test_reviews = split_train_test(reviews_by_user)
    print(f"Total interactions: {len(train_reviews)} train, {len(test_reviews)} test.")

    # Write the synthetic train/test CSV files.
    train_path = os.path.join(args.output_dir, "train.csv")
    test_path = os.path.join(args.output_dir, "test.csv")
    print(f"Writing train.csv to {train_path}...")
    write_csv(train_path, train_reviews, users_global, businesses_global)
    print(f"Writing test.csv to {test_path}...")
    write_csv(test_path, test_reviews, users_global, businesses_global)
    print("Synthetic test/train set creation complete.")

if __name__ == "__main__":
    main()
