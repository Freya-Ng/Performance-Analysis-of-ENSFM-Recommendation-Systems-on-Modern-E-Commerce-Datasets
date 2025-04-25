#!/usr/bin/env python
"""
Version 4.0

This script generates training and test CSV files for a recommendation system based on Yelp data.
It processes user, business, and review JSON files to create reduced feature vectors.

Features Used:
    For Users:
        - User ID Feature (reindexed)
        - Useful Count (binned)
        - Average Star Rating (binned)
        - Review Count (binned)
    
    For Business:
        - Business ID Feature (reindexed, with offset +2000)
        - Average Star Rating (binned)
        - Review Count (binned)
        - City (text)

Referenced JSON Keys in Business JSON:
    address, attributes, business_id, categories, city, hours, is_open, latitude, longitude,
    name, postal_code, review_count, stars, state

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
        return 0
    elif star < 3.5:
        return 1
    elif star < 4.0:
        return 2
    elif star < 4.5:
        return 3
    else:
        return 4

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
            # Original user feature is built with the original mapping.
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
    Only open businesses that belong to one of the allowed categories are kept.
    
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
                # Split the categories and strip extra whitespace.
                cats = [cat.strip() for cat in raw_cats.split(',') if cat.strip()]
            # Filter out businesses that do not belong to at least one allowed category.
            if not set(cats).intersection(allowed_categories):
                continue
            # Only keep open businesses.
            if data.get("is_open", 0) != 1:
                continue
            if bid not in business_mapping:
                business_mapping[bid] = idx
                idx += 1
            # Reindex business with an offset (+2000) so the business feature ID is unique.
            business_id_feat = business_mapping[bid] + 1 + 2000
            stars_val = float(data.get('stars', 0.0))
            stars_bin = bin_avg_star(stars_val) + 3000
            rc = int(data.get('review_count', 0))
            rc_bin = bin_review_count_business(rc)
            # Retrieve the city (text) from business JSON.
            city_val = data.get('city', '')
            # Append city_val to the business feature vector.
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
    Keep only users with at least min_user_reviews reviews.
    """
    return {uid: reviews for uid, reviews in reviews_by_user.items() if len(reviews) >= min_user_reviews}

def stratified_subsample_reviews(reviews_by_user, subsample_ratio):
    """
    Performs stratified subsampling of users based on how many reviews they have.
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
    For each user, uses the most recent review as test and the rest as train (leave-one-out).
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
    Converts a feature list to a dash-separated string.
    """
    return "-".join(str(x) for x in feature_list)

def write_csv(filename, interactions, users, businesses):
    """
    Writes CSV rows where each row contains a user feature string and a business feature string.
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
        description="Generate train.csv and test.csv files with minimal features for recommendation for multiple categories."
    )
    parser.add_argument("--user_file", type=str, default="yelp_user.json", help="Path to Yelp user JSON file")
    parser.add_argument("--business_file", type=str, default="yelp_business.json", help="Path to Yelp business JSON file")
    parser.add_argument("--review_file", type=str, default="yelp_review.json", help="Path to Yelp review JSON file")
    parser.add_argument("--min_user_reviews", type=int, default=5, help="Minimum reviews per user to keep")
    parser.add_argument("--subsample_ratio", type=float, default=1.0, help="Subsample ratio for users (0-1)")
    args = parser.parse_args()

    # Mapping from allowed category to output directory.
    # Ensure the output directories use double backslashes or raw strings on Windows.
    category_outputs = {
        "Food": r"D:\Project\CARS\Yelp JSON\yelp_dataset\Food",
        "Home Services": r"D:\Project\CARS\Yelp JSON\yelp_dataset\Home Services",
        "Shopping": r"D:\Project\CARS\Yelp JSON\yelp_dataset\Shopping"
    }

    # Load users once (non-elite only).
    print("Loading user data...")
    users, _ = load_users(args.user_file)
    print(f"Loaded {len(users)} users.")

    # Process each category separately.
    for category, out_dir in category_outputs.items():
        print(f"\nProcessing category: {category}")

        # Ensure the output directory exists.
        os.makedirs(out_dir, exist_ok=True)

        # Load business data for this category using the allowed category filter.
        print("Loading business data...")
        businesses, _ = load_businesses(args.business_file, allowed_categories={category})
        print(f"Loaded {len(businesses)} businesses for category '{category}'.")

        # Load review data that corresponds to these businesses and users.
        print("Loading review data...")
        reviews_by_user = load_reviews(args.review_file, users, businesses)
        reviews_by_user = frequency_filter_reviews(reviews_by_user, args.min_user_reviews)
        print(f"After filtering, {len(reviews_by_user)} users remain (min {args.min_user_reviews} reviews) for category '{category}'.")

        if args.subsample_ratio < 1.0:
            reviews_by_user = stratified_subsample_reviews(reviews_by_user, args.subsample_ratio)
            print(f"After subsampling, {len(reviews_by_user)} users remain for category '{category}'.")

        # ===== Reindex Users =====
        new_user_mapping = {}
        new_users = {}
        new_user_index = 0
        for uid in reviews_by_user.keys():
            new_user_mapping[uid] = new_user_index
            old_feats = users[uid]
            # Update user ID feature with new index (starting from 1).
            new_feats = [new_user_index + 1] + old_feats[1:]
            new_users[uid] = new_feats
            new_user_index += 1
        users_cat = new_users
        print(f"Users reindexed: {len(users_cat)} users with interactions for category '{category}'.")

        # ===== Reindex Businesses =====
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
            # Update business ID feature with new index (starting from 1, preserving offset +2000).
            new_feats = [new_business_index + 1 + 2000] + old_feats[1:]
            new_businesses[bid] = new_feats
            new_business_index += 1
        businesses_cat = new_businesses
        print(f"Businesses reindexed: {len(businesses_cat)} businesses with interactions for category '{category}'.")

        # ===== Split Data (Leave-One-Out Evaluation) =====
        # For each user, the most recent review is held out as test while the rest are used as train.
        train_reviews, test_reviews = split_train_test(reviews_by_user)
        print(f"Split data: {len(train_reviews)} train interactions, {len(test_reviews)} test interactions for category '{category}'.")

        # Write the train/test CSV files into the corresponding folder.
        train_path = os.path.join(out_dir, "train.csv")
        test_path = os.path.join(out_dir, "test.csv")
        print(f"Writing train.csv to {train_path}...")
        write_csv(train_path, train_reviews, users_cat, businesses_cat)
        print(f"Writing test.csv to {test_path}...")
        write_csv(test_path, test_reviews, users_cat, businesses_cat)
        print(f"Done processing category '{category}'.")
    
    print("All categories processed.")

if __name__ == "__main__":
    main()
