import os
import json
import datetime
import csv
import argparse
import random
from collections import defaultdict

# === Data Loading Functions without Binning ===

def load_users(user_file):
    """
    Loads Yelp user JSON and builds a reduced user feature vector.
    Now includes:
      - elite_years: how many seasons the user was “elite” (0 if none)
    Final user feature vector:
      [ user_id, unique_useful, unique_avg_star, unique_review_count, elite_years ]
    """
    users = {}
    user_mapping = {}
    idx = 0
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
            unique_avg_star   = int(round(avg_star * 10)) + 1000
            review_count_val  = int(data.get('review_count', 0))
            unique_review_count = review_count_val + 2000
            useful_val        = int(data.get('useful', 0))
            unique_useful     = useful_val + 3000

            # NEW: count of elite years
            elite_list = data.get('elite') or []
            # either as count:
            elite_years = len(elite_list)
            # or as binary: elite_flag = 1 if elite_years>0 else 0

            users[uid] = [
                user_id_feat,
                unique_useful,
                unique_avg_star,
                unique_review_count,
                elite_years
            ]
    return users, user_mapping


def load_businesses(business_file, allowed_categories=None):
    """
    Loads Yelp business JSON and builds a reduced business feature vector.
    Now includes:
      - is_open: 1 if open, 0 if closed
    Final business feature vector:
      [ business_id, unique_avg_star, unique_review_count, city_id, is_open ]
    """
    if allowed_categories is None:
        allowed_categories = {"Food", "Restaurants", "Shopping"}

    businesses = {}
    business_mapping = {}
    city_mapping = {}
    idx = 0

    with open(business_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            bid = data['business_id']
            cats = []
            if data.get('categories'):
                cats = [c.strip() for c in data['categories'].split(',')]

            # filter categories as before
            if not set(cats).intersection(allowed_categories):
                continue

            # assign compact index
            if bid not in business_mapping:
                business_mapping[bid] = idx
                idx += 1
            business_id_feat = business_mapping[bid] + 1 + 4000

            # existing features
            stars_val   = float(data.get('stars', 0.0))
            unique_avg_star   = int(round(stars_val * 10)) + 5000
            review_count_val  = int(data.get('review_count', 0))
            unique_review_count = review_count_val + 6000

            # city → compact ID
            city_raw = data.get('city', '').strip()
            if city_raw not in city_mapping:
                city_mapping[city_raw] = len(city_mapping) + 1
            city_id = city_mapping[city_raw]

            # NEW: open flag
            is_open = int(data.get('is_open', 0))

            businesses[bid] = [
                business_id_feat,
                unique_avg_star,
                unique_review_count,
                city_id,
                is_open
            ]
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

def get_most_common_city(uid, reviews_by_user, businesses):
    """
    Determines the most common city that a user interacts with based on their reviews.
    
    Parameters:
      uid (str): The user ID
      reviews_by_user (dict): Map of user ID to their reviews
      businesses (dict): Map of business ID to business features
      
    Returns:
      int: The ID of the most common city, or 0 if no data available
    """
    city_counts = defaultdict(int)
    for review in reviews_by_user[uid]:
        bid = review['business_id']
        # City is the 4th feature (index 3) in business features
        city_id = businesses[bid][3]
        city_counts[city_id] += 1
    
    if not city_counts:
        return 0  # Default if no data available
    
    # Return the city with highest frequency
    return max(city_counts.items(), key=lambda x: x[1])[0]

def add_main_city_to_users(users, reviews_by_user, businesses):
    """
    Adds the most common city feature to each user's feature vector.
    
    Parameters:
      users (dict): Map of user ID to user features
      reviews_by_user (dict): Map of user ID to their reviews
      businesses (dict): Map of business ID to business features
      
    Returns:
      dict: Updated map of user ID to user features with main city added
    """
    updated_users = {}
    for uid in users:
        if uid in reviews_by_user:
            main_city = get_most_common_city(uid, reviews_by_user, businesses)
            # Add the main city as a new feature at the end
            updated_features = users[uid] + [main_city]
            updated_users[uid] = updated_features
        else:
            # For users without reviews, set city to 0
            updated_features = users[uid] + [0]
            updated_users[uid] = updated_features
    return updated_users

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
    parser.add_argument("--subsample_ratio", type=float, default=0.5, help="Subsample ratio for users (0-1)")
    args = parser.parse_args()

    # Mapping from allowed category to output directory.
    category_outputs = {
        "Food": r"D:\Project\CARS\Yelp JSON\yelp_dataset\Food",
        "Restaurants": r"D:\Project\CARS\Yelp JSON\yelp_dataset\Restaurants",
        "Shopping": r"D:\Project\CARS\Yelp JSON\yelp_dataset\Shopping"
    }

    # Load users (non-elite only).
    print("Loading user data...")
    users, _ = load_users(args.user_file)
    print(f"Loaded {len(users)} users.")

    # Process each category separately.
    for category, out_dir in category_outputs.items():
        print(f"\nProcessing category: {category}")

        # Ensure the output directory exists.
        os.makedirs(out_dir, exist_ok=True)

        # Load business data filtered by allowed categories.
        print("Loading business data...")
        businesses, _ = load_businesses(args.business_file, allowed_categories={category})
        print(f"Loaded {len(businesses)} businesses for category '{category}'.")

        # Load review data for users and businesses.
        print("Loading review data...")
        reviews_by_user = load_reviews(args.review_file, users, businesses)
        reviews_by_user = frequency_filter_reviews(reviews_by_user, args.min_user_reviews)
        print(f"After filtering, {len(reviews_by_user)} users remain (min {args.min_user_reviews} reviews) for category '{category}'.")

        if args.subsample_ratio < 1.0:
            reviews_by_user = stratified_subsample_reviews(reviews_by_user, args.subsample_ratio)
            print(f"After subsampling, {len(reviews_by_user)} users remain for category '{category}'.")
            
        # Add main city feature to users based on the current category's reviews
        print(f"Adding main city feature to users for category '{category}'...")
        users_with_city = add_main_city_to_users(users, reviews_by_user, businesses)
        print(f"Main city feature added to user vectors for category '{category}'.")

        # ===== Reindex Users =====
        new_user_mapping = {}
        new_users = {}
        new_user_index = 0
        for uid in reviews_by_user.keys():
            new_user_mapping[uid] = new_user_index
            old_feats = users_with_city[uid]  # Use the updated users with city feature
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
            # Update business ID feature with new index (starting from 1, preserving the offset).
            new_feats = [new_business_index + 1 + 4000] + old_feats[1:]
            new_businesses[bid] = new_feats
            new_business_index += 1
        businesses_cat = new_businesses
        print(f"Businesses reindexed: {len(businesses_cat)} businesses with interactions for category '{category}'.")

        # ===== Split Data (Leave-One-Out Evaluation) =====
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