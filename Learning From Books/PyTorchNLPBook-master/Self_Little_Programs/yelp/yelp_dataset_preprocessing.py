import collections
import numpy as np
import pandas as pd
import re

from argparse import Namespace

args = Namespace(
    raw_train_dataset_csv="E:\\github lab\\data\\yelp_data\\raw_train.csv",
    raw_test_dataset_csv="E:\\github lab\\data\\yelp_data\\raw_test.csv",
    train_proportion=0.7,
    val_proportion=0.3,
    output_munged_csv="E:\\github lab\\data\\yelp_data\\reviews_with_splits_full.csv",
    seed=1337
)
# Read raw data
train_reviews = pd.read_csv(args.raw_train_dataset_csv, header=None, names=['rating', 'review'])
train_reviews = train_reviews[~pd.isnull(train_reviews.review)]
test_reviews = pd.read_csv(args.raw_test_dataset_csv, header=None, names=['rating', 'review'])
test_reviews = test_reviews[~pd.isnull(test_reviews.review)]

by_rating = collections.defaultdict(list)
for _,row in train_reviews.iterrows():
    by_rating[row.rating].append(row.to_dict())

final_list = []
np.random.seed(args.seed)
for _,item_list in sorted(by_rating.items()):
    np.random.shuffle(item_list)
    n_total = len(item_list)
    n_train = int(args.train_proportion * n_total)
    n_val = int(args.val_proportion * n_total)
    # Give data point a split attribute
    for item in item_list[:n_train]:
        item['split'] = 'train'
    for item in item_list[n_train:n_train+n_val]:
        item['split'] = 'val'
    # Add to final list
    final_list.extend(item_list)

for _, row in test_reviews.iterrows():
    row_dict = row.to_dict()
    row_dict['split'] = 'test'
    final_list.append(row_dict)

final_reviews = pd.DataFrame(final_list)

# Preprocess the reviews
def preprocess_text(text):
    if type(text) == float:
        print(text)
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text

final_reviews.review = final_reviews.review.apply(preprocess_text)

final_reviews['rating'] = final_reviews.rating.apply({1: 'negative', 2: 'positive'}.get)

final_reviews.to_csv(args.output_munged_csv, index=False)
