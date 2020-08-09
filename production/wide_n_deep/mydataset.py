import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import feature_column

from utils import create_effective_age, create_weekday, create_daytime, participant_action, participant_focus
from config import _ONE_MIN, _ONE_HOUR, _PEAK_INTERVAL_LONG, _NONPEAK_INTERVAL_LONG

class Dataset:

    def __init__(self, path2file):
        self.df = pd.read_csv(path2file)
        self.real, self.sparse, self.inputs = None, None, None
        self.train_ds, self.val_ds, self.test_ds = None, None, None


    def create_synthetic_features(self):
        # convert to seconds
        self.df.posted_time /= 1000

        self.df["effective_age_long"] = create_effective_age(self.df.posted_time, _PEAK_INTERVAL_LONG,
                                                             _NONPEAK_INTERVAL_LONG)

        posted_time = self.df.posted_time
        self.df["weekday"] = create_weekday(posted_time)
        self.df["daytime"] = create_daytime(posted_time)


    def create_feature_columns(self):
        _NUMERIC_COLUMNS = ['posted_time', 'owner_influence', 'poster_influence', 'participant1_influence',
                            'participant2_influence', 'participant3_influence', 'participant4_influence',
                            'participant5_influence']

        _BINARY_COLUMNS = ["is_mentions_me", "is_mentions_connections", "is_commented_by_me",
                           "is_commented_by_connections", "is_liked_by_me", "is_liked_by_connections",
                           "poster_is_employee","poster_is_in_connections",
                           "participant1_is_employee", "participant1_is_in_connections",
                           "participant2_is_employee", "participant2_is_in_connections",
                           "participant3_is_employee", "participant3_is_in_connections",
                           "participant4_is_employee", "participant4_is_in_connections",
                           "participant5_is_employee", "participant5_is_in_connections"]

        _GENDER_COLUMNS = ["poster_gender", "participant1_gender", "participant2_gender", "participant3_gender",
                           "participant4_gender", "participant5_gender"]

        self.real = {
            colname: feature_column.numeric_column(colname) \
            for colname in _NUMERIC_COLUMNS
        }

        self.sparse = dict()

        app_type = feature_column.categorical_column_with_vocabulary_list(
            'app_type', self.df.app_type.unique())
        app_type_1hot = feature_column.indicator_column(app_type)
        self.sparse["app_type"] = app_type_1hot

        owner_type = feature_column.categorical_column_with_vocabulary_list(
            'owner_type', self.df.owner_type.unique())
        owner_type_1hot = feature_column.indicator_column(owner_type)
        self.sparse["owner_type"] = owner_type_1hot

        poster_focus = feature_column.categorical_column_with_vocabulary_list(
            'poster_focus', ['engineering', 'sales', 'marketing', 'management', 'financial', 'other'])
        poster_focus_1hot = feature_column.indicator_column(poster_focus)
        self.sparse["poster_focus"] = poster_focus_1hot

        for col in _GENDER_COLUMNS:
            feature = feature_column.categorical_column_with_vocabulary_list(col, self.df[col].unique())
            feature_1hot = feature_column.indicator_column(feature)
            self.sparse[col] = feature_1hot

        participant1_action = participant_action("participant1_action")
        participant2_action = participant_action("participant2_action")
        participant3_action = participant_action("participant3_action")
        participant4_action = participant_action("participant4_action")
        participant5_action = participant_action("participant5_action")

        participant1_focus = participant_focus("participant1_focus")
        participant2_focus = participant_focus("participant2_focus")
        participant3_focus = participant_focus("participant3_focus")
        participant4_focus = participant_focus("participant4_focus")
        participant5_focus = participant_focus("participant5_focus")

        self.sparse["participant2_action"] = feature_column.indicator_column(participant2_action)
        self.sparse["participant3_action"] = feature_column.indicator_column(participant3_action)
        self.sparse["participant1_action"] = feature_column.indicator_column(participant1_action)
        self.sparse["participant4_action"] = feature_column.indicator_column(participant4_action)
        self.sparse["participant5_action"] = feature_column.indicator_column(participant5_action)

        self.sparse["participant1_focus"] = feature_column.indicator_column(participant1_focus)
        self.sparse["participant2_focus"] = feature_column.indicator_column(participant2_focus)
        self.sparse["participant3_focus"] = feature_column.indicator_column(participant3_focus)
        self.sparse["participant4_focus"] = feature_column.indicator_column(participant4_focus)
        self.sparse["participant5_focus"] = feature_column.indicator_column(participant5_focus)

        self.inputs = {
            colname: tf.keras.layers.Input(name=colname, shape=(), dtype='float32') \
            for colname in self.real.keys()
        }

        self.inputs.update({
            colname: tf.keras.layers.Input(name=colname, shape=(), dtype='string') \
            for colname in self.sparse.keys()
        })

        for col in _BINARY_COLUMNS:
            feature = feature_column.categorical_column_with_vocabulary_list(
                col, self.df[col].unique())
            feature_1hot = feature_column.indicator_column(feature)
            self.sparse[col] = feature_1hot

        likes_num = feature_column.numeric_column("number_of_likes")
        likes_num_buckets = feature_column.bucketized_column(likes_num, boundaries=[2, 5, 10, 20, 50, 100])
        self.sparse["number_of_likes"] = likes_num_buckets

        comments_num = feature_column.numeric_column("number_of_comments")
        comments_num_buckets = feature_column.bucketized_column(comments_num, boundaries=[1, 2, 5, 10, 20, 50, 100])
        self.sparse["number_of_comments"] = comments_num_buckets

        age_boundaries = [30 * _ONE_MIN, _ONE_HOUR, 2 * _ONE_HOUR, 3 * _ONE_HOUR, 4 * _ONE_HOUR, 24 * _ONE_HOUR]

        age = feature_column.numeric_column("effective_age_long")
        age_buckets = feature_column.bucketized_column(age, boundaries=age_boundaries)
        self.sparse["effective_age_long"] = age_buckets

        daytime = feature_column.numeric_column("daytime")
        daytime_buckets = feature_column.bucketized_column(daytime, boundaries=[8, 12, 16, 24])
        self.sparse["daytime"] = daytime_buckets

        weekday = feature_column.categorical_column_with_vocabulary_list(
            'weekday', self.df.weekday.unique())
        weekday_1hot = feature_column.indicator_column(weekday)
        self.sparse["weekday"] = weekday_1hot

        self.inputs.update({
            colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int64') \
            for colname in
            _BINARY_COLUMNS + ["number_of_likes", "number_of_comments", "effective_age_long", "daytime", "weekday"]
        })

        # hash_bucket_size=30 because there are 7 possible values in weekday and 4 in daytime, all possible combinations will be 28 ~ 30
        weekday_x_daytime = feature_column.crossed_column([weekday, daytime_buckets], hash_bucket_size=30)
        self.sparse["weekday_x_daytime"] = feature_column.indicator_column(weekday_x_daytime)

        # 6 bins in likes and 7 in comments
        likes_x_comments = feature_column.crossed_column([likes_num_buckets, comments_num_buckets], hash_bucket_size=45)
        self.sparse["likes_x_comments"] = feature_column.indicator_column(likes_x_comments)

        # 6 bins in likes, 3 in action and 7 in focus, 6*3*7=126~130
        likes_x_participant1_focus_n_action = feature_column.crossed_column(
            [likes_num_buckets, participant1_action, participant1_focus], hash_bucket_size=130)
        self.sparse["likes_x_participant1_focus_n_action"] = feature_column.indicator_column(
            likes_x_participant1_focus_n_action)

        likes_x_participant2_focus_n_action = feature_column.crossed_column(
            [likes_num_buckets, participant2_action, participant2_focus], hash_bucket_size=130)
        self.sparse["likes_x_participant2_focus_n_action"] = feature_column.indicator_column(
            likes_x_participant2_focus_n_action)

        likes_x_participant3_focus_n_action = feature_column.crossed_column(
            [likes_num_buckets, participant3_action, participant3_focus], hash_bucket_size=130)
        self.sparse["likes_x_participant3_focus_n_action"] = feature_column.indicator_column(
            likes_x_participant3_focus_n_action)

        likes_x_participant4_focus_n_action = feature_column.crossed_column(
            [likes_num_buckets, participant4_action, participant4_focus], hash_bucket_size=130)
        self.sparse["likes_x_participant4_focus_n_action"] = feature_column.indicator_column(
            likes_x_participant4_focus_n_action)

        likes_x_participant5_focus_n_action = feature_column.crossed_column(
            [likes_num_buckets, participant5_action, participant5_focus], hash_bucket_size=130)
        self.sparse["likes_x_participant5_focus_n_action"] = feature_column.indicator_column(
            likes_x_participant5_focus_n_action)



    def preprocess(self):
        # for 'poster_focus' column change 'none' to 'other' category
        self.df.poster_focus.replace('none', 'other', inplace=True)

        redundant_cols = ["id", "updated_time", "age", "updated_age", "owner_id", "reactivity", "poster_id",
                          "poster_is_lead", "participant1_is_lead", "participant2_is_lead", "participant3_is_lead",
                          "participant4_is_lead", "participant5_is_lead", "poster_order", "participant1_order",
                          "participant2_order", "participant3_order", "participant4_order", "participant5_order",
                          "participant1_id", "participant2_id", "participant3_id", "participant4_id", "participant5_id"]

        self.df.drop(redundant_cols, axis=1, inplace=True)



    def split_data(self):

        # A utility method to create a tf.data dataset from a Pandas Dataframe
        def df_to_dataset(dataframe, shuffle=True, batch_size=32):
            dataframe = dataframe.copy()
            labels = dataframe.pop('rank')
            ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
            if shuffle:
                ds = ds.shuffle(buffer_size=len(dataframe))
            ds = ds.batch(batch_size)
            return ds

        # split on train, val and test sets
        train, test = train_test_split(self.df, test_size=0.2)
        train, val = train_test_split(train, test_size=0.2)

        # create tf.data train, val and test sets to fit into a model
        batch_size = 32
        self.train_ds = df_to_dataset(train, batch_size=batch_size)
        self.val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
        self.test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
