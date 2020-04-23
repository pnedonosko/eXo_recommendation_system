import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import feature_column


class Dataset:

    def __init__(self, path2file):
        self.df = pd.read_csv(path2file)
        self.feature_columns = []
        self.train_ds, self.val_ds, self.test_ds = None, None, None


    def create_feature_columns(self):
        feature_columns = []

        numeric_cols = ['owner_influence', 'is_commented_by_connections', 'is_liked_by_me', 'is_liked_by_connections',
                        'poster_gender', 'poster_influence',
                        'participant1_gender', 'participant1_influence', 'participant2_gender',
                        'participant2_influence', 'participant3_gender', 'participant3_influence']

        # numeric cols
        for header in numeric_cols:
            feature_columns.append(feature_column.numeric_column(header))

        # bucketized columns

        # age
        step = len(self.df.age) // 8
        sorted_ages = sorted(self.df.age)
        age_boundaries = [sorted_ages[i * step] for i in range(1, 8)]

        age = feature_column.numeric_column("age")
        age_buckets = feature_column.bucketized_column(age, boundaries=age_boundaries)
        feature_columns.append(age_buckets)

        # number_of_likes
        likes_num = feature_column.numeric_column("number_of_likes")
        likes_num_buckets = feature_column.bucketized_column(likes_num, boundaries=[2, 5, 10, 20, 50, 100])
        feature_columns.append(likes_num_buckets)


        # number_of_comments
        comments_num = feature_column.numeric_column("number_of_comments")
        comments_num_buckets = feature_column.bucketized_column(comments_num, boundaries=[1, 2, 5, 10, 20, 50, 100])
        feature_columns.append(comments_num_buckets)

        # indicator columns for categorical features

        app_type = feature_column.categorical_column_with_vocabulary_list(
            'app_type', self.df.app_type.unique())
        app_type_1hot = feature_column.indicator_column(app_type)
        feature_columns.append(app_type_1hot)

        owner_type = feature_column.categorical_column_with_vocabulary_list(
            'owner_type', self.df.owner_type.unique())
        owner_type_1hot = feature_column.indicator_column(owner_type)
        feature_columns.append(owner_type_1hot)

        poster_focus = feature_column.categorical_column_with_vocabulary_list(
            'poster_focus', ['engineering', 'sales', 'marketing', 'management', 'financial', 'other'])
        poster_focus_1hot = feature_column.indicator_column(poster_focus)
        feature_columns.append(poster_focus_1hot)

        # functions to reduce code duplication
        def participant_action(part_action):
            participant_action = feature_column.categorical_column_with_vocabulary_list(
                part_action, ['commented', 'liked', 'viewed'])
            return participant_action

        def participant_focus(part_f):
            participant_focus = feature_column.categorical_column_with_vocabulary_list(
                part_f, ['engineering', 'sales', 'marketing', 'management', 'financial', 'other', 'none'])
            return participant_focus

        participant1_action = participant_action("participant1_action")
        participant2_action = participant_action("participant2_action")
        participant3_action = participant_action("participant3_action")

        participant1_focus = participant_focus("participant1_focus")
        participant2_focus = participant_focus("participant2_focus")
        participant3_focus = participant_focus("participant3_focus")

        feature_columns.append(feature_column.indicator_column(participant1_action))
        feature_columns.append(feature_column.indicator_column(participant1_focus))
        feature_columns.append(feature_column.indicator_column(participant2_action))
        feature_columns.append(feature_column.indicator_column(participant2_focus))
        feature_columns.append(feature_column.indicator_column(participant3_action))
        feature_columns.append(feature_column.indicator_column(participant3_focus))

        # feature crosses for participant action and focus
        crossed_feature1 = feature_column.crossed_column([participant1_action, participant1_focus],
                                                         hash_bucket_size=1000)
        crossed_feature1 = feature_column.indicator_column(crossed_feature1)
        feature_columns.append(crossed_feature1)

        crossed_feature2 = feature_column.crossed_column([participant2_action, participant2_focus],
                                                         hash_bucket_size=1000)
        crossed_feature2 = feature_column.indicator_column(crossed_feature2)
        feature_columns.append(crossed_feature2)

        crossed_feature3 = feature_column.crossed_column([participant3_action, participant3_focus],
                                                         hash_bucket_size=1000)
        crossed_feature3 = feature_column.indicator_column(crossed_feature3)
        feature_columns.append(crossed_feature3)


        self.feature_columns = feature_columns



    def preprocess(self):

        # replace categorical values of all 'gender' columns to binary
        self.df.poster_gender.replace(['male', 'female'], [1, 0], inplace=True)
        self.df.participant1_gender.replace(['male', 'female'], [1, 0], inplace=True)
        self.df.participant2_gender.replace(['male', 'female'], [1, 0], inplace=True)
        self.df.participant3_gender.replace(['male', 'female'], [1, 0], inplace=True)


        # for 'poster_focus' column change 'none' to 'other' category
        self.df.poster_focus.replace('none', 'other', inplace=True)

        # drop redundant columns with ids
        self.df.drop(['id', 'poster_id', 'participant1_id', 'participant2_id', 'participant3_id'], axis=1, inplace=True)




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
