import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset:

    def __init__(self, path2file):
        self.path2file = path2file
        self.df = pd.read_csv(path2file)
        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None



    def preprocess(self, remove_corr=True):

        # replace categorical values of all 'gender' columns to binary
        self.df.poster_gender.replace(['male', 'female'], [1, 0], inplace=True)
        self.df.participant1_gender.replace(['male', 'female'], [1, 0], inplace=True)
        self.df.participant2_gender.replace(['male', 'female'], [1, 0], inplace=True)
        self.df.participant3_gender.replace(['male', 'female'], [1, 0], inplace=True)

        # normalize (scale to 0-1 interval) columns bellow
        cols_to_norm = ['age', 'number_of_likes', 'number_of_comments']
        self.df[cols_to_norm] = self.df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        # for 'app_type' column change 'poll' & 'social' values to 'other' category
        self.df.app_type.replace(['poll', 'social'], 'other', inplace=True)

        # for 'poster_focus' column change 'none' to 'other' category
        self.df.poster_focus.replace('none', 'other', inplace=True)

        # change all categorical columns to dummy variables
        self.df = pd.get_dummies(self.df)

        if remove_corr:
            #define columns with low correlation to the target variable
            low_corr_columns = ['age', 'id', 'poster_id', 'poster_gender', 'participant1_id', 'participant2_id',
                                'participant3_id', 'participant1_gender', 'participant2_gender', 'participant3_gender']
            # remove these columns from the dataframe
            self.df.drop(low_corr_columns, axis=1, inplace=True)


    def split_data(self):
        # X - all columns except target feature
        X = self.df.loc[:, self.df.columns != 'rank']
        # Y - target variable
        Y = self.df['rank']

        # split dataset into train and test in 80/20 proportion
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
