from mydataset import Dataset
from wide_n_deep_model import MyModel
from config import _DNN_HIDDEN_UNITS

# read dataset
dataset = Dataset("../data/train/training.csv")

dataset.create_synthetic_features()

# create column features for further fit them into a model
dataset.create_feature_columns()

# preprocess it
dataset.preprocess()

#split on train/test
dataset.split_data()


# initialize model
model = MyModel()

# build it
model.build(dataset.inputs, dataset.sparse.values(), dataset.real.values(), _DNN_HIDDEN_UNITS)

# train
model.train(dataset.train_ds, dataset.val_ds)

# evaluate
model.evaluate(dataset.test_ds)