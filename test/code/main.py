from mydataset import Dataset
from model import MyModel

# read dataset
dataset = Dataset("training-1.csv")

# create column features for further fit them into a model
dataset.create_feature_columns()

# preprocess it
dataset.preprocess()

#split on train/test
dataset.split_data()


# initialize model
model = MyModel()

# build it
model.build(dataset.feature_columns)

# train
model.train(dataset.train_ds, dataset.val_ds)

# evaluate
model.evaluate(dataset.test_ds)






