from mydataset import Dataset
from model import MyModel

# read dataset
dataset = Dataset("training-1.csv")
# preprocess it
dataset.preprocess()
#split on train/test
dataset.split_data()

# initialize model
model = MyModel(dataset.x_train.shape[1])
# build it
model.build()
# train
model.train(dataset.x_train, dataset.y_train)

# evaluate
model.evaluate(dataset.x_test, dataset.y_test)
