import keras
from keras import layers
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',None)

dataset_path='housing.data'

column_names = ['CRIM','ZN','INDUS','CHAS','NOX',
                'RM', 'AGE', 'DIS','RAD','TAX','PTRATION', 'B', 'LSTAT', 'MEDV']

raw_data=pd.read_csv(dataset_path,names=column_names,
                     na_values="?",comment='\t',
                     sep=' ',skipinitialspace=True)

dataset=raw_data.copy()



#training data portion

p=0.8

#train set
#randomly select 80% data as train set
trainSet=dataset.sample(frac=p,random_state=0)

#test set
testSet=dataset.drop(trainSet.index)

# fig,ax=plt.subplots()
# x=trainSet['RM']
# y=trainSet['MEDV']
# ax.scatter(x,y,edgecolors=(0,0,0))
# ax.set_xlabel('RM')
# ax.set_ylabel('MEDV')
# plt.show()


trainInput=trainSet['RM']
trainTarget=trainSet['MEDV']
testInput=testSet['RM']
testTarget=trainSet['MEDV']


model=keras.Sequential([
    layers.Dense(1,use_bias=True,input_shape=(1,))
])

#Adam optimizern
optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.01,
    beta_1=0.9,
    beta_2=0.99,
    epsilon=1e-05,
    amsgrad=False,
    name='Adam'
)

model.compile(loss='mse',optimizer=optimizer,metrics=['mae','mse'])


#stop training if less than 0.01 improvement for more than 100
n_idle_epochs=100
earlyStopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=n_idle_epochs,min_delta=0.01)

class NEPOCHLogger(tf.keras.callbacks.Callback):

    def __init__(self,per_epoch=100):
        self.seen=0
        self.per_epoch=per_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch%self.per_epoch==0:
            print('Epoch {}, loss {:.2f}, val_loss {:.2f}, mae {:.2f}, val_mae {:.2f}, mse {:.2f}, val_mse {:.2f}'\
              .format(epoch, logs['loss'], logs['val_loss'],logs['mae'], logs['val_mae'],logs['mse'], logs['val_mse']))

log_display=NEPOCHLogger(per_epoch=100)

#training loop

n_epochs=2000


'''
n_epochs: number of epochs
batch_size: number of samples per batch as the training is conducted with mini-batch optimization.
validation_split: keep a portion of training data for unbiased validation. 
verbose: set to 0 as we want a short summary and not all the details!!
callbacks: A callback is a tool to customize the behavior of the model during training, testing, etc.
'''
history=model.fit(
    trainInput, trainTarget, batch_size=256,
    epochs=n_epochs, validation_split = 0.1, verbose=0, callbacks=[earlyStopping,log_display])

print('keys:',history.history.keys())
