from google.colab import drive
drive.mount('/content/drive')

#import the following libraries
import numpy as np
import regex as re
import tarfile,time,os,shutil,string
from tensorflow import keras
from itertools import groupby,chain
from tensorflow.keras import layers
import tensorflow as tf
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.layers.experimental.preprocessing import Rescaling,RandomFlip,RandomRotation
from keras.layers import Dense,Input,Conv2D,SeparableConv2D,add,Flatten,GlobalAveragePooling2D,Dropout,MaxPooling2D,Lambda
from keras.models import Model,Sequential
from keras.applications import InceptionResNetV2,ResNet152V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler,EarlyStopping,ReduceLROnPlateau
# To disable all logging output from TensorFlow 
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

#get_data function is used for fetching the training and testing data from given directory
def get_data(path,num_classes):
  #the value passed in split to get training and testing is as per the requirement mentioned
  #for caltech101 and caltech256 dataset it gets randomly 30 images from class for training and remaining for testing
  #fetch training data
  train=tf.keras.preprocessing.image_dataset_from_directory(
      directory=path,
      labels="inferred",
      label_mode="int",
      validation_split=0.6654,
      subset="training",
      seed=123,
      image_size=(224,224),
      batch_size=1)
  
  #fetch testing data
  test=tf.keras.preprocessing.image_dataset_from_directory(
      directory= path,
      labels="inferred",
      label_mode="int",
      validation_split=0.6654,
      subset="validation",
      seed=123,
      image_size=(224,224),
      batch_size=1)
  
  #noramlize the data  
  normalization = Rescaling(1./255)
  #data agumentation
  train_aug =  Sequential([Rescaling(1./255),
                                  RandomFlip("horizontal_and_vertical"),
                                  RandomRotation(0.2)])
  
  #normalize train and test data apply one hot encoding on labels and squeeze train and test to its required dimension
  #apply train_aug on train set and normalization only on test set
  #becomes equal to the shape of the input to the convolution network 
  train_n = train.map(lambda x, y: (tf.squeeze(train_aug(x)), tf.squeeze(tf.one_hot(y,num_classes))))
  test_n = test.map(lambda x, y: (tf.squeeze(normalization(x)), tf.squeeze(tf.one_hot(y,num_classes))))
  
  #the output of train_n and test_n is mapped dataset so we iterate over the train_n and test_n
  #and store the features and labels in list
  train_inputs=[] #for training features
  train_labels=[] #for training labels
  test_inputs=[] #for testing features
  test_labels=[] #for testing labels
  c=0
  tr=iter(train_n) #create train_n iterator
  tt=iter(test_n)  #create test_n iterator
  for i in range(0,len(test_n)):
    if c<len(train_n):
      i,j=next(tr)
      train_inputs.append(i)
      train_labels.append(j)
      c+=1
    i,j=next(tt)
    test_inputs.append(i)
    test_labels.append(j)

  #convert the list to tensor
  train_inputs=tf.convert_to_tensor(train_inputs, dtype=tf.float32)
  train_labels=tf.convert_to_tensor(train_labels, dtype=tf.float32)
  test_inputs=tf.convert_to_tensor(test_inputs, dtype=tf.float32)
  test_labels=tf.convert_to_tensor(test_labels, dtype=tf.float32)
  
  #free memory
  del train_n,test_n,tr,tt,train,test
  return train_inputs,train_labels,test_inputs,test_labels

#function to plot Loss and accuracy Curves on training set
def plotgraph(history,value,val):
  plt.figure(figsize=[8,6])
  plt.plot(history.history['loss'],'firebrick',linewidth=3.0)
  plt.plot(history.history['accuracy'],'turquoise',linewidth=3.0)
  plt.legend(['Training loss','Training Accuracy'],fontsize=18)
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Loss and Accuracy',fontsize=16)
  plt.title('Loss Curves and Accuracy Curves for {} for run {}'.format(value,val),fontsize=16)

#this function builds model by deep feature extraction, then feature fusion and then classifying by ANN as a classifier
def design_model(train,test,ytrain,ytest,epoch,nclas,value,val):
  #intialize time for training and testing
  tr=0.0
  tt=0.0

  #start time
  st=time.time()
  
  #input shape
  image_input = Input(shape=(None,None,3))
  #adding a lambda layer to reshape to size 224 by 224 
  #this make sure that image are converted to 224,224 before entering the model
  prep=Lambda(lambda x: tf.image.resize(x,(224, 224)))(image_input)
  #using pretained weights from imagnet for transfer learning 
  #using  InceptionResNetV2 model
  dmodel = InceptionResNetV2(include_top=False, weights='imagenet',input_shape=(224,224,3))(prep)
  dx = GlobalAveragePooling2D()(dmodel) 
  i_model = Model(image_input,dx)
  i_model.summary()
  
  #get the deep feature from InceptionResNetV2, feature extraction from InceptionResNetV2 model
  #get training deep features
  df_train_i=i_model.predict(train,
                             batch_size=32,
                             workers=50,
                             use_multiprocessing=True,
                             verbose=1)

  #get testing deep features
  df_test_i=i_model.predict(test,
                            batch_size=32,
                            workers=50,
                            use_multiprocessing=True,
                            verbose=1)
  
  #using pretained weights from imagnet for transfer learning
  #using ResNet152V2 model 
  vmodel = ResNet152V2(include_top=False, weights='imagenet',input_shape=(224,224,3))(prep)
  vx = GlobalAveragePooling2D()(vmodel) 
  r_model = Model(image_input,vx)
  r_model.summary()

  #get deep features from ResNet152V2, extraction of deep features
  #get training features  
  df_train_r=r_model.predict(train,
                             batch_size=32,
                             workers=50,
                             use_multiprocessing=True,
                             verbose=1)

  #get testing features
  df_test_r=r_model.predict(test,
                            batch_size=32,
                            workers=50,
                            use_multiprocessing=True,
                            verbose=1)
  
  #combining of deep features from both the InceptionResNetV2 and ResNet152V2 models, Feature fusion or feature combination
  #fusion of deep features extracted from InceptionResNetV2 and ResNet152V2 model and creating final_train and final_test set of features 
  final_train = tf.keras.layers.Concatenate()([df_train_i,df_train_r])
  final_test = tf.keras.layers.Concatenate()([df_test_i,df_test_r])

  #my own fully connected classifier(ANN)
  classifier = Sequential([Dense(4096, activation='relu', input_shape=final_train[0].shape),
                         Dropout(0.5),
                         Dense(nclas, activation='softmax')])

  #optimizer 
  opt = Adam(lr=1e-4, decay=1e-4 / 50)

  #compile my own classifier
  classifier.compile(loss="categorical_crossentropy", optimizer=opt,
                metrics=["accuracy"])

  #reduce_lr method is used to reduce the learning rate if the learning rate is stagnant or if there are no major improvements during training
  reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.001)
  #early stopping method is used to montior the loss if there are no significant reductions in loss then halts the training
  es = EarlyStopping(monitor='loss',patience=10)
  #fit the model
  history = classifier.fit(final_train,ytrain,epochs=epoch,
                          batch_size=32,
                          shuffle=True,
                          verbose=1,
                          workers=50,
                          use_multiprocessing=True,  
                          callbacks=[reduce_lr,es]) 
  
  #total time for training
  tr=time.time()-st
  #plot the graph of accuracy and loss for training
  plotgraph(history,value,val)
  #start time for testing
  st=time.time()
  #evalute the model
  (loss, accuracy) = classifier.evaluate(final_test,ytest, batch_size=32,verbose=1,workers=50,
                          use_multiprocessing=True)
  
  #total time for testing
  tt=time.time()-st

  #training and testing accuracy
  train_acc=history.history["accuracy"][-1:][0]
  test_acc=accuracy

  #free memory
  del i_model,df_train_i,df_test_i,r_model,df_train_r,df_test_r,final_train,final_test,classifier,history
  
  return tr,tt,train_acc,test_acc

#function to process Caltech101 dataset
def caltech101():
  #hyper parameters for caltech101
  epoch=500
  nclas=102 
  st=time.time()
  #get the data
  cx_train,cy_train,cx_test,cy_test=get_data('/content/drive/MyDrive/Caltech101',nclas)
  print("length of training data is {} and length of testing data is {}".format(len(cx_train),len(cx_test)))
  print("time required to preprocess and get the data {}".format(np.round(time.time()-st,4)))

  #intialize time for training and testing
  ctr_time=0.0
  ctt_time=0.0
  #intialize empty list for training and testing accuray
  ctr_acc=[]
  ctt_acc=[]
  #this loop runs for 3 runs gets the training and testing time and training and testing accuracy
  for i in range(0,3):
    ctr, ctt, ctrain_acc, ctest_acc = design_model(cx_train,cx_test,cy_train,cy_test,epoch,nclas,"Caltech101",i+1)
    #add the time for training and testing
    ctr_time+=ctr
    ctt_time+=ctt
    #append the training and testing accuracy
    ctr_acc.append(ctrain_acc)
    ctt_acc.append(ctest_acc)

  #calculate the average of training accuracy of 3 runs
  cavg_tr=tf.add_n(ctr_acc)/3
  cavg_tr=np.float(cavg_tr)

  #calculate the average of testing accuracy of 3 runs
  cavg_tt=tf.add_n(ctt_acc)/3
  cavg_tt=np.float(cavg_tt)

  #write output in a Caltech101_output.txt file 
  file1=open("/content/drive/MyDrive/Caltech101_output1.txt","w")
  file1.write("Total training time: {}\n".format(ctr_time))
  file1.write("Total testing time: {}\n".format(ctt_time))
  file1.write("training accuracy: {}\n".format(ctr_acc))
  file1.write("testing accuracy: {}\n".format(ctt_acc))
  file1.write("average training accuracy: {}\n".format(cavg_tr))
  file1.write("average testing accuracy: {}\n".format(cavg_tt))
  file1.close()

  #free memory 
  del cx_train,cy_train,cx_test,cy_test

#function to process cifar 10 dataset
def cifar10data():
  st=time.time()
  # load dataset
  (train10X, train10y), (test10X, test10y) = cifar10.load_data()
  
  #normalize the data
  X_train_mean = np.mean(train10X, axis=(0,1,2))
  X_train_std = np.std(train10X, axis=(0,1,2))
  train10X = (train10X - X_train_mean) / X_train_std
  test10X = (test10X - X_train_mean) / X_train_std

  #data augumentation for training data
  width_shift = 3/32
  height_shift = 3/32
  flip = True

  train_datagen = ImageDataGenerator(
      horizontal_flip=flip,
      width_shift_range=width_shift,
      height_shift_range=height_shift)
  
  #apply data augumentation on train data
  train_datagen.fit(train10X)

  #one hot encoding
  Y_train10 = tf.squeeze(tf.one_hot(train10y,10))
  Y_test10 = tf.squeeze(tf.one_hot(test10y,10))
  
  # summarize loaded dataset
  print('Train Cifar10: X=%s, y=%s' % (train10X.shape, Y_train10.shape))
  print('Test Cifar10: X=%s, y=%s' % (test10X.shape, Y_test10.shape))
  print("time required to preprocess and get the data {}".format(np.round(time.time()-st,4)))
  
  #hyperparameters for cifar10
  epoch=500
  nclas=10

  #initialize the training and testing time
  citr_time=0.0
  citt_time=0.0

  #intialize the training and testing accuracy list
  citr_acc=[]
  citt_acc=[]

  #this loop runs for 3 runs gets the training and testing time and training and testing accuracy
  for i in range(0,3):
    citr, citt, citrain_acc, citest_acc=design_model(train10X,test10X,Y_train10,Y_test10,epoch,nclas,"Cifar10",i+1)
    #add the time for training and testing
    citr_time+=citr
    citt_time+=citt
    #append the training and testing accuracy
    citr_acc.append(citrain_acc)
    citt_acc.append(citest_acc)

  #calculate the average of training accuracy of 3 runs
  ciavg_tr=tf.add_n(citr_acc)/3
  ciavg_tr=np.float(ciavg_tr)

  #calculate the average of testing accuracy of 3 runs
  ciavg_tt=tf.add_n(citt_acc)/3
  ciavg_tt=np.float(ciavg_tt)

  #write output in a Cifar10_output.txt file
  file1=open("/content/drive/MyDrive/Cifar10_output1.txt","w")
  file1.write("Total training time: {}\n".format(citr_time))
  file1.write("Total testing time: {}\n".format(citt_time))
  file1.write("training accuracy: {}\n".format(citr_acc))
  file1.write("testing accuracy: {}\n".format(citt_acc))
  file1.write("average training accuracy: {}\n".format(ciavg_tr))
  file1.write("average testing accuracy: {}\n".format(ciavg_tt))
  file1.close()

  #free memorry
  del train10X,test10X,Y_train10,Y_test10

#function to process caltech256 10 dataset
def caltech256():
  #hyper parameters for caltech256
  epoch=500
  nclas=257

  st=time.time()
  #get data
  c_train,y_train,c_test,y_test=get_data('/content/drive/MyDrive/Caltech256',nclas)
  print("length of training data is {} and length of testing data is {}".format(len(c_train),len(c_test)))
  print("time required to preprocess and get the data {}".format(np.round(time.time()-st,4)))

  #initialize the training and testing time
  catr_time=0.0
  catt_time=0.0
  #intialize empty list for training and testing accuray
  catr_acc=[]
  catt_acc=[]

  #this loop runs for 3 runs gets the training and testing time and training and testing accuracy
  for i in range(0,3):
    catr, catt, catrain_acc, catest_acc=design_model(c_train,c_test,y_train,y_test,epoch,nclas,"Caltech256",i+1)
    #add the time for training and testing
    catr_time+=catr
    catt_time+=catt
    #append the training and testing accuracy
    catr_acc.append(catrain_acc)
    catt_acc.append(catest_acc)

  #calculate the average of training accuracy of 3 runs
  caavg_tr=tf.add_n(catr_acc)/3
  caavg_tr=np.float(caavg_tr)

  #calculate the average of testing accuracy of 3 runs
  caavg_tt=tf.add_n(catt_acc)/3
  caavg_tt=np.float(caavg_tt)

  #write output in a Caltech256_output.txt file
  file1=open("/content/drive/MyDrive/Caltech256_output1.txt","w")
  file1.write("Total training time: {}\n".format(catr_time))
  file1.write("Total testing time: {}\n".format(catt_time))
  file1.write("training accuracy: {}\n".format(catr_acc))
  file1.write("testing accuracy: {}\n".format(catt_acc))
  file1.write("average training accuracy: {}\n".format(caavg_tr))
  file1.write("average testing accuracy: {}\n".format(caavg_tt))
  file1.close()

  #free memory
  del c_train,y_train,c_test,y_test

#after output generated from above functions is completed this function read txt file
#print the results stored in the text files
def print_results():
  #read caltech101 output
  f = open("/content/drive/MyDrive/Caltech101_output1.txt", "r")
  caltech101_op = f.readlines()
  f.close()

  #read cifar10 output
  f1 = open("/content/drive/MyDrive/Cifar10_output1.txt", "r")
  cifar10_op = f1.readlines()
  f1.close()
  
  #read caltech256 output
  f2 = open("/content/drive/MyDrive/Caltech256_output1.txt", "r")
  caltech256_op = f2.readlines()
  f2.close()

  #loop to replace few sumbols and split to get the desired numbers or output 
  for i in range(len(caltech101_op)):
    caltech101_op[i]=caltech101_op[i].replace('\n','').split(':')[1]
    cifar10_op[i]=cifar10_op[i].replace('\n','').split(':')[1]
    caltech256_op[i]=caltech256_op[i].replace('\n','').split(':')[1]
    if i==2 or i==3:
      caltech101_op[i]=caltech101_op[i].replace("[","").replace("]","").replace(',','').split(" ")[1:]
      cifar10_op[i]=cifar10_op[i].replace("[","").replace("]","").replace(',','').split(" ")[1:]
      caltech256_op[i]=caltech256_op[i].replace("[","").replace("]","").replace(',','').split(" ")[1:]

  #create list so results are fetched according to the list
  data_list=["Caltech101","Cifar10","Caltech256"]
  #if data_list[k]="Caltec101" then set value as caltech101_op similarly for other datasets
  #caltech101_op list similary for other two  datasets
  #first ie at zero and first positon conatins single element total time for training and testing
  #positon 2 and 3 contains a sublist of training and testing accuracy for each runs so for loop is added to iterate over the sublist
  #second last and last elements are average training and testing accuray
  for k in range(0,len(data_list)):
    c=0
    print("<<<<<<<<<<<<<<<<<<<<{}>>>>>>>>>>>>>>>>>>>>".format(data_list[k]))
    if data_list[k]=="Caltech101":
      value=caltech101_op
    if data_list[k]=="Cifar10":
      value=cifar10_op
    if data_list[k]=="Caltech256":
      value=caltech256_op
    for i in range(0,len(caltech101_op)):
      if i==2:
        for j in range(0,3):
          print("training accuracy for {} for run {} is : {}%".format(data_list[k],j+1,np.round(float(value[i][j])*100,2)))
      if i==3:
        for j in range(0,3):
          print("testing accuracy for {} for run {} is : {}%".format(data_list[k],j+1,np.round(float(value[i][j])*100,2)))
    #print the total training and testing time 
    print("Total training time for {} is {}".format(data_list[k],np.round(float(value[0]),4)))
    print("Average of 3 runs training accuracy for {} is : {}%".format(data_list[k],np.round(float(value[4])*100,2)))
    print("Total testing time for {} is {}".format(data_list[k],np.round(float(value[1]),4)))
    print("Average of 3 runs testing accuracy for {} is : {}%".format(data_list[k],np.round(float(value[5])*100,2)))
    print("------------------------------------------")

  print("<<<<<<<<<<<<<<<<<<<<TOP1 training Accuracy>>>>>>>>>>>>>>>>>>>>")
  print("Top1 training Accuracy for Caltech101 is: {}".format(np.round(float(caltech101_op[4])*100,2)))
  print("Top1 training Accuracy for Cifar10 is: {}".format(np.round(float(cifar10_op[4])*100,2)))
  print("Top1 training Accuracy for Caltech256 is: {}".format(np.round(float(caltech256_op[4])*100,2)))
  print("Average of Top1 Training Accuracy is: {}".format((np.round(float(caltech101_op[4])*100,2)+np.round(float(cifar10_op[4])*100,2)+np.round(float(caltech256_op[4])*100,2))/3))
  print("------------------------------------------")

  print("<<<<<<<<<<<<<<<<<<<<TOP1 testing Accuracy>>>>>>>>>>>>>>>>>>>>")
  print("Top1 testing Accuracy for Caltech101 is: {}".format(np.round(float(caltech101_op[5])*100,2)))
  print("Top1 testing Accuracy for Cifar10 is: {}".format(np.round(float(cifar10_op[5])*100,2)))
  print("Top1 testing Accuracy for Caltech256 is: {}".format(np.round(float(caltech256_op[5])*100,2)))
  print("Average of Top1 Testing Accuracy is: {}".format((np.round(float(caltech101_op[5])*100,2)+np.round(float(cifar10_op[5])*100,2)+np.round(float(caltech256_op[5])*100,2))/3))
  print("------------------------------------------")

caltech101()

cifar10data()

caltech256()

print_results()

