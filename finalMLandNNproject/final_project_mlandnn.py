from google.colab import drive
drive.mount('/content/drive')
import numpy as np
import time,os,re,shutil
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense,Activation,Flatten,Input
from keras.models import Model
from keras.utils import np_utils
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import VGG16,decode_predictions
from keras.datasets import cifar10,cifar100
# To disable all logging output from TensorFlow 
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

#oslem checker using python to see the shapes of matrix when implemted in cuda c 
def oselm(df_train,df_test,Y_train,Y_test):
  #size of columns and input neurons number of neurons used is 1k
  #hidden_size = 1000
  #input_weights = np.random.normal(size=[1024,hidden_size])

  #divide into batch
  df_tr=np.array_split(df_train,5)
  df_try=np.array_split(Y_train,5)
  #print("Shape of batches train{} train label{}".format(np.shape(df_tr[0]),np.shape(df_try[0])))
  for i in range(0,5):
    if i==0:
      a = np.dot(df_tr[i],input_weights)
      #print("Shape of dot product {}".format(np.shape(a)))
      a = np.maximum(a, 0, a) # ReLU
      #print("Shape of relu {}".format(np.shape(a)))
      at=np.transpose(a)
      #print("Shape of transpose {}".format(np.shape(at)))
      M=np.linalg.pinv(np.dot(at,a))
      #print("shape of dot {}".format(np.shape(np.dot(at,a))))
      #print("Shape of M {}".format(np.shape(M)))
      beta=np.dot(np.linalg.pinv(a),df_try[i])
      #print("Shape of beta {}".format(np.shape(beta))) 
      #print("shape of inverse of a{}".format(np.shape(np.linalg.pinv(a))))

    else:
      a = np.dot(df_tr[i],input_weights)
      a = np.maximum(a, 0, a) # ReLU
      at=np.transpose(a)
      M=M-(M@at@np.linalg.inv(np.eye(10000)+a@M@at)@a@M)  
      #print("M {}".format(np.shape(M)))
      beta = beta+(M@at@(df_try[i]-a@beta)) 
      #print("beta {}".format(np.shape(beta)))
  a = np.dot(df_test,input_weights)
  a = np.maximum(a, 0, a) # ReLU
  y=np.dot(a,beta)
  total = y.shape[0]
  #calcualte accuracy
  correct=0
  for i in range(total):
    predicted = np.argmax(y[i])
    test = np.argmax(Y_test[i])
    correct = correct + (1 if predicted == test else 0)
  #print('Accuracy: {:f}'.format((correct/total)*100))

#extracting deep features from vgg16 and ResNet152V2
def vggandres(trainX,testX,trainy,testy,val):
  
  # input shape for vgg16 model and extracting deep feature from block5 conv layer 
  image_input = Input(shape=(32,32, 3))
  #using pretained weights from imagnet for transfer learning 
  model = VGG16(input_tensor=image_input,include_top=False, weights='imagenet') 
  last_layer = model.get_layer('block5_pool').output
  x= Flatten(name='flatten')(last_layer) 
  custom_vgg_model2 = Model(image_input,x)  
  # freeze all the layers except the dense layers
  for layer in custom_vgg_model2.layers[:-6]:
    layer.trainable = False
  optim = tf.keras.optimizers.RMSprop(learning_rate=0.00001)
  custom_vgg_model2.compile(loss='categorical_crossentropy',optimizer=optim,metrics=['accuracy'])
  #deep feature from vgg16
  df_train_vgg=custom_vgg_model2.predict(trainX,batch_size=32,verbose=1)
  df_test_vgg=custom_vgg_model2.predict(testX,batch_size=32,verbose=1)

  #design of resnet152V2 model and extracting deep feature from conv5_block_3_2_relu layer
  model=tf.keras.applications.ResNet152V2(include_top=False,weights="imagenet",input_tensor=image_input)
  last_layer = model.get_layer('conv5_block3_2_relu').output
  x= Flatten(name='flatten')(last_layer) 
  custom_res_model2 = Model(image_input,x)  
  optim = tf.keras.optimizers.RMSprop(learning_rate=0.00001)
  custom_res_model2.compile(loss='categorical_crossentropy',optimizer=optim,metrics=['accuracy'])
  #deep feature from resnet152V2
  df_train_res=custom_res_model2.predict(trainX,batch_size=32,verbose=1)
  df_test_res=custom_res_model2.predict(testX,batch_size=32,verbose=1)

  #combining of deep features from both the models
  final_train = tf.keras.layers.Concatenate()([df_train_vgg,df_train_res])
  final_test = tf.keras.layers.Concatenate()([df_test_vgg,df_test_res])

  #save deep features in text file to run cuda
  np.savetxt("/content/drive/MyDrive/df_train_{}.txt".format(val),final_train) 
  np.savetxt("/content/drive/MyDrive/df_test_{}.txt".format(val),final_test) 
  np.savetxt("/content/drive/MyDrive/df_trainy_{}.txt".format(val),trainy) 
  np.savetxt("/content/drive/MyDrive/df_testy_{}.txt".format(val),testy) 

  #just to check the dimension of how it will look in cuda the below oselm checker is used
  #oselm(final_train,final_test,trainy,testy)

def main():
  # load dataset
  (train10X, train10y), (test10X, test10y) = cifar10.load_data()
  #normalize the data
  X_train_mean = np.mean(train10X, axis=(0,1,2))
  X_train_std = np.std(train10X, axis=(0,1,2))
  train10X = (train10X - X_train_mean) / X_train_std
  test10X = (test10X - X_train_mean) / X_train_std
  #one hot encoding
  Y_train10 = np_utils.to_categorical(train10y,10)
  Y_test10 = np_utils.to_categorical(test10y,10)
  # summarize loaded dataset
  print('Train Cifar10: X=%s, y=%s' % (train10X.shape, Y_train10.shape))
  print('Test Cifar10: X=%s, y=%s' % (test10X.shape, Y_test10.shape))
  st=time.time()  
  vggandres(train10X,test10X,Y_train10,Y_test10,"cifar10")
  print("time extracting deep features for cifar10 {}".format(time.time()-st))

  # load dataset
  (train100X, train100y), (test100X, test100y) = cifar100.load_data()
  #normalize dataset
  X_train_mean = np.mean(train100X, axis=(0,1,2))
  X_train_std = np.std(train100X, axis=(0,1,2))
  train100X = (train100X - X_train_mean) / X_train_std
  test100X = (test100X - X_train_mean) / X_train_std
  #one hot encoding
  Y_train100 = np_utils.to_categorical(train100y,100)
  Y_test100 = np_utils.to_categorical(test100y,100)
  #summarize the dataset
  print('Train Cifar100: X=%s, y=%s' % (train100X.shape, Y_train100.shape))
  print('Test Cifar100: X=%s, y=%s' % (test100X.shape, Y_test100.shape))
  st=time.time()
  vggandres(train100X,test100X,Y_train100,Y_test100,"cifar100") 
  print("time extracting deep features for cifar100 {}".format(time.time()-st))

"""run this main function three times for 3 runs after first run and extracting deep features run the cuda code do it same for three times"""

main()

"""**Below is the code for cifar 10 in cuda for oselm i have created 5 batches of 10 by jumping pointers from 0 to 4 in cuda c as you run the above main function for first time run below code for cifar 10 and below cifar 10 there is code for cifar 100 so after running main function for first time run both the cuda codes then run main function again whihc overwrite the text files and then run the cuda code again for 3 times to average of 3 accuracy and then top1 accuracy**

**CIFAR10 cuda c code**
"""

# Commented out IPython magic to ensure Python compatibility.
!pip install git+git://github.com/andreinechaev/nvcc4jupyter.git
# %load_ext nvcc_plugin

# Commented out IPython magic to ensure Python compatibility.
# %%cuda --name my_curand.cu
# #include <stdio.h>
# #include <cstdlib>
# #include <vector>
# #include <random>
# #include <algorithm>
# #include <math.h>
# #include <cuda.h>
# #include <stdlib.h> 
# #include <time.h> 
# #include <iostream>
# #include <ctime>
# #include <fstream>
# #include <string>
# #include <cuda_runtime.h>
# #include <device_launch_parameters.h>
# #include <cublas_v2.h>
# #include <cuda_runtime_api.h>
#  
# using namespace std;
# #define blocksize 8
#  
# //function to generate input weights for hidden neurons
# float random_float(float min, float max){
#      return ((float)rand() / RAND_MAX) * (max - min) + min;
# }
# 
# //for calculating inverse
# __global__ void nodiag_normalize(float *A,float *I, int n, int i){
#     int x = blockIdx.x * blockDim.x + threadIdx.x;
#  	  int y = blockIdx.y * blockDim.y + threadIdx.y;
#  	  if(x < n && y < n){
#          if(x == i && x!=y){
#              I[x*n + y] /= A[i*n + i];
#              A[x*n + y] /= A[i*n + i];
#          }
#      }	
# }
# 
# //for calculating inverse
# __global__ void diag_normalize(float *A,float *I, int n, int i){
#     int x = blockIdx.x * blockDim.x + threadIdx.x;
#  	  int y = blockIdx.y * blockDim.y + threadIdx.y;
#  	  if(x < n && y < n){
#          if(x == y && x == i){
#              I[x*n + y] /= A[i*n + i];
#              A[x*n + y] /= A[i*n + i];
#          }
#      }
# }
# 
# //for calculating inverse
# __global__ void gaussjordan(float *A,float *I, int n, int i){
#      int x = blockIdx.x * blockDim.x + threadIdx.x;
#      int y = blockIdx.y * blockDim.y + threadIdx.y;
#      if(x < n && y < n){
#          if(x != i){
#              I[x*n + y] -= I[i*n + y] * A[x*n + i];
#              if(y != i){
#                  A[x*n + y] -= A[i*n + y] * A[x*n + i];
#              }
#          }
#      }
# }
# 
# //for calculating inverse
#  __global__ void set_zero(float *A,float *I, int n, int i){
#      int x = blockIdx.x * blockDim.x + threadIdx.x;
#  	  int y = blockIdx.y * blockDim.y + threadIdx.y;
#      if(x < n && y < n){
#          if(x != i){
#              if(y == i){
#                  A[x*n + y] = 0;
#              }
#          }
#      }
# }
#  
# //read data from text file
# void matrix_read(float *L,const char *path,int m,int n){
#   FILE *fp;
#  	int row, col;
#  
#  	fp = fopen(path, "r");//open output file
#  	if (fp == NULL)//open failed
#  		return;
#  
#  	for (row = 0; row < m; row++){
#  		for (col = 0; col < n; col++)
#  		if (fscanf(fp, "%f,", &L[row * n + col]) == EOF) break;//read data
#  
#  		if (feof(fp)) break;//if the file is over
#  	}
#  
#  	fclose(fp);//close file 
# }
#  
# //function to generate identity matrix
# float identity(float *a,int num){
#      int row,col;
#      for(row = 0; row < num; row++){
#          for(col = 0; col < num; col++){
#              if(row == col){
#                  a[row*num+col]=1.0f;
#              }
#              else{
#                  a[row*num+col]=0.0f;
#              }
#          }
#     }
#     return 0;
# }
#  
# //relu activation function
# float relu(float *a,int m,int p){
#      for(int i=0;i<m;i++){
#          for(int j=0;j<p;j++){
#              if(a[i*p+j]>0){
#                  a[i*p+j]=a[i*p+j];
#                  //printf("no change");
#                  }
#              else{
#                  a[i*p+j]=0.0;
#                  //printf("do the change");
#              }
#          }
#      }
#      return 0;
# }
# 
# //this for checking the intermediate ouput by taking it into text file 
# void savetofile(float *A, string s, int n, int h){
#      std::ofstream plik;
#      plik.open(s);
#      for(int i = 0; i<n; i++){
#          for(int j = 0; j<h; j++){
#              plik << A[i*h + j] << "\t";
#          }
#          plik << endl;
#      }
#      plik.close();
# }
#  
# //oselm i have created 5 batches from 0 to 4  manually and pointer locations are jumped
# //to access the chunks of data for processing in batch
# int main(){
#     //seed to get random data 
#     srand(time(0));
#     float alpha = 1.0f;
#  	  float beta = 0.0f;
#  
#      //cifar 10 params
#      int n=50000,k=1024,m=10000,p=1000,l=10;
#      
#     //dimension for calculating inverse 
#     dim3 threadsPerBlock(blocksize, blocksize);
#     dim3 numBlocks((p + blocksize - 1) / blocksize, (p + blocksize - 1) / blocksize);
#      
#     //generate input hidden neurons
#     float *cpu_h,*gpu_h;
#     cudaMallocHost((void **) &cpu_h, sizeof(float)*k*p);
#     cudaMalloc((void **) &gpu_h, sizeof(float)*k*p);
#     //generate input weights for hidden neurons
#     for (int i = 0; i < k; ++i) {
#         for (int j = 0; j < p; ++j) {
#             cpu_h[i * p + j] = random_float(-2.0, 2.0);    
#         }
#     }
#     //copy to gpu host is cpu and device is gpu
#     cudaMemcpy(gpu_h,cpu_h, sizeof(float)*k*p, cudaMemcpyHostToDevice);
#     //savetofile(cpu_h,"H.txt",k,p); 
# 
#     //identity matrix generation and storing on gpu and cpu
#     float *cpu_I1,*cpu_I2,*gpu_I1,*gpu_I2;
#     cudaMallocHost((void **) &cpu_I1, sizeof(float)*p*p);
#     cudaMallocHost((void **) &cpu_I2, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_I1, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpu_I2, sizeof(float)*m*m);
#  
#     identity(cpu_I1,p);
#     //savetofile(cpu_I1,"I1.txt",p,p); 
#     identity(cpu_I2,m);
#     //savetofile(cpu_I2,"I2.txt",m,m); 
#     cudaMemcpy(gpu_I1,cpu_I1, sizeof(float)*p*p, cudaMemcpyHostToDevice);
#     cudaMemcpy(gpu_I2,cpu_I2, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#  
#     //get train data and label
#     float *cpu_tr,*cpu_trl,*gpu_tr,*gpu_trl;
#     cudaMallocHost((void **) &cpu_tr, sizeof(float)*n*k);//train data
#     cudaMallocHost((void **) &cpu_trl, sizeof(float)*n*l);//train label
#  
#     cudaMalloc((void **) &gpu_tr, sizeof(float)*n*k);
#     cudaMalloc((void **) &gpu_trl, sizeof(float)*n*l);
#  
#     const char path[] = "/content/drive/MyDrive/df_train_cifar10.txt";
#     matrix_read(cpu_tr,path,n,k);
#     const char path1[] = "/content/drive/MyDrive/df_trainy_cifar10.txt";
#     matrix_read(cpu_trl,path1,n,l);
#      
#     cudaMemcpy(gpu_tr,cpu_tr, sizeof(float)*n*k, cudaMemcpyHostToDevice);
#     cudaMemcpy(gpu_trl,cpu_trl, sizeof(float)*n*l, cudaMemcpyHostToDevice);
#  
#     //get test data and label
#     float *cpu_tt,*cpu_ttl,*gpu_tt,*gpu_ttl;
#     cudaMallocHost((void **) &cpu_tt, sizeof(float)*m*k);//test data
#     cudaMallocHost((void **) &cpu_ttl, sizeof(float)*m*l);//test label
#  
#     cudaMalloc((void **) &gpu_tt, sizeof(float)*m*k);
#     cudaMalloc((void **) &gpu_ttl, sizeof(float)*m*l);
#  
#     const char path2[] = "/content/drive/MyDrive/df_test_cifar10.txt";
#     matrix_read(cpu_tt,path2,m,k);
#     const char path3[] = "/content/drive/MyDrive/df_testy_cifar10.txt";
#     matrix_read(cpu_ttl,path3,m,l);
#  
#     cudaMemcpy(gpu_tt,cpu_tt, sizeof(float)*m*k, cudaMemcpyHostToDevice);
#     cudaMemcpy(gpu_ttl,cpu_ttl, sizeof(float)*m*l, cudaMemcpyHostToDevice);
#  
#     //to store output of dot product and  activation function
#     float *cpu_x,*gpu_x; 
#     cudaMallocHost((void **) &cpu_x, sizeof(float)*m*p);
#     cudaMalloc((void **) &gpu_x, sizeof(float)*m*p);
#  
#     //for storing inverse
#     float *cpu_m,*gpu_m;
#     cudaMallocHost((void **) &cpu_m, sizeof(float)*p*p); 
#     cudaMalloc((void **) &gpu_m, sizeof(float)*p*p); 
#     
#     cublasHandle_t handle;   
#     //maually implemented oslem and divdied in batches of 5 from 0 to 4
#     ///////////////////////batch1
#     //activation,relu
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,m,k,&alpha,gpu_h,p,gpu_tr+0*m*k,k,&beta,gpu_x,p); 
#     cublasDestroy(handle); 
#      
#     cudaMemcpy(cpu_x,gpu_x, sizeof(float)*m*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_x,"mamul1.txt",m,p); 
#     relu(cpu_x,m,p); //output of activation function
#     //savetofile(cpu_x,"relu.txt",m,p);
#    
#     //copy X*XT
#     float *gpu_r;
#     cudaMalloc((void **) &gpu_r, sizeof(float)*m*p);
#     cudaMemcpy(gpu_r,cpu_x, sizeof(float)*m*p, cudaMemcpyHostToDevice);    
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,p,m,p,&alpha,gpu_r,p,gpu_r,m,&beta,gpu_m,p);
#     cublasDestroy(handle); 
#     cudaMemcpy(cpu_m,gpu_m, sizeof(float)*p*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_m,"transpose.txt",p,p);
#      
#     float *cpu_mi,*gpu_mi;
#     cudaMalloc((void **) &gpu_mi, sizeof(float)*p*p); 
#     cudaMallocHost((void **) &cpu_mi, sizeof(float)*p*p);
#     //inverse of x*XT
#     for (int i = 0; i<p; i++){
#         nodiag_normalize << <numBlocks, threadsPerBlock >> >(gpu_x,gpu_I1,p,i);
#  		    diag_normalize << <numBlocks, threadsPerBlock >> >(gpu_x,gpu_I1,p,i);
#  		    gaussjordan << <numBlocks, threadsPerBlock >> >(gpu_x,gpu_I1,p,i);
#  		    set_zero << <numBlocks, threadsPerBlock >> >(gpu_x,gpu_I1,p,i);
#      }
#     cudaMemcpy(cpu_mi,gpu_I1, sizeof(float)*p*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_mi,"inverse.txt",p,p);
#     cudaMemcpy(gpu_mi,cpu_mi, sizeof(float)*p*p, cudaMemcpyHostToDevice);
#  
#     //Xtrain *Xt
#     float *cpu_xty,*gpu_xty;
#     cudaMallocHost((void **) &cpu_xty, sizeof(float)*p*l);
#     cudaMalloc((void **) &gpu_xty, sizeof(float)*p*l);
#  
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,l,p,m,&alpha,gpu_trl+0*m*l,l,gpu_x,p,&beta,gpu_xty,l);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_xty,gpu_xty, sizeof(float)*p*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_xty,"label.txt",p,l);
#    
#     float *cpu_B,*gpu_B;
#     cudaMallocHost((void **) &cpu_B, sizeof(float)*p*l);
#     cudaMalloc((void **) &gpu_B, sizeof(float)*p*l);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,p,p,&alpha,gpu_xty,l,gpu_I1,p,&beta,gpu_B,l);  
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_B,gpu_B, sizeof(float)*p*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_B,"beta.txt",p,l);
#      
#     ////////////////////////////////////batch2
#     //activation,relu
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,m,k,&alpha,gpu_h,p,gpu_tr+1*m*k,k,&beta,gpu_x,p); 
#     cublasDestroy(handle); 
#     cudaMemcpy(cpu_x,gpu_x, sizeof(float)*m*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_x,"mamul2.txt",m,p); 
#     relu(cpu_x,m,p); //output of activation function
#     //savetofile(cpu_x,"relu2.txt",m,p);
#      
#     //calc a3,m*xt
#     float *cpu_a3,*gpu_a3;
#     cudaMallocHost((void **) &cpu_a3, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a3, sizeof(float)*m*m);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,p,p,&alpha,gpu_x,p,gpu_mi,p,&beta,gpu_a3,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a3,gpu_a3, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a3,"A3.txt",m,m);
#      
#     //calc a1,x*m
#     float *cpu_a1,*gpu_a1;
#     cudaMallocHost((void **) &cpu_a1, sizeof(float)*m*p);
#     cudaMalloc((void **) &gpu_a1, sizeof(float)*m*p);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,m,p,&alpha,gpu_mi,p,gpu_x,p,&beta,gpu_a1,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a1,gpu_a1, sizeof(float)*m*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a1,"A1.txt",m,p);
#      
#     //calc a2,a1*xt
#     float *cpu_a2,*gpu_a2;
#     cudaMallocHost((void **) &cpu_a2, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a2, sizeof(float)*m*m);
#  
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,m,p,&alpha,gpu_x,p,gpu_a1,p,&beta,gpu_a2,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a2,gpu_a2, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a2,"A2.txt",m,m);
#      
#     //addition
#     for(int row=0;row<m;row++){
#         for(int col=0;col<m;col++){
#             cpu_a2[row*m+col]=cpu_a2[row*m+col]+cpu_I2[row*m+col];
#         }
#     }
#     //savetofile(cpu_a2,"Add.txt",m,m);
#      
#     //inverse
#     float *cpu_a2i,*gpu_a2i;
#     cudaMallocHost((void **) &cpu_a2i, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a2i, sizeof(float)*m*m);
#     cudaMemcpy(gpu_a2i,cpu_a2, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#     for(int i = 0; i<m; i++){
#         nodiag_normalize <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#         diag_normalize <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#  		    gaussjordan <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#  		    set_zero <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#     }
#     cudaMemcpy(cpu_a2i,gpu_I2, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a2i,"XMXTinverse.txt",m,m);
#      
#     //calc A4,A3*A2
#     float *cpu_a4,*gpu_a4,*gpuu_a2;
#     cudaMallocHost((void **) &gpuu_a2, sizeof(float)*m*m);
#     cudaMemcpy(gpuu_a2,cpu_a2i, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#  
#     cudaMallocHost((void **) &cpu_a4, sizeof(float)*p*m);
#     cudaMalloc((void **) &gpu_a4, sizeof(float)*p*m);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,p,m,&alpha,gpuu_a2,m,gpu_a3,m,&beta,gpu_a4,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a4,gpu_a4, sizeof(float)*p*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a4,"A4.txt",p,m);
#  
#     //calc A5,X*A4
#     float *cpu_a5,*gpu_a5,*cpuu_a5,*gpuu_a5;
#     cudaMallocHost((void **) &cpu_a5, sizeof(float)*p*p);
#     cudaMallocHost((void **) &cpuu_a5, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpu_a5, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpuu_a5, sizeof(float)*p*p);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,p,m,&alpha,gpu_x,p,gpu_a4,m,&beta,gpu_a5,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a5,gpu_a5, sizeof(float)*p*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a5,"A5.txt",p,p);
#  
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,p,p,&alpha,gpu_mi,p,gpu_a5,p,&beta,gpuu_a5,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpuu_a5,gpuu_a5, sizeof(float)*p*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpuu_a5,"finalA5.txt",p,p);
#              
#     for(int row=0;row<p;row++){
#         for(int col=0;col<p;col++){
#             cpu_mi[row*p+col]=cpu_mi[row*p+col]-cpuu_a5[row*p+col];
#         }
#     }
#     //savetofile(cpu_mi,"SubA5.txt",p,p);
#     cudaMemcpy(gpu_mi,cpu_mi, sizeof(float)*p*p, cudaMemcpyHostToDevice);
#      
#     //calc A6
#     float *cpu_a6,*gpu_a6;
#     cudaMallocHost((void **) &cpu_a6, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpu_a6, sizeof(float)*m*l);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,m,p,&alpha,gpu_B,l,gpu_x,p,&beta,gpu_a6,l); 
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a6,gpu_a6, sizeof(float)*m*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a6,"A6.txt",m,l);
#      
#     float *cpu_ty,*gpu_ty;
#     cudaMallocHost((void **) &cpu_ty, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpu_ty, sizeof(float)*m*l);
#     cudaMemcpy(gpu_ty,cpu_trl+1*m*l, sizeof(float)*m*l, cudaMemcpyHostToDevice);
#     cudaMemcpy(cpu_ty,gpu_ty, sizeof(float)*m*l, cudaMemcpyDeviceToHost);
#  
#     for(int row=0;row<m;row++){
#         for(int col=0;col<l;col++){
#             cpu_ty[row*l+col]=cpu_ty[row*l+col]-cpu_a6[row*l+col];
#         }
#     }
#     //savetofile(cpu_ty,"SubA6.txt",m,l);
#      
#     //final beta
#     float *gpu_cc,*cpu_f,*gpuu_a3;
#     cudaMallocHost((void **) &cpu_f, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpu_cc, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpuu_a3, sizeof(float)*m*m);
#     cudaMemcpy(gpuu_a3,cpu_a3, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#     cudaMemcpy(gpu_cc,cpu_ty, sizeof(float)*m*l, cudaMemcpyHostToDevice);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,p,m,&alpha,gpu_cc,l,gpuu_a3,m,&beta,gpu_B,l);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_f,gpu_B, sizeof(float)*p*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_f,"finalopt.txt",p,l);   
#       
#     ////////////////////////batch3
#     //activation,relu
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,m,k,&alpha,gpu_h,p,gpu_tr+2*m*k,k,&beta,gpu_x,p); 
#     cublasDestroy(handle); 
#     cudaMemcpy(cpu_x,gpu_x, sizeof(float)*m*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_x,"mamul2.txt",m,p); 
#     relu(cpu_x,m,p); //output of activation function
#     //savetofile(cpu_x,"relu2.txt",m,p);
#      
#     //calc a3,m*xt
#     cudaMallocHost((void **) &cpu_a3, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a3, sizeof(float)*m*m);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,p,p,&alpha,gpu_x,p,gpu_mi,p,&beta,gpu_a3,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a3,gpu_a3, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a3,"A3.txt",m,m);
#      
#     //calc a1,x*m
#     cudaMallocHost((void **) &cpu_a1, sizeof(float)*m*p);
#     cudaMalloc((void **) &gpu_a1, sizeof(float)*m*p);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,m,p,&alpha,gpu_mi,p,gpu_x,p,&beta,gpu_a1,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a1,gpu_a1, sizeof(float)*m*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a1,"A1.txt",m,p);
#      
#     //calc a2,a1*xt
#     cudaMallocHost((void **) &cpu_a2, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a2, sizeof(float)*m*m);
#  
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,m,p,&alpha,gpu_x,p,gpu_a1,p,&beta,gpu_a2,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a2,gpu_a2, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a2,"A2.txt",m,m);
#      
#     //addition
#     for(int row=0;row<m;row++){
#         for(int col=0;col<m;col++){
#             cpu_a2[row*m+col]=cpu_a2[row*m+col]+cpu_I2[row*m+col];
#         }
#     }
#     //savetofile(cpu_a2,"Add.txt",m,m);
#      
#     //inverse
#     cudaMallocHost((void **) &cpu_a2i, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a2i, sizeof(float)*m*m);
#     cudaMemcpy(gpu_a2i,cpu_a2, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#     for(int i = 0; i<m; i++){
#         nodiag_normalize <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#         diag_normalize <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#  		    gaussjordan <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#  		    set_zero <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#     }
#     cudaMemcpy(cpu_a2i,gpu_I2, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a2i,"XMXTinverse.txt",m,m);
#      
#     //calc A4,A3*A2
#     cudaMallocHost((void **) &gpuu_a2, sizeof(float)*m*m);
#     cudaMemcpy(gpuu_a2,cpu_a2i, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#  
#     cudaMallocHost((void **) &cpu_a4, sizeof(float)*p*m);
#     cudaMalloc((void **) &gpu_a4, sizeof(float)*p*m);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,p,m,&alpha,gpuu_a2,m,gpu_a3,m,&beta,gpu_a4,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a4,gpu_a4, sizeof(float)*p*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a4,"A4.txt",p,m);
#  
#     //calc A5,X*A4
#     cudaMallocHost((void **) &cpu_a5, sizeof(float)*p*p);
#     cudaMallocHost((void **) &cpuu_a5, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpu_a5, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpuu_a5, sizeof(float)*p*p);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,p,m,&alpha,gpu_x,p,gpu_a4,m,&beta,gpu_a5,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a5,gpu_a5, sizeof(float)*p*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a5,"A5.txt",p,p);
#  
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,p,p,&alpha,gpu_mi,p,gpu_a5,p,&beta,gpuu_a5,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpuu_a5,gpuu_a5, sizeof(float)*p*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpuu_a5,"finalA5.txt",p,p);
#              
#     for(int row=0;row<p;row++){
#         for(int col=0;col<p;col++){
#             cpu_mi[row*p+col]=cpu_mi[row*p+col]-cpuu_a5[row*p+col];
#         }
#     }
#     //savetofile(cpu_mi,"SubA5.txt",p,p);
#     cudaMemcpy(gpu_mi,cpu_mi, sizeof(float)*p*p, cudaMemcpyHostToDevice);
#      
#     //calc A6
#     cudaMallocHost((void **) &cpu_a6, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpu_a6, sizeof(float)*m*l);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,m,p,&alpha,gpu_B,l,gpu_x,p,&beta,gpu_a6,l); 
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a6,gpu_a6, sizeof(float)*m*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a6,"A6.txt",m,l);
#     
#     cudaMallocHost((void **) &cpu_ty, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpu_ty, sizeof(float)*m*l);
#     cudaMemcpy(gpu_ty,cpu_trl+2*m*l, sizeof(float)*m*l, cudaMemcpyHostToDevice);
#     cudaMemcpy(cpu_ty,gpu_ty, sizeof(float)*m*l, cudaMemcpyDeviceToHost);
#  
#     for(int row=0;row<m;row++){
#         for(int col=0;col<l;col++){
#             cpu_ty[row*l+col]=cpu_ty[row*l+col]-cpu_a6[row*l+col];
#         }
#       }
#     //savetofile(cpu_ty,"SubA6.txt",m,l);
#      
#     //final beta
#     cudaMallocHost((void **) &cpu_f, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpu_cc, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpuu_a3, sizeof(float)*m*m);
#     cudaMemcpy(gpuu_a3,cpu_a3, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#     cudaMemcpy(gpu_cc,cpu_ty, sizeof(float)*m*l, cudaMemcpyHostToDevice);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,p,m,&alpha,gpu_cc,l,gpuu_a3,m,&beta,gpu_B,l);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_f,gpu_B, sizeof(float)*p*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_f,"finalopt.txt",p,l);
# 
#     //////////////////////////////////batch4
#     //activation,relu
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,m,k,&alpha,gpu_h,p,gpu_tr+3*m*k,k,&beta,gpu_x,p); 
#     cublasDestroy(handle); 
#     cudaMemcpy(cpu_x,gpu_x, sizeof(float)*m*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_x,"mamul2.txt",m,p); 
#     relu(cpu_x,m,p); //output of activation function
#     //savetofile(cpu_x,"relu2.txt",m,p);
#      
#     //calc a3,m*xt
#     cudaMallocHost((void **) &cpu_a3, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a3, sizeof(float)*m*m);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,p,p,&alpha,gpu_x,p,gpu_mi,p,&beta,gpu_a3,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a3,gpu_a3, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a3,"A3.txt",m,m);
#      
#     //calc a1,x*m
#     cudaMallocHost((void **) &cpu_a1, sizeof(float)*m*p);
#     cudaMalloc((void **) &gpu_a1, sizeof(float)*m*p);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,m,p,&alpha,gpu_mi,p,gpu_x,p,&beta,gpu_a1,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a1,gpu_a1, sizeof(float)*m*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a1,"A1.txt",m,p);
#      
#     //calc a2,a1*xt
#     cudaMallocHost((void **) &cpu_a2, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a2, sizeof(float)*m*m);
#  
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,m,p,&alpha,gpu_x,p,gpu_a1,p,&beta,gpu_a2,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a2,gpu_a2, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a2,"A2.txt",m,m);
#      
#     //addition
#     for(int row=0;row<m;row++){
#         for(int col=0;col<m;col++){
#             cpu_a2[row*m+col]=cpu_a2[row*m+col]+cpu_I2[row*m+col];
#             }
#       }
#     //savetofile(cpu_a2,"Add.txt",m,m);
#      
#     //inverse
#     cudaMallocHost((void **) &cpu_a2i, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a2i, sizeof(float)*m*m);
#     cudaMemcpy(gpu_a2i,cpu_a2, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#     for(int i = 0; i<m; i++){
#         nodiag_normalize <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#         diag_normalize <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#  		    gaussjordan <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#  		    set_zero <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#       }
#     cudaMemcpy(cpu_a2i,gpu_I2, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a2i,"XMXTinverse.txt",m,m);
#      
#     //calc A4,A3*A2
#     cudaMallocHost((void **) &gpuu_a2, sizeof(float)*m*m);
#     cudaMemcpy(gpuu_a2,cpu_a2i, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#  
#     cudaMallocHost((void **) &cpu_a4, sizeof(float)*p*m);
#     cudaMalloc((void **) &gpu_a4, sizeof(float)*p*m);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,p,m,&alpha,gpuu_a2,m,gpu_a3,m,&beta,gpu_a4,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a4,gpu_a4, sizeof(float)*p*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a4,"A4.txt",p,m);
#  
#     //calc A5,X*A4
#     cudaMallocHost((void **) &cpu_a5, sizeof(float)*p*p);
#     cudaMallocHost((void **) &cpuu_a5, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpu_a5, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpuu_a5, sizeof(float)*p*p);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,p,m,&alpha,gpu_x,p,gpu_a4,m,&beta,gpu_a5,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a5,gpu_a5, sizeof(float)*p*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a5,"A5.txt",p,p);
#  
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,p,p,&alpha,gpu_mi,p,gpu_a5,p,&beta,gpuu_a5,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpuu_a5,gpuu_a5, sizeof(float)*p*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpuu_a5,"finalA5.txt",p,p);
#              
#     for(int row=0;row<p;row++){
#         for(int col=0;col<p;col++){
#             cpu_mi[row*p+col]=cpu_mi[row*p+col]-cpuu_a5[row*p+col];
#             }
#         }
#     //savetofile(cpu_mi,"SubA5.txt",p,p);
#     cudaMemcpy(gpu_mi,cpu_mi, sizeof(float)*p*p, cudaMemcpyHostToDevice);
#      
#     //calc A6
#     cudaMallocHost((void **) &cpu_a6, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpu_a6, sizeof(float)*m*l);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,m,p,&alpha,gpu_B,l,gpu_x,p,&beta,gpu_a6,l); 
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a6,gpu_a6, sizeof(float)*m*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a6,"A6.txt",m,l);
#      
#     cudaMallocHost((void **) &cpu_ty, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpu_ty, sizeof(float)*m*l);
#     cudaMemcpy(gpu_ty,cpu_trl+3*m*l, sizeof(float)*m*l, cudaMemcpyHostToDevice);
#     cudaMemcpy(cpu_ty,gpu_ty, sizeof(float)*m*l, cudaMemcpyDeviceToHost);
#  
#     for(int row=0;row<m;row++){
#         for(int col=0;col<l;col++){
#             cpu_ty[row*l+col]=cpu_ty[row*l+col]-cpu_a6[row*l+col];
#         }
#       }
#     //savetofile(cpu_ty,"SubA6.txt",m,l);
#      
#     //final beta
#     cudaMallocHost((void **) &cpu_f, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpu_cc, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpuu_a3, sizeof(float)*m*m);
#     cudaMemcpy(gpuu_a3,cpu_a3, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#     cudaMemcpy(gpu_cc,cpu_ty, sizeof(float)*m*l, cudaMemcpyHostToDevice);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,p,m,&alpha,gpu_cc,l,gpuu_a3,m,&beta,gpu_B,l);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_f,gpu_B, sizeof(float)*p*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_f,"finalopt.txt",p,l);       
#      
#     //////////////////////batch 5 final batch
#     //activation,relu
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,m,k,&alpha,gpu_h,p,gpu_tr+4*m*k,k,&beta,gpu_x,p); 
#     cublasDestroy(handle); 
#     cudaMemcpy(cpu_x,gpu_x, sizeof(float)*m*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_x,"mamul2.txt",m,p); 
#     relu(cpu_x,m,p); //output of activation function
#     //savetofile(cpu_x,"relu2.txt",m,p);
#      
#     //calc a3,m*xt
#     cudaMallocHost((void **) &cpu_a3, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a3, sizeof(float)*m*m);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,p,p,&alpha,gpu_x,p,gpu_mi,p,&beta,gpu_a3,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a3,gpu_a3, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a3,"A3.txt",m,m);
#      
#     //calc a1,x*m
#     cudaMallocHost((void **) &cpu_a1, sizeof(float)*m*p);
#     cudaMalloc((void **) &gpu_a1, sizeof(float)*m*p);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,m,p,&alpha,gpu_mi,p,gpu_x,p,&beta,gpu_a1,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a1,gpu_a1, sizeof(float)*m*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a1,"A1.txt",m,p);
#      
#     //calc a2,a1*xt
#     cudaMallocHost((void **) &cpu_a2, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a2, sizeof(float)*m*m);
#  
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,m,p,&alpha,gpu_x,p,gpu_a1,p,&beta,gpu_a2,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a2,gpu_a2, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a2,"A2.txt",m,m);
#      
#     //addition
#     for(int row=0;row<m;row++){
#         for(int col=0;col<m;col++){
#             cpu_a2[row*m+col]=cpu_a2[row*m+col]+cpu_I2[row*m+col];
#           }
#       }
#     //savetofile(cpu_a2,"Add.txt",m,m);
#      
#     //inverse
#     cudaMallocHost((void **) &cpu_a2i, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a2i, sizeof(float)*m*m);
#     cudaMemcpy(gpu_a2i,cpu_a2, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#     for(int i = 0; i<m; i++){
#         nodiag_normalize <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#         diag_normalize <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#  		    gaussjordan <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#  		    set_zero <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#       }
#     cudaMemcpy(cpu_a2i,gpu_I2, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a2i,"XMXTinverse.txt",m,m);
#      
#     //calc A4,A3*A2
#     cudaMallocHost((void **) &gpuu_a2, sizeof(float)*m*m);
#     cudaMemcpy(gpuu_a2,cpu_a2i, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#  
#     cudaMallocHost((void **) &cpu_a4, sizeof(float)*p*m);
#     cudaMalloc((void **) &gpu_a4, sizeof(float)*p*m);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,p,m,&alpha,gpuu_a2,m,gpu_a3,m,&beta,gpu_a4,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a4,gpu_a4, sizeof(float)*p*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a4,"A4.txt",p,m);
#  
#     //calc A5,X*A4
#     cudaMallocHost((void **) &cpu_a5, sizeof(float)*p*p);
#     cudaMallocHost((void **) &cpuu_a5, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpu_a5, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpuu_a5, sizeof(float)*p*p);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,p,m,&alpha,gpu_x,p,gpu_a4,m,&beta,gpu_a5,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a5,gpu_a5, sizeof(float)*p*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a5,"A5.txt",p,p);
#  
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,p,p,&alpha,gpu_mi,p,gpu_a5,p,&beta,gpuu_a5,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpuu_a5,gpuu_a5, sizeof(float)*p*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpuu_a5,"finalA5.txt",p,p);
#              
#     for(int row=0;row<p;row++){
#         for(int col=0;col<p;col++){
#             cpu_mi[row*p+col]=cpu_mi[row*p+col]-cpuu_a5[row*p+col];
#             }
#         }
#     //savetofile(cpu_mi,"SubA5.txt",p,p);
#     cudaMemcpy(gpu_mi,cpu_mi, sizeof(float)*p*p, cudaMemcpyHostToDevice);
#      
#     //calc A6
#     cudaMallocHost((void **) &cpu_a6, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpu_a6, sizeof(float)*m*l);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,m,p,&alpha,gpu_B,l,gpu_x,p,&beta,gpu_a6,l); 
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a6,gpu_a6, sizeof(float)*m*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a6,"A6.txt",m,l);
#      
#     cudaMallocHost((void **) &cpu_ty, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpu_ty, sizeof(float)*m*l);
#     cudaMemcpy(gpu_ty,cpu_trl+4*m*l, sizeof(float)*m*l, cudaMemcpyHostToDevice);
#     cudaMemcpy(cpu_ty,gpu_ty, sizeof(float)*m*l, cudaMemcpyDeviceToHost);
#  
#     for(int row=0;row<m;row++){
#         for(int col=0;col<l;col++){
#             cpu_ty[row*l+col]=cpu_ty[row*l+col]-cpu_a6[row*l+col];
#         }
#     }
#     //savetofile(cpu_ty,"SubA6.txt",m,l);
#      
#     //final beta
#     cudaMallocHost((void **) &cpu_f, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpu_cc, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpuu_a3, sizeof(float)*m*m);
#     cudaMemcpy(gpuu_a3,cpu_a3, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#     cudaMemcpy(gpu_cc,cpu_ty, sizeof(float)*m*l, cudaMemcpyHostToDevice);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,p,m,&alpha,gpu_cc,l,gpuu_a3,m,&beta,gpu_B,l);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_f,gpu_B, sizeof(float)*p*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_f,"finalopt.txt",p,l);   
#      
#     //testing part 
#     float *cpu_o,*gpu_o;
#     cudaMallocHost((void **) &cpu_o, sizeof(float)*m*p);
#     cudaMalloc((void **) &gpu_o, sizeof(float)*m*p);
#  
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,m,k,&alpha,gpu_h,p,gpu_tt,k,&beta,gpu_o,p); 
#     cublasDestroy(handle); 
#     cudaMemcpy(cpu_o,gpu_o, sizeof(float)*m*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_o,"testop.txt",m,p); 
#     relu(cpu_o,m,p); //output of activation function
#     //savetofile(cpu_o,"testrelu.txt",m,p);
#    
#     float *cpu_y,*gpu_y,*g_fy;
#     cudaMalloc((void **) &g_fy, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpu_y, sizeof(float)*m*p);
#     cudaMemcpy(gpu_y,cpu_o, sizeof(float)*m*p, cudaMemcpyHostToDevice);
#     cudaMallocHost((void **) &cpu_y, sizeof(float)*m*l);
#      
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,p,p,&alpha,gpu_B,p,gpu_y,m,&beta,g_fy,l);
#     cublasDestroy(handle); 
#     cudaMemcpy(cpu_y,g_fy, sizeof(float)*m*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_y,"finaltest.txt",m,l);
#     //calculation of accuracy
#     float c;  
#     for(int i=0;i<m;++i){
#         for(int j=0;j<l;++j){
#             if(cpu_y[i*l+j]==cpu_ttl[i*l+j]){
#                 c+=1.0;
#            }
#         }
#     }
#     float acc;
#     acc=c/100000;
#     std::ofstream myfile;
#     myfile.open ("Accuracy.txt");
#     myfile << acc;
#     myfile.close();
#     printf("Accuracy is %f\t",acc);
#     cudaFreeHost(cpu_h);
#     cudaFreeHost(cpu_I1);
#     cudaFreeHost(cpu_I2);
#     cudaFreeHost(cpu_tr);
#     cudaFreeHost(cpu_trl);
#     cudaFreeHost(cpu_tt);
#     cudaFreeHost(cpu_ttl);
#     cudaFreeHost(cpu_x);
#     cudaFreeHost(cpu_m);
#     cudaFreeHost(cpu_mi);
#     cudaFreeHost(cpu_xty);
#     cudaFreeHost(cpu_B);
#     cudaFreeHost(cpu_a3);
#     cudaFreeHost(cpu_a1);
#     cudaFreeHost(cpu_a2);
#     cudaFreeHost(cpu_a2i);
#     cudaFreeHost(cpu_a4);
#     cudaFreeHost(cpu_a5);
#     cudaFreeHost(cpuu_a5);
#     cudaFreeHost(cpu_a6);
#     cudaFreeHost(cpu_ty);
#     cudaFreeHost(cpu_f);
#     cudaFreeHost(cpu_o);
#     cudaFreeHost(cpu_y);
#     cudaFree(gpu_y);
#     cudaFree(gpu_r);
#     cudaFree(g_fy);
#     cudaFree(gpu_o);
#     cudaFree(gpu_cc);
#     cudaFree(gpu_a6);
#     cudaFree(gpu_a5);
#     cudaFree(gpuu_a5);
#     cudaFree(gpu_xty);
#     cudaFree(gpu_h);
#     cudaFree(gpu_I1);
#     cudaFree(gpu_I2);
#     cudaFree(gpu_tr);
#     cudaFree(gpu_trl);
#     cudaFree(gpu_tt);
#     cudaFree(gpu_ttl);
#     cudaFree(gpu_x);
#     cudaFree(gpu_m);
#     cudaFree(gpu_mi);
#     cudaFree(gpu_B);
#     cudaFree(gpu_a3);
#     cudaFree(gpu_a1);
#     cudaFree(gpu_a2);
#     cudaFree(gpu_a2i);
#     cudaFree(gpu_a4);
#     cudaFree(gpuu_a2);
#     cudaFree(gpu_ty);    
#     return 0;
# }

!nvcc -o /content/src/my_curand /content/src/my_curand.cu -lcurand -lcublas
!/content/src/my_curand

"""**for cifar 100 there are path changes rest is same beelow is the code for cifar 100**

**CIFAR100 code**
"""

# Commented out IPython magic to ensure Python compatibility.
# %%cuda --name my_curand1.cu
# #include <stdio.h>
# #include <cstdlib>
# #include <vector>
# #include <random>
# #include <algorithm>
# #include <math.h>
# #include <cuda.h>
# #include <stdlib.h> 
# #include <time.h> 
# #include <iostream>
# #include <ctime>
# #include <fstream>
# #include <string>
# #include <cuda_runtime.h>
# #include <device_launch_parameters.h>
# #include <cublas_v2.h>
# #include <cuda_runtime_api.h>
#  
# using namespace std;
# #define blocksize 8
#  
# //function to generate input weights for hidden neurons
# float random_float(float min, float max){
#      return ((float)rand() / RAND_MAX) * (max - min) + min;
# }
# 
# //for calculating inverse
# __global__ void nodiag_normalize(float *A,float *I, int n, int i){
#     int x = blockIdx.x * blockDim.x + threadIdx.x;
#  	  int y = blockIdx.y * blockDim.y + threadIdx.y;
#  	  if(x < n && y < n){
#          if(x == i && x!=y){
#              I[x*n + y] /= A[i*n + i];
#              A[x*n + y] /= A[i*n + i];
#          }
#      }	
# }
# 
# //for calculating inverse
# __global__ void diag_normalize(float *A,float *I, int n, int i){
#     int x = blockIdx.x * blockDim.x + threadIdx.x;
#  	  int y = blockIdx.y * blockDim.y + threadIdx.y;
#  	  if(x < n && y < n){
#          if(x == y && x == i){
#              I[x*n + y] /= A[i*n + i];
#              A[x*n + y] /= A[i*n + i];
#          }
#      }
# }
# 
# //for calculating inverse
# __global__ void gaussjordan(float *A,float *I, int n, int i){
#      int x = blockIdx.x * blockDim.x + threadIdx.x;
#      int y = blockIdx.y * blockDim.y + threadIdx.y;
#      if(x < n && y < n){
#          if(x != i){
#              I[x*n + y] -= I[i*n + y] * A[x*n + i];
#              if(y != i){
#                  A[x*n + y] -= A[i*n + y] * A[x*n + i];
#              }
#          }
#      }
# }
# 
# //for calculating inverse
#  __global__ void set_zero(float *A,float *I, int n, int i){
#      int x = blockIdx.x * blockDim.x + threadIdx.x;
#  	  int y = blockIdx.y * blockDim.y + threadIdx.y;
#      if(x < n && y < n){
#          if(x != i){
#              if(y == i){
#                  A[x*n + y] = 0;
#              }
#          }
#      }
# }
#  
# //read data from text file
# void matrix_read(float *L,const char *path,int m,int n){
#   FILE *fp;
#  	int row, col;
#  
#  	fp = fopen(path, "r");//open output file
#  	if (fp == NULL)//open failed
#  		return;
#  
#  	for (row = 0; row < m; row++){
#  		for (col = 0; col < n; col++)
#  		if (fscanf(fp, "%f,", &L[row * n + col]) == EOF) break;//read data
#  
#  		if (feof(fp)) break;//if the file is over
#  	}
#  
#  	fclose(fp);//close file 
# }
#  
# //function to generate identity matrix
# float identity(float *a,int num){
#      int row,col;
#      for(row = 0; row < num; row++){
#          for(col = 0; col < num; col++){
#              if(row == col){
#                  a[row*num+col]=1.0f;
#              }
#              else{
#                  a[row*num+col]=0.0f;
#              }
#          }
#     }
#     return 0;
# }
#  
# //relu activation function
# float relu(float *a,int m,int p){
#      for(int i=0;i<m;i++){
#          for(int j=0;j<p;j++){
#              if(a[i*p+j]>0){
#                  a[i*p+j]=a[i*p+j];
#                  //printf("no change");
#                  }
#              else{
#                  a[i*p+j]=0.0;
#                  //printf("do the change");
#              }
#          }
#      }
#      return 0;
# }
# 
# //this for checking the intermediate ouput by taking it into text file 
# void savetofile(float *A, string s, int n, int h){
#      std::ofstream plik;
#      plik.open(s);
#      for(int i = 0; i<n; i++){
#          for(int j = 0; j<h; j++){
#              plik << A[i*h + j] << "\t";
#          }
#          plik << endl;
#      }
#      plik.close();
# }
#  
# //oselm i have created 5 batches from 0 to 4  manually and pointer locations are jumped
# //to access the chunks of data for processing in batch
# int main(){
#     //seed to get random data 
#     srand(time(0));
#     float alpha = 1.0f;
#  	  float beta = 0.0f;
#  
#      //cifar 100 params
#      int n=50000,k=1024,m=10000,p=1000,l=100;
#      
#     //dimension for calculating inverse 
#     dim3 threadsPerBlock(blocksize, blocksize);
#     dim3 numBlocks((p + blocksize - 1) / blocksize, (p + blocksize - 1) / blocksize);
#      
#     //generate input hidden neurons
#     float *cpu_h,*gpu_h;
#     cudaMallocHost((void **) &cpu_h, sizeof(float)*k*p);
#     cudaMalloc((void **) &gpu_h, sizeof(float)*k*p);
#     //generate input weights for hidden neurons
#     for (int i = 0; i < k; ++i) {
#         for (int j = 0; j < p; ++j) {
#             cpu_h[i * p + j] = random_float(-2.0, 2.0);    
#         }
#     }
#     //copy to gpu host is cpu and device is gpu
#     cudaMemcpy(gpu_h,cpu_h, sizeof(float)*k*p, cudaMemcpyHostToDevice);
#     //savetofile(cpu_h,"H.txt",k,p); 
# 
#     //identity matrix generation and storing on gpu and cpu
#     float *cpu_I1,*cpu_I2,*gpu_I1,*gpu_I2;
#     cudaMallocHost((void **) &cpu_I1, sizeof(float)*p*p);
#     cudaMallocHost((void **) &cpu_I2, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_I1, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpu_I2, sizeof(float)*m*m);
#  
#     identity(cpu_I1,p);
#     //savetofile(cpu_I1,"I1.txt",p,p); 
#     identity(cpu_I2,m);
#     //savetofile(cpu_I2,"I2.txt",m,m); 
#     cudaMemcpy(gpu_I1,cpu_I1, sizeof(float)*p*p, cudaMemcpyHostToDevice);
#     cudaMemcpy(gpu_I2,cpu_I2, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#  
#     //get train data and label
#     float *cpu_tr,*cpu_trl,*gpu_tr,*gpu_trl;
#     cudaMallocHost((void **) &cpu_tr, sizeof(float)*n*k);//train data
#     cudaMallocHost((void **) &cpu_trl, sizeof(float)*n*l);//train label
#  
#     cudaMalloc((void **) &gpu_tr, sizeof(float)*n*k);
#     cudaMalloc((void **) &gpu_trl, sizeof(float)*n*l);
#  
#     const char path[] = "/content/drive/MyDrive/df_train_cifar100.txt";
#     matrix_read(cpu_tr,path,n,k);
#     const char path1[] = "/content/drive/MyDrive/df_trainy_cifar100.txt";
#     matrix_read(cpu_trl,path1,n,l);
#      
#     cudaMemcpy(gpu_tr,cpu_tr, sizeof(float)*n*k, cudaMemcpyHostToDevice);
#     cudaMemcpy(gpu_trl,cpu_trl, sizeof(float)*n*l, cudaMemcpyHostToDevice);
#  
#     //get test data and label
#     float *cpu_tt,*cpu_ttl,*gpu_tt,*gpu_ttl;
#     cudaMallocHost((void **) &cpu_tt, sizeof(float)*m*k);//test data
#     cudaMallocHost((void **) &cpu_ttl, sizeof(float)*m*l);//test label
#  
#     cudaMalloc((void **) &gpu_tt, sizeof(float)*m*k);
#     cudaMalloc((void **) &gpu_ttl, sizeof(float)*m*l);
#  
#     const char path2[] = "/content/drive/MyDrive/df_test_cifar100.txt";
#     matrix_read(cpu_tt,path2,m,k);
#     const char path3[] = "/content/drive/MyDrive/df_testy_cifar100.txt";
#     matrix_read(cpu_ttl,path3,m,l);
#  
#     cudaMemcpy(gpu_tt,cpu_tt, sizeof(float)*m*k, cudaMemcpyHostToDevice);
#     cudaMemcpy(gpu_ttl,cpu_ttl, sizeof(float)*m*l, cudaMemcpyHostToDevice);
#  
#     //to store output of dot product and  activation function
#     float *cpu_x,*gpu_x; 
#     cudaMallocHost((void **) &cpu_x, sizeof(float)*m*p);
#     cudaMalloc((void **) &gpu_x, sizeof(float)*m*p);
#  
#     //for storing inverse
#     float *cpu_m,*gpu_m;
#     cudaMallocHost((void **) &cpu_m, sizeof(float)*p*p); 
#     cudaMalloc((void **) &gpu_m, sizeof(float)*p*p); 
#     
#     cublasHandle_t handle;   
#     //maually implemented oslem and divdied in batches of 5 from 0 to 4
#     ///////////////////////batch1
#     //activation,relu
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,m,k,&alpha,gpu_h,p,gpu_tr+0*m*k,k,&beta,gpu_x,p); 
#     cublasDestroy(handle); 
#      
#     cudaMemcpy(cpu_x,gpu_x, sizeof(float)*m*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_x,"mamul1.txt",m,p); 
#     relu(cpu_x,m,p); //output of activation function
#     //savetofile(cpu_x,"relu.txt",m,p);
#    
#     //copy X*XT
#     float *gpu_r;
#     cudaMalloc((void **) &gpu_r, sizeof(float)*m*p);
#     cudaMemcpy(gpu_r,cpu_x, sizeof(float)*m*p, cudaMemcpyHostToDevice);    
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,p,m,p,&alpha,gpu_r,p,gpu_r,m,&beta,gpu_m,p);
#     cublasDestroy(handle); 
#     cudaMemcpy(cpu_m,gpu_m, sizeof(float)*p*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_m,"transpose.txt",p,p);
#      
#     float *cpu_mi,*gpu_mi;
#     cudaMalloc((void **) &gpu_mi, sizeof(float)*p*p); 
#     cudaMallocHost((void **) &cpu_mi, sizeof(float)*p*p);
#     //inverse of x*XT
#     for (int i = 0; i<p; i++){
#         nodiag_normalize << <numBlocks, threadsPerBlock >> >(gpu_x,gpu_I1,p,i);
#  		    diag_normalize << <numBlocks, threadsPerBlock >> >(gpu_x,gpu_I1,p,i);
#  		    gaussjordan << <numBlocks, threadsPerBlock >> >(gpu_x,gpu_I1,p,i);
#  		    set_zero << <numBlocks, threadsPerBlock >> >(gpu_x,gpu_I1,p,i);
#      }
#     cudaMemcpy(cpu_mi,gpu_I1, sizeof(float)*p*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_mi,"inverse.txt",p,p);
#     cudaMemcpy(gpu_mi,cpu_mi, sizeof(float)*p*p, cudaMemcpyHostToDevice);
#  
#     //Xtrain *Xt
#     float *cpu_xty,*gpu_xty;
#     cudaMallocHost((void **) &cpu_xty, sizeof(float)*p*l);
#     cudaMalloc((void **) &gpu_xty, sizeof(float)*p*l);
#  
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,l,p,m,&alpha,gpu_trl+0*m*l,l,gpu_x,p,&beta,gpu_xty,l);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_xty,gpu_xty, sizeof(float)*p*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_xty,"label.txt",p,l);
#    
#     float *cpu_B,*gpu_B;
#     cudaMallocHost((void **) &cpu_B, sizeof(float)*p*l);
#     cudaMalloc((void **) &gpu_B, sizeof(float)*p*l);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,p,p,&alpha,gpu_xty,l,gpu_I1,p,&beta,gpu_B,l);  
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_B,gpu_B, sizeof(float)*p*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_B,"beta.txt",p,l);
#      
#     ////////////////////////////////////batch2
#     //activation,relu
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,m,k,&alpha,gpu_h,p,gpu_tr+1*m*k,k,&beta,gpu_x,p); 
#     cublasDestroy(handle); 
#     cudaMemcpy(cpu_x,gpu_x, sizeof(float)*m*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_x,"mamul2.txt",m,p); 
#     relu(cpu_x,m,p); //output of activation function
#     //savetofile(cpu_x,"relu2.txt",m,p);
#      
#     //calc a3,m*xt
#     float *cpu_a3,*gpu_a3;
#     cudaMallocHost((void **) &cpu_a3, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a3, sizeof(float)*m*m);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,p,p,&alpha,gpu_x,p,gpu_mi,p,&beta,gpu_a3,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a3,gpu_a3, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a3,"A3.txt",m,m);
#      
#     //calc a1,x*m
#     float *cpu_a1,*gpu_a1;
#     cudaMallocHost((void **) &cpu_a1, sizeof(float)*m*p);
#     cudaMalloc((void **) &gpu_a1, sizeof(float)*m*p);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,m,p,&alpha,gpu_mi,p,gpu_x,p,&beta,gpu_a1,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a1,gpu_a1, sizeof(float)*m*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a1,"A1.txt",m,p);
#      
#     //calc a2,a1*xt
#     float *cpu_a2,*gpu_a2;
#     cudaMallocHost((void **) &cpu_a2, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a2, sizeof(float)*m*m);
#  
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,m,p,&alpha,gpu_x,p,gpu_a1,p,&beta,gpu_a2,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a2,gpu_a2, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a2,"A2.txt",m,m);
#      
#     //addition
#     for(int row=0;row<m;row++){
#         for(int col=0;col<m;col++){
#             cpu_a2[row*m+col]=cpu_a2[row*m+col]+cpu_I2[row*m+col];
#         }
#     }
#     //savetofile(cpu_a2,"Add.txt",m,m);
#      
#     //inverse
#     float *cpu_a2i,*gpu_a2i;
#     cudaMallocHost((void **) &cpu_a2i, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a2i, sizeof(float)*m*m);
#     cudaMemcpy(gpu_a2i,cpu_a2, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#     for(int i = 0; i<m; i++){
#         nodiag_normalize <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#         diag_normalize <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#  		    gaussjordan <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#  		    set_zero <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#     }
#     cudaMemcpy(cpu_a2i,gpu_I2, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a2i,"XMXTinverse.txt",m,m);
#      
#     //calc A4,A3*A2
#     float *cpu_a4,*gpu_a4,*gpuu_a2;
#     cudaMallocHost((void **) &gpuu_a2, sizeof(float)*m*m);
#     cudaMemcpy(gpuu_a2,cpu_a2i, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#  
#     cudaMallocHost((void **) &cpu_a4, sizeof(float)*p*m);
#     cudaMalloc((void **) &gpu_a4, sizeof(float)*p*m);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,p,m,&alpha,gpuu_a2,m,gpu_a3,m,&beta,gpu_a4,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a4,gpu_a4, sizeof(float)*p*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a4,"A4.txt",p,m);
#  
#     //calc A5,X*A4
#     float *cpu_a5,*gpu_a5,*cpuu_a5,*gpuu_a5;
#     cudaMallocHost((void **) &cpu_a5, sizeof(float)*p*p);
#     cudaMallocHost((void **) &cpuu_a5, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpu_a5, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpuu_a5, sizeof(float)*p*p);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,p,m,&alpha,gpu_x,p,gpu_a4,m,&beta,gpu_a5,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a5,gpu_a5, sizeof(float)*p*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a5,"A5.txt",p,p);
#  
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,p,p,&alpha,gpu_mi,p,gpu_a5,p,&beta,gpuu_a5,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpuu_a5,gpuu_a5, sizeof(float)*p*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpuu_a5,"finalA5.txt",p,p);
#              
#     for(int row=0;row<p;row++){
#         for(int col=0;col<p;col++){
#             cpu_mi[row*p+col]=cpu_mi[row*p+col]-cpuu_a5[row*p+col];
#         }
#     }
#     //savetofile(cpu_mi,"SubA5.txt",p,p);
#     cudaMemcpy(gpu_mi,cpu_mi, sizeof(float)*p*p, cudaMemcpyHostToDevice);
#      
#     //calc A6
#     float *cpu_a6,*gpu_a6;
#     cudaMallocHost((void **) &cpu_a6, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpu_a6, sizeof(float)*m*l);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,m,p,&alpha,gpu_B,l,gpu_x,p,&beta,gpu_a6,l); 
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a6,gpu_a6, sizeof(float)*m*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a6,"A6.txt",m,l);
#      
#     float *cpu_ty,*gpu_ty;
#     cudaMallocHost((void **) &cpu_ty, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpu_ty, sizeof(float)*m*l);
#     cudaMemcpy(gpu_ty,cpu_trl+1*m*l, sizeof(float)*m*l, cudaMemcpyHostToDevice);
#     cudaMemcpy(cpu_ty,gpu_ty, sizeof(float)*m*l, cudaMemcpyDeviceToHost);
#  
#     for(int row=0;row<m;row++){
#         for(int col=0;col<l;col++){
#             cpu_ty[row*l+col]=cpu_ty[row*l+col]-cpu_a6[row*l+col];
#         }
#     }
#     //savetofile(cpu_ty,"SubA6.txt",m,l);
#      
#     //final beta
#     float *gpu_cc,*cpu_f,*gpuu_a3;
#     cudaMallocHost((void **) &cpu_f, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpu_cc, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpuu_a3, sizeof(float)*m*m);
#     cudaMemcpy(gpuu_a3,cpu_a3, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#     cudaMemcpy(gpu_cc,cpu_ty, sizeof(float)*m*l, cudaMemcpyHostToDevice);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,p,m,&alpha,gpu_cc,l,gpuu_a3,m,&beta,gpu_B,l);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_f,gpu_B, sizeof(float)*p*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_f,"finalopt.txt",p,l);   
#       
#     ////////////////////////batch3
#     //activation,relu
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,m,k,&alpha,gpu_h,p,gpu_tr+2*m*k,k,&beta,gpu_x,p); 
#     cublasDestroy(handle); 
#     cudaMemcpy(cpu_x,gpu_x, sizeof(float)*m*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_x,"mamul2.txt",m,p); 
#     relu(cpu_x,m,p); //output of activation function
#     //savetofile(cpu_x,"relu2.txt",m,p);
#      
#     //calc a3,m*xt
#     cudaMallocHost((void **) &cpu_a3, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a3, sizeof(float)*m*m);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,p,p,&alpha,gpu_x,p,gpu_mi,p,&beta,gpu_a3,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a3,gpu_a3, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a3,"A3.txt",m,m);
#      
#     //calc a1,x*m
#     cudaMallocHost((void **) &cpu_a1, sizeof(float)*m*p);
#     cudaMalloc((void **) &gpu_a1, sizeof(float)*m*p);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,m,p,&alpha,gpu_mi,p,gpu_x,p,&beta,gpu_a1,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a1,gpu_a1, sizeof(float)*m*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a1,"A1.txt",m,p);
#      
#     //calc a2,a1*xt
#     cudaMallocHost((void **) &cpu_a2, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a2, sizeof(float)*m*m);
#  
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,m,p,&alpha,gpu_x,p,gpu_a1,p,&beta,gpu_a2,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a2,gpu_a2, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a2,"A2.txt",m,m);
#      
#     //addition
#     for(int row=0;row<m;row++){
#         for(int col=0;col<m;col++){
#             cpu_a2[row*m+col]=cpu_a2[row*m+col]+cpu_I2[row*m+col];
#         }
#     }
#     //savetofile(cpu_a2,"Add.txt",m,m);
#      
#     //inverse
#     cudaMallocHost((void **) &cpu_a2i, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a2i, sizeof(float)*m*m);
#     cudaMemcpy(gpu_a2i,cpu_a2, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#     for(int i = 0; i<m; i++){
#         nodiag_normalize <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#         diag_normalize <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#  		    gaussjordan <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#  		    set_zero <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#     }
#     cudaMemcpy(cpu_a2i,gpu_I2, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a2i,"XMXTinverse.txt",m,m);
#      
#     //calc A4,A3*A2
#     cudaMallocHost((void **) &gpuu_a2, sizeof(float)*m*m);
#     cudaMemcpy(gpuu_a2,cpu_a2i, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#  
#     cudaMallocHost((void **) &cpu_a4, sizeof(float)*p*m);
#     cudaMalloc((void **) &gpu_a4, sizeof(float)*p*m);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,p,m,&alpha,gpuu_a2,m,gpu_a3,m,&beta,gpu_a4,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a4,gpu_a4, sizeof(float)*p*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a4,"A4.txt",p,m);
#  
#     //calc A5,X*A4
#     cudaMallocHost((void **) &cpu_a5, sizeof(float)*p*p);
#     cudaMallocHost((void **) &cpuu_a5, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpu_a5, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpuu_a5, sizeof(float)*p*p);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,p,m,&alpha,gpu_x,p,gpu_a4,m,&beta,gpu_a5,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a5,gpu_a5, sizeof(float)*p*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a5,"A5.txt",p,p);
#  
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,p,p,&alpha,gpu_mi,p,gpu_a5,p,&beta,gpuu_a5,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpuu_a5,gpuu_a5, sizeof(float)*p*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpuu_a5,"finalA5.txt",p,p);
#              
#     for(int row=0;row<p;row++){
#         for(int col=0;col<p;col++){
#             cpu_mi[row*p+col]=cpu_mi[row*p+col]-cpuu_a5[row*p+col];
#         }
#     }
#     //savetofile(cpu_mi,"SubA5.txt",p,p);
#     cudaMemcpy(gpu_mi,cpu_mi, sizeof(float)*p*p, cudaMemcpyHostToDevice);
#      
#     //calc A6
#     cudaMallocHost((void **) &cpu_a6, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpu_a6, sizeof(float)*m*l);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,m,p,&alpha,gpu_B,l,gpu_x,p,&beta,gpu_a6,l); 
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a6,gpu_a6, sizeof(float)*m*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a6,"A6.txt",m,l);
#     
#     cudaMallocHost((void **) &cpu_ty, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpu_ty, sizeof(float)*m*l);
#     cudaMemcpy(gpu_ty,cpu_trl+2*m*l, sizeof(float)*m*l, cudaMemcpyHostToDevice);
#     cudaMemcpy(cpu_ty,gpu_ty, sizeof(float)*m*l, cudaMemcpyDeviceToHost);
#  
#     for(int row=0;row<m;row++){
#         for(int col=0;col<l;col++){
#             cpu_ty[row*l+col]=cpu_ty[row*l+col]-cpu_a6[row*l+col];
#         }
#       }
#     //savetofile(cpu_ty,"SubA6.txt",m,l);
#      
#     //final beta
#     cudaMallocHost((void **) &cpu_f, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpu_cc, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpuu_a3, sizeof(float)*m*m);
#     cudaMemcpy(gpuu_a3,cpu_a3, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#     cudaMemcpy(gpu_cc,cpu_ty, sizeof(float)*m*l, cudaMemcpyHostToDevice);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,p,m,&alpha,gpu_cc,l,gpuu_a3,m,&beta,gpu_B,l);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_f,gpu_B, sizeof(float)*p*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_f,"finalopt.txt",p,l);
# 
#     //////////////////////////////////batch4
#     //activation,relu
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,m,k,&alpha,gpu_h,p,gpu_tr+3*m*k,k,&beta,gpu_x,p); 
#     cublasDestroy(handle); 
#     cudaMemcpy(cpu_x,gpu_x, sizeof(float)*m*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_x,"mamul2.txt",m,p); 
#     relu(cpu_x,m,p); //output of activation function
#     //savetofile(cpu_x,"relu2.txt",m,p);
#      
#     //calc a3,m*xt
#     cudaMallocHost((void **) &cpu_a3, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a3, sizeof(float)*m*m);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,p,p,&alpha,gpu_x,p,gpu_mi,p,&beta,gpu_a3,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a3,gpu_a3, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a3,"A3.txt",m,m);
#      
#     //calc a1,x*m
#     cudaMallocHost((void **) &cpu_a1, sizeof(float)*m*p);
#     cudaMalloc((void **) &gpu_a1, sizeof(float)*m*p);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,m,p,&alpha,gpu_mi,p,gpu_x,p,&beta,gpu_a1,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a1,gpu_a1, sizeof(float)*m*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a1,"A1.txt",m,p);
#      
#     //calc a2,a1*xt
#     cudaMallocHost((void **) &cpu_a2, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a2, sizeof(float)*m*m);
#  
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,m,p,&alpha,gpu_x,p,gpu_a1,p,&beta,gpu_a2,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a2,gpu_a2, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a2,"A2.txt",m,m);
#      
#     //addition
#     for(int row=0;row<m;row++){
#         for(int col=0;col<m;col++){
#             cpu_a2[row*m+col]=cpu_a2[row*m+col]+cpu_I2[row*m+col];
#             }
#       }
#     //savetofile(cpu_a2,"Add.txt",m,m);
#      
#     //inverse
#     cudaMallocHost((void **) &cpu_a2i, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a2i, sizeof(float)*m*m);
#     cudaMemcpy(gpu_a2i,cpu_a2, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#     for(int i = 0; i<m; i++){
#         nodiag_normalize <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#         diag_normalize <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#  		    gaussjordan <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#  		    set_zero <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#       }
#     cudaMemcpy(cpu_a2i,gpu_I2, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a2i,"XMXTinverse.txt",m,m);
#      
#     //calc A4,A3*A2
#     cudaMallocHost((void **) &gpuu_a2, sizeof(float)*m*m);
#     cudaMemcpy(gpuu_a2,cpu_a2i, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#  
#     cudaMallocHost((void **) &cpu_a4, sizeof(float)*p*m);
#     cudaMalloc((void **) &gpu_a4, sizeof(float)*p*m);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,p,m,&alpha,gpuu_a2,m,gpu_a3,m,&beta,gpu_a4,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a4,gpu_a4, sizeof(float)*p*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a4,"A4.txt",p,m);
#  
#     //calc A5,X*A4
#     cudaMallocHost((void **) &cpu_a5, sizeof(float)*p*p);
#     cudaMallocHost((void **) &cpuu_a5, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpu_a5, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpuu_a5, sizeof(float)*p*p);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,p,m,&alpha,gpu_x,p,gpu_a4,m,&beta,gpu_a5,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a5,gpu_a5, sizeof(float)*p*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a5,"A5.txt",p,p);
#  
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,p,p,&alpha,gpu_mi,p,gpu_a5,p,&beta,gpuu_a5,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpuu_a5,gpuu_a5, sizeof(float)*p*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpuu_a5,"finalA5.txt",p,p);
#              
#     for(int row=0;row<p;row++){
#         for(int col=0;col<p;col++){
#             cpu_mi[row*p+col]=cpu_mi[row*p+col]-cpuu_a5[row*p+col];
#             }
#         }
#     //savetofile(cpu_mi,"SubA5.txt",p,p);
#     cudaMemcpy(gpu_mi,cpu_mi, sizeof(float)*p*p, cudaMemcpyHostToDevice);
#      
#     //calc A6
#     cudaMallocHost((void **) &cpu_a6, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpu_a6, sizeof(float)*m*l);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,m,p,&alpha,gpu_B,l,gpu_x,p,&beta,gpu_a6,l); 
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a6,gpu_a6, sizeof(float)*m*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a6,"A6.txt",m,l);
#      
#     cudaMallocHost((void **) &cpu_ty, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpu_ty, sizeof(float)*m*l);
#     cudaMemcpy(gpu_ty,cpu_trl+3*m*l, sizeof(float)*m*l, cudaMemcpyHostToDevice);
#     cudaMemcpy(cpu_ty,gpu_ty, sizeof(float)*m*l, cudaMemcpyDeviceToHost);
#  
#     for(int row=0;row<m;row++){
#         for(int col=0;col<l;col++){
#             cpu_ty[row*l+col]=cpu_ty[row*l+col]-cpu_a6[row*l+col];
#         }
#       }
#     //savetofile(cpu_ty,"SubA6.txt",m,l);
#      
#     //final beta
#     cudaMallocHost((void **) &cpu_f, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpu_cc, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpuu_a3, sizeof(float)*m*m);
#     cudaMemcpy(gpuu_a3,cpu_a3, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#     cudaMemcpy(gpu_cc,cpu_ty, sizeof(float)*m*l, cudaMemcpyHostToDevice);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,p,m,&alpha,gpu_cc,l,gpuu_a3,m,&beta,gpu_B,l);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_f,gpu_B, sizeof(float)*p*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_f,"finalopt.txt",p,l);       
#      
#     //////////////////////batch 5 final batch
#     //activation,relu
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,m,k,&alpha,gpu_h,p,gpu_tr+4*m*k,k,&beta,gpu_x,p); 
#     cublasDestroy(handle); 
#     cudaMemcpy(cpu_x,gpu_x, sizeof(float)*m*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_x,"mamul2.txt",m,p); 
#     relu(cpu_x,m,p); //output of activation function
#     //savetofile(cpu_x,"relu2.txt",m,p);
#      
#     //calc a3,m*xt
#     cudaMallocHost((void **) &cpu_a3, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a3, sizeof(float)*m*m);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,p,p,&alpha,gpu_x,p,gpu_mi,p,&beta,gpu_a3,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a3,gpu_a3, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a3,"A3.txt",m,m);
#      
#     //calc a1,x*m
#     cudaMallocHost((void **) &cpu_a1, sizeof(float)*m*p);
#     cudaMalloc((void **) &gpu_a1, sizeof(float)*m*p);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,m,p,&alpha,gpu_mi,p,gpu_x,p,&beta,gpu_a1,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a1,gpu_a1, sizeof(float)*m*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a1,"A1.txt",m,p);
#      
#     //calc a2,a1*xt
#     cudaMallocHost((void **) &cpu_a2, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a2, sizeof(float)*m*m);
#  
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,m,p,&alpha,gpu_x,p,gpu_a1,p,&beta,gpu_a2,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a2,gpu_a2, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a2,"A2.txt",m,m);
#      
#     //addition
#     for(int row=0;row<m;row++){
#         for(int col=0;col<m;col++){
#             cpu_a2[row*m+col]=cpu_a2[row*m+col]+cpu_I2[row*m+col];
#           }
#       }
#     //savetofile(cpu_a2,"Add.txt",m,m);
#      
#     //inverse
#     cudaMallocHost((void **) &cpu_a2i, sizeof(float)*m*m);
#     cudaMalloc((void **) &gpu_a2i, sizeof(float)*m*m);
#     cudaMemcpy(gpu_a2i,cpu_a2, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#     for(int i = 0; i<m; i++){
#         nodiag_normalize <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#         diag_normalize <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#  		    gaussjordan <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#  		    set_zero <<<numBlocks, threadsPerBlock >>>(gpu_a2i,gpu_I2,m,i);
#       }
#     cudaMemcpy(cpu_a2i,gpu_I2, sizeof(float)*m*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a2i,"XMXTinverse.txt",m,m);
#      
#     //calc A4,A3*A2
#     cudaMallocHost((void **) &gpuu_a2, sizeof(float)*m*m);
#     cudaMemcpy(gpuu_a2,cpu_a2i, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#  
#     cudaMallocHost((void **) &cpu_a4, sizeof(float)*p*m);
#     cudaMalloc((void **) &gpu_a4, sizeof(float)*p*m);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,p,m,&alpha,gpuu_a2,m,gpu_a3,m,&beta,gpu_a4,m);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a4,gpu_a4, sizeof(float)*p*m, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a4,"A4.txt",p,m);
#  
#     //calc A5,X*A4
#     cudaMallocHost((void **) &cpu_a5, sizeof(float)*p*p);
#     cudaMallocHost((void **) &cpuu_a5, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpu_a5, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpuu_a5, sizeof(float)*p*p);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,p,m,&alpha,gpu_x,p,gpu_a4,m,&beta,gpu_a5,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a5,gpu_a5, sizeof(float)*p*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a5,"A5.txt",p,p);
#  
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,p,p,&alpha,gpu_mi,p,gpu_a5,p,&beta,gpuu_a5,p);
#     cublasDestroy(handle);
#     cudaMemcpy(cpuu_a5,gpuu_a5, sizeof(float)*p*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpuu_a5,"finalA5.txt",p,p);
#              
#     for(int row=0;row<p;row++){
#         for(int col=0;col<p;col++){
#             cpu_mi[row*p+col]=cpu_mi[row*p+col]-cpuu_a5[row*p+col];
#             }
#         }
#     //savetofile(cpu_mi,"SubA5.txt",p,p);
#     cudaMemcpy(gpu_mi,cpu_mi, sizeof(float)*p*p, cudaMemcpyHostToDevice);
#      
#     //calc A6
#     cudaMallocHost((void **) &cpu_a6, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpu_a6, sizeof(float)*m*l);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,m,p,&alpha,gpu_B,l,gpu_x,p,&beta,gpu_a6,l); 
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_a6,gpu_a6, sizeof(float)*m*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_a6,"A6.txt",m,l);
#      
#     cudaMallocHost((void **) &cpu_ty, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpu_ty, sizeof(float)*m*l);
#     cudaMemcpy(gpu_ty,cpu_trl+4*m*l, sizeof(float)*m*l, cudaMemcpyHostToDevice);
#     cudaMemcpy(cpu_ty,gpu_ty, sizeof(float)*m*l, cudaMemcpyDeviceToHost);
#  
#     for(int row=0;row<m;row++){
#         for(int col=0;col<l;col++){
#             cpu_ty[row*l+col]=cpu_ty[row*l+col]-cpu_a6[row*l+col];
#         }
#     }
#     //savetofile(cpu_ty,"SubA6.txt",m,l);
#      
#     //final beta
#     cudaMallocHost((void **) &cpu_f, sizeof(float)*p*p);
#     cudaMalloc((void **) &gpu_cc, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpuu_a3, sizeof(float)*m*m);
#     cudaMemcpy(gpuu_a3,cpu_a3, sizeof(float)*m*m, cudaMemcpyHostToDevice);
#     cudaMemcpy(gpu_cc,cpu_ty, sizeof(float)*m*l, cudaMemcpyHostToDevice);
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,p,m,&alpha,gpu_cc,l,gpuu_a3,m,&beta,gpu_B,l);
#     cublasDestroy(handle);
#     cudaMemcpy(cpu_f,gpu_B, sizeof(float)*p*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_f,"finalopt.txt",p,l);   
#      
#     //testing part 
#     float *cpu_o,*gpu_o;
#     cudaMallocHost((void **) &cpu_o, sizeof(float)*m*p);
#     cudaMalloc((void **) &gpu_o, sizeof(float)*m*p);
#  
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,p,m,k,&alpha,gpu_h,p,gpu_tt,k,&beta,gpu_o,p); 
#     cublasDestroy(handle); 
#     cudaMemcpy(cpu_o,gpu_o, sizeof(float)*m*p, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_o,"testop.txt",m,p); 
#     relu(cpu_o,m,p); //output of activation function
#     //savetofile(cpu_o,"testrelu.txt",m,p);
#    
#     float *cpu_y,*gpu_y,*g_fy;
#     cudaMalloc((void **) &g_fy, sizeof(float)*m*l);
#     cudaMalloc((void **) &gpu_y, sizeof(float)*m*p);
#     cudaMemcpy(gpu_y,cpu_o, sizeof(float)*m*p, cudaMemcpyHostToDevice);
#     cudaMallocHost((void **) &cpu_y, sizeof(float)*m*l);
#      
#     cublasCreate(&handle);
#     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,p,p,&alpha,gpu_B,p,gpu_y,m,&beta,g_fy,l);
#     cublasDestroy(handle); 
#     cudaMemcpy(cpu_y,g_fy, sizeof(float)*m*l, cudaMemcpyDeviceToHost);
#     //savetofile(cpu_y,"finaltest.txt",m,l);
#     //calculation of accuracy
#     float c;  
#     for(int i=0;i<m;++i){
#         for(int j=0;j<l;++j){
#             if(cpu_y[i*l+j]==cpu_ttl[i*l+j]){
#                 c+=1.0;
#            }
#         }
#     }
#     float acc;
#     acc=c/1000000;
#     std::ofstream myfile;
#     myfile.open ("Accuracy.txt");
#     myfile << acc;
#     myfile.close();
#     printf("Accuracy is %f\t",acc);
#     cudaFreeHost(cpu_h);
#     cudaFreeHost(cpu_I1);
#     cudaFreeHost(cpu_I2);
#     cudaFreeHost(cpu_tr);
#     cudaFreeHost(cpu_trl);
#     cudaFreeHost(cpu_tt);
#     cudaFreeHost(cpu_ttl);
#     cudaFreeHost(cpu_x);
#     cudaFreeHost(cpu_m);
#     cudaFreeHost(cpu_mi);
#     cudaFreeHost(cpu_xty);
#     cudaFreeHost(cpu_B);
#     cudaFreeHost(cpu_a3);
#     cudaFreeHost(cpu_a1);
#     cudaFreeHost(cpu_a2);
#     cudaFreeHost(cpu_a2i);
#     cudaFreeHost(cpu_a4);
#     cudaFreeHost(cpu_a5);
#     cudaFreeHost(cpuu_a5);
#     cudaFreeHost(cpu_a6);
#     cudaFreeHost(cpu_ty);
#     cudaFreeHost(cpu_f);
#     cudaFreeHost(cpu_o);
#     cudaFreeHost(cpu_y);
#     cudaFree(gpu_y);
#     cudaFree(gpu_r);
#     cudaFree(g_fy);
#     cudaFree(gpu_o);
#     cudaFree(gpu_cc);
#     cudaFree(gpu_a6);
#     cudaFree(gpu_a5);
#     cudaFree(gpuu_a5);
#     cudaFree(gpu_xty);
#     cudaFree(gpu_h);
#     cudaFree(gpu_I1);
#     cudaFree(gpu_I2);
#     cudaFree(gpu_tr);
#     cudaFree(gpu_trl);
#     cudaFree(gpu_tt);
#     cudaFree(gpu_ttl);
#     cudaFree(gpu_x);
#     cudaFree(gpu_m);
#     cudaFree(gpu_mi);
#     cudaFree(gpu_B);
#     cudaFree(gpu_a3);
#     cudaFree(gpu_a1);
#     cudaFree(gpu_a2);
#     cudaFree(gpu_a2i);
#     cudaFree(gpu_a4);
#     cudaFree(gpuu_a2);
#     cudaFree(gpu_ty);    
#     return 0;
# }

!nvcc -o /content/src/my_curand1 /content/src/my_curand1.cu -lcurand -lcublas
!/content/src/my_curand1
