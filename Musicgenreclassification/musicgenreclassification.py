# Commented out IPython magic to ensure Python compatibility.
#import libraries
import pandas as pd
import numpy as np
from google.colab import drive
import IPython.display as ipd
import os,librosa,re,itertools,zipfile,operator
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import librosa.display
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from scipy.io import wavfile
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_curve,auc
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder,OneHotEncoder
import seaborn as sns
from sklearn import preprocessing,metrics
import sklearn.utils
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn
# %matplotlib inline
 
#mount drive so we can use our own files 
#i am using cloab to store my data files so if running locally then comment the drive.mount part or if using colab then uncomment 
#also change path details as mentioned in report prprojectcodeexecution.pdf 
drive.mount('/content/drive/',force_remount=True)
#done for just one file only have to fit a loop for all  fles in similar way

#if running from scratch that is preporcessing and extraction then install the eye3d and librosa library and other libraries as mentioned
#also uncomment the below lines for the same
#this package is used to get the informatio on file properties and load the adio files
#!pip install eyed3
#import eyed3

#this fucntion is used to replace the genre with the given value s
def check_replace(cdf,title,df):
  for i in cdf["title"].unique():
    for j in range(0,len(df)):
      if operator.contains(df["labels"][j],i): 
        df["labels"][j]='{}'.format(title)
  return df

#function to replace genre values which were not caught using check_replace function
#so its a double check to ensure only required genre are kept 
def genre_replace(genre,value,df):
  for i in genre:
    df['labels'] = df['labels'].replace(i,value,regex=True)
  return df

#function to unzip data into a directory 
def extract_data(path):
  #extract zip file contents
  zip_ref = zipfile.ZipFile(path, 'r')
  zip_ref.extractall("/devvoice")
  zip_ref.close()

#get data from extracted files and path 
#this function also adds the label information too
def get_data(path,value):
  #here value is the name of the folder and path is the file path
  data=[]
  labels=[]
  entries =  os.listdir("/{}/{}".format(path,value))
  for i in entries:
    data.append("/{}/{}/{}".format(path,value,i))
    labels.append(value)

  #create dataframe
  lists = [data,labels]
  df = pd.concat([pd.Series(x) for x in lists], axis=1)
  df.columns=["data","labels"]

  return df

#get fma small data and preprocess the data merge the genre into parent genre and label the data
def fma_small():
  #get genre information from the files and convert them into a dataframe
  data=[]
  labels=[]
  #get the folders inside fma_small folder
  entries =  os.listdir("/devvoice/fma_small/")
  #check if value is digit 
  p=[s for s in entries if s.isdigit()]
  #get all folder one by one preprocess them,read file properties and add label to audio files
  for i in p:
    entries =  os.listdir("/devvoice/fma_small/{}/".format(i))
    for  j in entries:
      audiofile = eyed3.load('/devvoice/fma_small/{}/{}'.format(i,j))
      data.append("/devvoice/fma_small/{}/{}".format(i,j))
      labels.append('{}'.format(audiofile.tag.genre))
      #print(audiofile.tag.genre)
      #print(audiofile.tag.title)
      #print(audiofile.tag.artist)
      #print(audiofile.tag.album)
      #print(audiofile.tag.album_artist)
      #print(audiofile.tag.track_num)

  #create dataframe
  lists = [data,labels]
  df = pd.concat([pd.Series(x) for x in lists], axis=1)
  df.columns=["data","labels"]

  return df

#preprocessing fma data it contains more than 161 genre, so we have merge this genre into their parent genre
#also the are few genre which are unknown so these genres are discarded
def preprocess_fma(df):
  #preprocess dataframe and remove unwanted symbols from data file 
  df['labels']=df['labels'].str.replace('\d+','')
  df["labels"]=df["labels"].str.replace("(",'')
  df["labels"]=df["labels"].str.replace(")",'')
  df["labels"]=df["labels"].str.replace("-",' ')
  df["labels"]=df["labels"].str.lower()

  #merging of genre based on id into proper groups 
  #each genre has a id and all genre with same id belong to one class so genre are merged according to their genre id
  gen=pd.read_csv('/content/drive/My Drive/genres.csv')

  #preprocess dataframe and remove unwanted symbols from genre information file
  #this file contains the mapping information to parent genre based on the genre id
  gen["title"]=gen["title"].str.replace('\d+','')
  gen["title"]=gen["title"].str.replace("(",'')
  gen["title"]=gen["title"].str.replace(")",'')
  gen["title"]=gen["title"].str.replace("-",' ')
  gen["title"]=gen["title"].str.lower()

  #gnum are the different genre id  and genr are the major genre out which at the end we select only 8 genres
  gnum=[2,3,4,5,9,10,12,15,17,1235,21,20,8,38,14,13]
  genr=["international","blues","jazz","classical",
        "country","pop","rock","electronic","folk",
        "instrumental","hip hop","musical theater","Old-Time / Historic only one",
        "experimental","deep funk","jazz"]

  #we are replacing the genre based on genre id information with their parent genre in genr
  for i in range(0,len(gnum)):
    df1=gen[gen["top_level"]==gnum[i]][["title"]]
    df1.reset_index(drop=True,inplace=True)
    df=check_replace(df1,genr[i],df)
    df.reset_index(drop=True,inplace=True)

  #few child genre are not caught while replacing them so then the information for these genre was collected,
  #the genre were identifed and placed into different categories and then the genre were replace with their parent genre

  #rock
  rock=['alienated','psychedelic','anticapitalist','netlabel','visual','stoner','rootsstep',
        'riding','revival','other','other   misc','rock   misc','leader','jealousy','doom',
        'darkwave','chuckberryguitar','b.a.n.g']
  df=genre_replace(rock,'rock',df)

  #electronic
  el=['drum n bass','chillout','drum and bass','breakcore','evident core','electro',
      'indietronic','tech_electro','tripstep','electronicnic','brass band','trance',
      'upitup records','acid','altema records','arena hair chip','chip','surrism phonoethics',
      'mmba','micromusic','kak r phonic','future blap','future roots','club','chillstep',
      'breaks']
  df=genre_replace(el,"electronic",df)

  #experimental
  exp=['avantgarde','field recording','fieldrecording','collage',
       'ghetto blop experiments','acoustic']
  df=genre_replace(exp,'experimental',df)

  #international
  inter=['reggae dub','brazil','greek','reggae','reggae dub','indie',
         'no weird america','weirdo carnival', 'world','urban','project.co.uk   part ',
         'patriotic','breaking dub','cosmic analog dub','dub','dub fusion','loving dub',
         'shadow dub','shadow international','international fusion','loving international',
         'song poem','\ufffe䔀氀攀挀琀爀漀\u2000䐀甀戀']

  df=genre_replace(inter,'international',df)

  #classical
  folk=['contemporary','ethnic','orchestral','singer/songwriter','story','guitarhook','erstwhile',
        "children's music"]
  df=genre_replace(folk,'folk',df)

  #hip hop
  hip=['abstract','alternatif', 'alternative','plunderphonic','plunderphonics','hip hops',
       'unknown','remix','guitarhook']
  df=genre_replace(hip,'hip hop',df)

  #deep funk
  funk=['soul randb','new plastic soul','hornsection']
  df=genre_replace(funk,'deep funk',df)

  #instrumental
  ins=['none','soundscapes']
  df=genre_replace(ins,'instrumental',df)

  #jazz
  jazz=['weather','piano']
  df=genre_replace(jazz,'jazz',df)

  return df

#this function is used extract the zip files and place them in one folder
def extract_data_folder(path,value):
  #extract zip file contents
  zip_ref = zipfile.ZipFile(path, 'r')
  zip_ref.extractall("/devvoice/{}".format(value))
  zip_ref.close()

#get genre data from a new genre folder 
#this fucntion is used to extract genre information from the folder 
#we have many audio files from differnet zip files so preprocessing of each audio files is diffferent
#so preprocessing function are created accodringly 
def get_genre_au():
  data=[]
  labels=[]
  list1=["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
  for i in list1:
    entries =  os.listdir("/devvoice/genres/{}/".format(i))
    for  j in entries:
      data.append("/devvoice/genres/{}/{}".format(i,j))
      labels.append('{}'.format(i))

  #create dataframe
  lists = [data,labels]
  df = pd.concat([pd.Series(x) for x in lists], axis=1)
  df.columns=["data","labels"]

  return df

#function to generate the combined data
#this function is used to  combine the output extracted from all the zip files and place them in one dataframe
#also the csv file of each dataset extracted is saved and finally the combined output of all the files is saved in a csv file
def gen_com_data():
  #preprocess fma_small data
  df_small=extract_data("/content/drive/My Drive/fma_small (1).zip")
  df_small=fma_small()
  df_small=preprocess_fma(df_small)
  df_small.to_csv("fmasmall.csv",index=False)

  #preprocess data from electronic.zip
  extract_data('/content/drive/My Drive/electronic.zip')
  df_electronic=get_data("devvoice","electronic")
  df_electronic.to_csv("electronic.csv",index=False)

  #preprocess data from folkcountry.zip
  extract_data('/content/drive/My Drive/folkcountry.zip')
  df_folk_country=get_data("devvoice","folkcountry")
  df_folk_country.to_csv('flok.csv',index=False)

  #preprocess data from funksoulrnb.zip
  extract_data('/content/drive/My Drive/funksoulrnb.zip')
  df_funk_soul=get_data("devvoice","funksoulrnb")
  df_funk_soul.to_csv('funck_soul.csv',index=False)

  #preprocess data from rock.zip
  extract_data('/content/drive/My Drive/rock.zip')
  df_rock=get_data("devvoice","rock")
  df_rock.to_csv('rock.csv',index=False)

  #preprocess data from alternative.zip
  extract_data_folder('/content/drive/My Drive/alternative.zip','alternative')
  df_alternative=get_data('devvoice','alternative')
  df_alternative.to_csv("alternative.csv",index=False)

  #preprocess data from blues.zip
  extract_data_folder('/content/drive/My Drive/blues.zip','blues')
  df_blues=get_data('devvoice','blues')
  df_blues.to_csv('blues.csv',index=False)

  #preprocess data from jazz.zip
  extract_data_folder('/content/drive/My Drive/jazz.zip','jazz')
  df_jazz=get_data('devvoice','jazz')
  df_jazz.to_csv('jazz.csv',index=False)

  #preprocess data from pop.zip
  extract_data_folder('/content/drive/My Drive/pop.zip','pop')
  df_pop=get_data('devvoice','pop')
  df_pop.to_csv('pop.csv',index=False)

  #preprocess data from hiphop.zip
  extract_data_folder('/content/drive/My Drive/raphiphop.zip','hip hop')
  df_hiphop=get_data('devvoice','hip hop')
  df_hiphop.to_csv('hiphop.csv',index=False)

  #preprocess data from archive.zip
  extract_data('/content/drive/My Drive/archive.zip')
  df_genre=get_genre_au()
  df_genre.to_csv("genredata.csv",index=False)

  #create dataframe
  lists = [df_small,df_funk_soul,df_electronic,df_folk_country,df_rock,df_alternative,df_blues,df_jazz,df_pop,df_hiphop,df_genre]
  final_df = pd.concat(lists, axis=0)
  final_df.columns=["data","labels"]
  final_df.reset_index(drop=True,inplace=True)

  #this is the final file generated by combing the output of all the above given files
  final_df.to_csv("final_dataframe.csv",index=False)

  return final_df

#different features are extracted form the audio files and a data file is created on whih the models are built
def features_extract(new_df):
  #zero crossing rate
  zcr_final = np.zeros((len(new_df)))
  #chroma shift
  cs_final = np.zeros((len(new_df)))
  #spectral rolloff
  rolloff_final = np.zeros((len(new_df)))
  #rmse 
  RMSE_final = np.zeros((len(new_df)))
  #mfcc
  mfcc_final = np.zeros((len(new_df), 20))
  #spectral centroids
  centroids_final = np.zeros((len(new_df)))
  #spectral contrast
  contrast_final = np.zeros((len(new_df)))
  #spectral bandwidth
  bandwidth_final = np.zeros((len(new_df)))
  #tonetz
  tonnetz_final = np.zeros((len(new_df)))
  corrupted=[]
  for i in range(len(new_df)):
    try:
      x, sr = librosa.load(new_df["data"][i])

      zero_crossings = librosa.feature.zero_crossing_rate(x)
      zcr_final[i] = np.mean(zero_crossings)

      temp_mfcc = []
      mfccs = librosa.feature.mfcc(x, sr)
      for j in mfccs:
        temp_mfcc.append(np.mean(j))
      mfcc_final[i] = np.array(temp_mfcc)

      chroma_shift = librosa.feature.chroma_stft(x, sr)
      cs_final[i] = np.mean(chroma_shift)

      rolloff = librosa.feature.spectral_rolloff(x, sr)
      rolloff_final[i] = np.mean(rolloff)

      RMSE = librosa.feature.rmse(x)
      RMSE_final[i] = np.mean(RMSE)

      spectral_centroids = librosa.feature.spectral_centroid(x, sr)
      centroids_final[i] = np.mean(spectral_centroids)

      spectral_contrast = librosa.feature.spectral_contrast(x, sr)
      contrast_final[i] = np.mean(spectral_contrast)

      spectral_bandwidth = librosa.feature.spectral_bandwidth(x, sr)
      bandwidth_final[i] = np.mean(spectral_bandwidth)

      tonnetz = librosa.feature.tonnetz(x, sr)
      tonnetz_final[i] = np.mean(tonnetz)

    except:
      print("Corrputed File Path: {}".format(new_df["data"][i]))
      corrupted.append(new_df["data"][i])
  new_df = new_df.assign(ZeroCrossingsRate=zcr_final, ChromaShift=cs_final, SpectralRolloff=rolloff_final,
                       RMSEnergy=RMSE_final, SpectralCentroids=centroids_final, SpectralContrast=contrast_final,
                       SpectralBandwidth=bandwidth_final, Tonnetz=tonnetz_final, MFCC0 = mfcc_final[:, 0], 
                       MFCC1 = mfcc_final[:, 1], MFCC2 = mfcc_final[:, 2], MFCC3 = mfcc_final[:, 3],
                       MFCC4 = mfcc_final[:, 4], MFCC5 = mfcc_final[:, 5], MFCC6 = mfcc_final[:, 6],
                       MFCC7 = mfcc_final[:, 7], MFCC8 = mfcc_final[:, 8], MFCC9 = mfcc_final[:, 9],
                       MFCC10 = mfcc_final[:, 10], MFCC11 = mfcc_final[:, 11], MFCC12 = mfcc_final[:, 12],
                       MFCC13 = mfcc_final[:, 13], MFCC14 = mfcc_final[:, 14], MFCC15 = mfcc_final[:, 15],
                       MFCC16 = mfcc_final[:, 16], MFCC17 = mfcc_final[:, 17], MFCC18 = mfcc_final[:, 18],
                       MFCC19 = mfcc_final[:, 19])

  #this is the final dataset that is used to generate models
  new_df.to_csv('fma_featureextraction.csv', sep=',', index=False, header=True)
  #also a corrupted file is created to store the files which were corrupted from the list of the files
  #it also sotre the list of the files which could not be processed so then these files are inspected 
  #after inspection if these files could be fixed then they are fixed or else discarded
  corrupted=pd.DataFrame(corrupted)
  corrupted.to_csv('corrupted.csv',index=False)
  return new_df,corrupted

#the final_data and final_df are two function that do data preprocessing and feature extraction
#we have preprocessed and extracted all the feature so commented these function if you want to run the 
#data preprocessing and feature extraction step then uncomment and run these function 
#also change paths and it requires more than 8 hours to extract for all the audio files
#final_data=gen_com_data()
#final_df,corr=features_extract(final_data)

#below codes were used during preprocssing to manually the list of genre which can be merged to parent genre 
#below code hepled in identifying the genre which could be merged together, it also helped in identifying whihc genre are merged and which are remaining
#check for the labels which are not there in the list1 
#check=np.setdiff1d(df["labels"],genr) 
#check=check.tolist()
#df1 = df[~df['labels'].isin(check)]
#df1.reset_index(drop=True,inplace=True)

#function to process the genre file and get final merged data
#this function reads the genre file and then after this final preprocessing setp data is ready for building models
def get_genre_file():
  #change path here
  #this file contains the data
  df=pd.read_csv('/content/drive/MyDrive/PR Project/datasets/fma_featureextraction.csv')
  df["labels"]=df["labels"].str.replace('folkcountry','folk',regex=True)
  df["labels"]=df["labels"].str.replace('reggae','international',regex=True)
  funk=['funksoulrnb','disco']
  df=genre_replace(funk,'deep funk',df)
  df["labels"]=df["labels"].str.replace('metal','rock',regex=True)
  hip=['alternative','hip hop']
  df=genre_replace(hip,'hiphop',df)

  df["labels"].unique()
  l_remove=['real','???','deep funk','classwar karaoke',
            'genre inconnu','musical theater','genre','cinematic',
            'dont brock','film music','goals','fungussonambulus','top ',
            'country','blues', 'jazz', 'classical']
  #remove rows
  df.drop(df[df['labels'].isin(l_remove)].index, inplace = True) 
  df.reset_index(drop=True,inplace=True)
  #replace empty rows with nan
  df.replace("","nan", inplace=True)
  #remove nan rows
  df.drop(df[df['labels'].isin(['nan'])].index, inplace = True) 
  df.reset_index(drop=True,inplace=True)

  return df

#random forest classifier model
def random_forest_classifier():
  #get the data
  df=get_genre_file()
  #for plotting take labels and class names
  class_names=df['labels'].unique()
  tick_marks = np.arange(len(df["labels"].unique()))
  # Create correlation matrix
  corr_matrix = df.corr().abs()
  #Using Pearson Correlation plot the correlation output
  plt.figure(figsize=(5,5))
  cor = df.corr()
  sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
  plt.show()
  # Select upper triangle of correlation matrix
  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
  # Find features with correlation greater than 0.99
  to_drop = [column for column in upper.columns if any(upper[column] > 0.99)]
  #drop the highly correlated feature
  df = df.drop(columns=to_drop, axis=0)
  #shuffle the dataset
  df = sklearn.utils.shuffle(df)   
  #drop the columns which are not required
  df.drop(columns=["data"],inplace=True)
  #normalize the dataset
  df[df.columns[1:]]=MinMaxScaler().fit_transform(df[df.columns[1:]])
  #split dataset into train and test
  test = df.groupby('labels').apply(lambda x:x.sample(int(len(x)/5))).reset_index(drop=True)
  train=df.loc[df.index.difference(test.index), ]
  train.reset_index(drop=True,inplace=True)
  test.reset_index(drop=True,inplace=True)
  X_train=train[train.columns[1:]]
  X_test=test[test.columns[1:]]
  y_train=train[train.columns[:1]]
  y_test=test[test.columns[:1]]
  print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
  #apply dimensionality reduction and take the best varaince or information
  pca = PCA(n_components=0.95, svd_solver='full')
  X_train = pca.fit_transform(X_train)
  X_test = pca.transform(X_test)
  #Create a Gaussian Classifier
  clf=RandomForestClassifier(n_estimators=1000,criterion='gini',verbose=1,n_jobs=len(train.columns),max_features='auto')
  clf.fit(X_train,np.ravel(y_train))
  #Train the model using the training sets y_pred=clf.predict(X_test)
  tr=clf.predict(X_train)
  print("Random Forest Classifier Training Accuracy:",metrics.accuracy_score(y_train,tr))
  y_pred=clf.predict(X_test)
  print("Random Forest Classifier Testing Accuracy:",metrics.accuracy_score(y_test, y_pred))
  # View the classification report for test data and predictions
  print("Classification report for random forest")
  print(classification_report(y_test, y_pred))
  # View confusion matrix for test data and predictions
  print("Confusion matrix for random forest")
  print(confusion_matrix(y_test, y_pred))
  confusion_matrix(y_test, y_pred)
  # Get and reshape confusion matrix data
  matrix = confusion_matrix(y_test, y_pred)
  matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
  # Build the plot of confusion matrix
  plt.figure(figsize=(16,7))
  sns.set(font_scale=1.4)
  sns.heatmap(matrix, annot=True, annot_kws={'size':5},
            cmap=plt.cm.Greens, linewidths=0.2)

  # Add labels to the plot
  tick_marks2 = tick_marks
  plt.xticks(tick_marks, class_names, rotation=25)
  plt.yticks(tick_marks2, class_names, rotation=0)
  plt.xlabel('Predicted label')
  plt.ylabel('True label')
  plt.title('Confusion Matrix for Random Forest Model')
  plt.show()

#K nearest nearest neighbour classifer model
def KNN_Classifier():
  #Reading data from csv file
  data = get_genre_file()
  data.drop(columns=["data"],inplace=True)

  test = sklearn.utils.shuffle(data.groupby('labels').apply(lambda x:x.sample(int(len(x)/5))).reset_index(drop=True))
  train = sklearn.utils.shuffle(data.loc[data.index.difference(test.index), ])

  train.reset_index(drop=True,inplace=True)
  test.reset_index(drop=True,inplace=True)
  train_data = train[train.columns[1:]]
  test_data = test[test.columns[1:]]
  train_labels = train[train.columns[:1]]
  test_labels = test[test.columns[:1]]

  #Hot encoding of labels
  # "electronic","experimental","folk","rock","hip hop","instrumental","international","pop"

  # integer encode train and test labels
  label_encoder = LabelEncoder()
  int_encode_tr = label_encoder.fit_transform(train_labels)
  int_encode_ts = label_encoder.fit_transform(test_labels)

  #hot encode train and test labels
  onehot_encoder = OneHotEncoder(sparse=False)
  int_encode_tr = int_encode_tr.reshape(len(int_encode_tr), 1)
  train_labels = onehot_encoder.fit_transform(int_encode_tr)
  int_encode_ts = int_encode_ts.reshape(len(int_encode_ts), 1)
  test_labels = onehot_encoder.fit_transform(int_encode_ts)

  #converting data into integer
  train_data = train_data.astype(float);
  test_data = test_data.astype(float);
  train_labels = train_labels.astype(int);
  test_labels = test_labels.astype(int);

  int_encode_tr = int_encode_tr.reshape(len(train_data),)
  int_encode_ts = int_encode_ts.reshape(len(test_data),)

  label_counter_tr = np.zeros(8)
  for i in range(len(int_encode_tr)):
    label_counter_tr[int_encode_tr[i]] = label_counter_tr[int_encode_tr[i]] + 1

  label_counter_ts = np.zeros(8)
  for i in range(len(int_encode_ts)):
    label_counter_ts[int_encode_ts[i]] = label_counter_ts[int_encode_ts[i]] + 1

  #printing number of data for train and test from each class
  print("=====================================================================================================")
  print("Train data per class : ",label_counter_tr)
  print("Test data per class  : ",label_counter_ts)
  print("Total data per class : ",label_counter_tr+label_counter_ts)
  print("=====================================================================================================")

  #======================== Running KNN for K = 1 to K = 21 ===================================================
  #KNN
  k = 21
  scores = np.zeros(k)
  for i in range(1,k):
    neighbors = KNeighborsClassifier(n_neighbors = i)
    neighbors.fit(train_data,int_encode_tr)
    pred_test_labels = neighbors.predict(test_data)
    scores[i] = metrics.accuracy_score(int_encode_ts,pred_test_labels)*100
  max_ind = np.argmax(scores)

  neighbors = KNeighborsClassifier(n_neighbors = max_ind)
  neighbors.fit(train_data,int_encode_tr)
  pred_train_labels = neighbors.predict(train_data)
  train_acc = metrics.accuracy_score(int_encode_tr,pred_train_labels)*100

  #printing training accuracy and report
  print("\nTraining accuracy without PCA : ",train_acc)
  print("\nClassification report : ")
  print(metrics.classification_report(int_encode_tr, pred_train_labels))
  print("\nConfusion matrix : ")
  print(metrics.confusion_matrix(int_encode_tr,pred_train_labels))

  neighbors = KNeighborsClassifier(n_neighbors = max_ind)
  neighbors.fit(train_data,int_encode_tr)
  pred_test_labels = neighbors.predict(test_data)
  test_acc = metrics.accuracy_score(int_encode_ts,pred_test_labels)*100

  #printing testing accuracy and report
  print("=====================================================================================================")
  print("\nTesting  accuracy without PCA : ",test_acc)
  print("\nClassification report : ")
  print(metrics.classification_report(int_encode_ts, pred_test_labels))
  print("\nConfusion matrix : ")
  print(metrics.confusion_matrix(int_encode_ts,pred_test_labels))


  #PCA-KNN
  k = 21
  scores = np.zeros(k)
  pca = PCA(n_components = 24)
  pca.fit(train_data)
  pca_train_data = pca.transform(train_data)
  pca_test_data = pca.transform(test_data)

  for i in range(1,k):
    neighbors = KNeighborsClassifier(n_neighbors = i)
    neighbors.fit(pca_train_data,int_encode_tr)
    pred_test_labels = neighbors.predict(pca_test_data)
    scores[i] = metrics.accuracy_score(int_encode_ts,pred_test_labels)*100
  print("=====================================================================================================")
  max_ind = np.argmax(scores)

  neighbors = KNeighborsClassifier(n_neighbors = max_ind)
  neighbors.fit(train_data,int_encode_tr)
  pred_train_labels = neighbors.predict(train_data)
  train_acc = metrics.accuracy_score(int_encode_tr,pred_train_labels)*100

  #printing training accuracy and report
  print("\nTraining accuracy with PCA : ",train_acc)
  print("\nClassification report : ")
  print(metrics.classification_report(int_encode_tr, pred_train_labels))
  print("\nConfusion matrix : ")
  print(metrics.confusion_matrix(int_encode_tr,pred_train_labels))

  neighbors = KNeighborsClassifier(n_neighbors = max_ind)
  neighbors.fit(train_data,int_encode_tr)
  pred_test_labels = neighbors.predict(test_data)
  test_acc = metrics.accuracy_score(int_encode_ts,pred_test_labels)*100

  #printing testing accuracy and report
  print("=====================================================================================================")
  print("\nTesting  accuracy with PCA : ",test_acc)
  print("\nClassification report : ")
  print(metrics.classification_report(int_encode_ts, pred_test_labels))
  print("\nConfusion matrix : ")
  print(metrics.confusion_matrix(int_encode_ts,pred_test_labels))

  sns.scatterplot( x = pca_train_data[:,0], y = pca_train_data[:,1], hue = int_encode_tr, palette="deep", legend="full")
  plt.show()

  fig = plt.figure()
  ax = plt.axes(projection='3d')
  ax.scatter3D( pca_train_data[:,0], pca_train_data[:,1], pca_train_data[:,2], c = int_encode_tr);
  plt.show()

#support vector machine classifier model
def SVMClassifier():
  #Reading data from csv file
  data = get_genre_file()
  data.drop(columns=["data"],inplace=True)
  data = sklearn.utils.shuffle(data)

  test = sklearn.utils.shuffle(data.groupby('labels').apply(lambda x:x.sample(int(len(x)/5))).reset_index(drop=True))
  train = sklearn.utils.shuffle(data.loc[data.index.difference(test.index), ])

  train.reset_index(drop=True,inplace=True)
  test.reset_index(drop=True,inplace=True)
  train_data = train[train.columns[1:]]
  test_data = test[test.columns[1:]]
  train_labels = train[train.columns[:1]]
  test_labels = test[test.columns[:1]]

  trainData = np.asarray(train_data)
  testData = np.asarray(test_data)
  trainLabels = np.asarray(train_labels)
  testLabels = np.asarray(test_labels)

  # SVM Classifier
  print("\nRBF Kernel")
  rbf = SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(trainData, trainLabels)
  predYtrain = rbf.predict(trainData)
  print(classification_report(trainLabels, predYtrain))
  print("Training Accuracy = {}% \n".format(accuracy_score(trainLabels, predYtrain)))
  predYtest = rbf.predict(testData)
  print(classification_report(testLabels, predYtest))
  print("Testing Accuracy = {}% \n".format(accuracy_score(testLabels, predYtest)))
    
  #Printing confusion matrix 
  confMatrix = confusion_matrix(testLabels, predYtest)
  print("Confusion matrix for RBF Kernel: ")
  print(confMatrix)

#function to call all the clasifer in sequence
def main():
  print("Random Forest Classfier for Music Genre Classification")
  random_forest_classifier()
  print("======================================================")
  print("======================================================")
  print("======================================================")
  print("======================================================")
  print("K Nearest Neighbour  Classfier for Music Genre Classification")
  KNN_Classifier()
  print("======================================================")
  print("======================================================")
  print("======================================================")
  print("======================================================")
  print("Support vector machine Classfier for Music Genre Classification")
  SVMClassifier()

main()


