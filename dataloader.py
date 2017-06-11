import numpy as np
import pandas as pd
from pandas import ExcelWriter
from itertools import chain
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.stats import skew
from bhloader import BHLoader

class AmesLoader:
	def __init__(self, x_filepath, y_filepath):
		self.tr_data_x=pd.read_csv(x_filepath)
		self.tr_data_y=pd.read_csv(y_filepath)
		#load Training_Data

		#create MinMax Scaled Data
		min_max_scaler = preprocessing.MinMaxScaler()
		np_scaled = min_max_scaler.fit_transform(self.tr_data_x)
		self.minmax_x = pd.DataFrame(np_scaled)
		np_scaled = min_max_scaler.fit_transform(self.tr_data_y)
		self.minmax_y = pd.DataFrame(np_scaled)

		self.minmax_y = np.array(self.minmax_y[1])
		self.minmax_y = self.minmax_y.reshape((len(self.minmax_y), 1))


		#Create Normal Distributed Data
		self.mean_x = np.mean(self.tr_data_x, axis=0)
		self.std_x = np.std(self.tr_data_x, axis=0)

		self.mean_y = np.mean(self.tr_data_y)
		self.std_y = np.std(self.tr_data_y)

		self.norm_x = (self.tr_data_x  - self.mean_x) / self.std_x
		self.norm_y = (self.tr_data_y - self.mean_y) / self.std_y

		self.norm_y = np.array(self.norm_y['0'])
		self.norm_y = self.norm_y.reshape((len(self.norm_y), 1))

		return 

	def loadRawData(self, filepath):
		tr_data=pd.read_csv(filepath)
		var=np.genfromtxt('data/variables.csv', dtype = None, delimiter=",")

		#tr_data
		# In[3]:

		# to determine the type of variables
		D=0; N=0; C=0; O=0; 
		listC=[]; listD=[]; listO=[];listN=[];
		for idx in range(79):
			if var[idx] == 'D':
				D+=1
				listD.append(idx+1)
			elif var[idx] =='C':
				C+=1
				listC.append(idx+1)
			elif var[idx] =='O':
				O+=1
				listO.append(idx+1)
			else:
				N+=1
				listN.append(idx+1)

		# In[4]:

		# reorder training data matrix : continuous, discrete, ordinal, nominal

		list1=list(tr_data.columns.values)
		listre=list(chain([0],listC,listD,listO,listN,[80]))
		list2=[list1[i] for i in listre]
		#tr_1=tr_data[listre]
		tr_1 = tr_data.reindex_axis(list2, axis=1)

		# tr_x(array) : 0~18 : conti// 19~32 : discrete // 33~55 : ordinal // 56~78 : nominal // 79 : price
		# tr_y(dataframe) : SalePrice 
		tr_y=tr_1['SalePrice'].values
		tr_x=tr_1.drop(['Id','SalePrice'],axis=1)


		# converting ordinal value to numerical value  
		# varlist : ordinal value list
		varlist=[['IR3','IR2','IR1','Reg'],['ELO','NoSeWa','NoSewr','AllPub'],['Sev','Mod','Gtl'],[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10],['Po','Fa','TA','Gd','Ex'],['Po','Fa','TA','Gd','Ex'],['Po','Fa','TA','Gd','Ex'],['Po','Fa','TA','Gd','Ex'],['No','Mn','Av','Gd'],['Unf','LwQ','Rec','BLQ','ALQ','GLQ'],
				['Unf','LwQ','Rec','BLQ','ALQ','GLQ'],['Po','Fa','TA','Gd','Ex'],['Mix','FuseP','FuseF','FuseA','SBrkr'],['Po','Fa','TA','Gd','Ex'],['Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'],['Po','Fa','TA','Gd','Ex'],['Unf','RFn','Fin'],['Po','Fa','TA','Gd','Ex'],['Po','Fa','TA','Gd','Ex'],['N','P','Y'],['Po','Fa','TA','Gd','Ex'],['MnWw','GdWo','MnPrv','GdPrv']]

		#worst=0 -> +1
		for idx in range(34,57):
			column=tr_x[list2[idx]].values
			for j in range(1260):
				if tr_x.isnull()[list2[idx]].values[j]:
					column[j]=column[j]
				else :
					k=varlist[idx-34].index(column[j])
					#k=varlist[idx-33].index(column[j])
					column[j]=k
			tr_x[list2[idx]]=column
		# In[7]:

		#Convert the data type of MSSubClass, since it it nominal var. even though it is expressed in discrete var. 
		tr_x['MSSubClass'] = tr_x['MSSubClass'].astype('str')


		# In[8]:

		# data normalization 
		tr_y = np.log1p(tr_y)

		numeric_feats = tr_x.dtypes[tr_x.dtypes != "object"].index

		skewed_feats = tr_x[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
		skewed_feats = skewed_feats[skewed_feats > 0.75]
		skewed_feats = skewed_feats.index

		tr_x[skewed_feats] = np.log1p(tr_x[skewed_feats])


		# In[9]:

		# By generating dummy variables, we can convert nominal var to several bianry discrete var.
		tr_x=pd.get_dummies(tr_x)
		# NA : the mean value of each columns
		tr_x=tr_x.fillna(tr_x.mean())
		# tr_x_array : value array
		tr_x_array=tr_x.values


		# In[10]:

		# Cross validation data setting
		X_train, X_test, y_train, y_test = train_test_split(tr_x_array, tr_y, test_size=0.4, random_state=0)

	def getMinMaxData(self, cross_val=False, isminibatch=True, mbSize=100):
		if isminibatch == True:
			batches_x = []
			batches_y = []
			for i in range(0, len(self.minmax_x[0]), mbSize):
				batches_x.append(self.minmax_x[i:i+mbSize])
				batches_y.append(self.minmax_y[i:i+mbSize])

			return batches_x, batches_y
		else:
			return self.minmax_x, self.minmax_y

	def getNormalizedData(self, cross_val=False):
		return self.norm_x, self.norm_y

	def preprocessData(self):
		#Normalize
		pass