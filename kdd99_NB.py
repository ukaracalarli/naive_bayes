# -*- coding: utf-8 -*-
"""
Created on Fri Aug 3 2018 06:08 
Modified by Umut KARACALARLI
Details:
    - Data is read from KDD99 csv data file
    - nonnumeric values are converted to scalar values
    - data is classified by Naive Bayes Classifier
    - Accuracy score is printed
    - Confusion matrix printed

https://github.com/ukaracalarli/Naive_Bayes is licensed under the GNU General Public License v3.0
"""
# In[10]:
# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
import os
import urllib
from sklearn.naive_bayes import GaussianNB
import time
import datetime


# In[20]:

        
""" Initalization of dataset configure predefined columns            
"""

os.chdir('C:\Users\ukara\PycharmProjects\ids_svm1')
train_data_from_text = urllib.urlopen('.\/data\/kddcup.data_10_percent_corrected')
test_data_from_text = urllib.urlopen('.\/data\/corrected')

""" Train data read from frame """
class_train = pd.read_csv(train_data_from_text, quotechar=',', skipinitialspace=True, names=['Duration', 'protocol_type', 'Service', 'Flag', 'src_bytes', 'dst_bytes', 'Land', 'wrong_fragment', 'Urgent', 'Hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'Count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate','Class'])
class_test = pd.read_csv(test_data_from_text, quotechar=',', skipinitialspace=True, names=['Duration', 'protocol_type', 'Service', 'Flag', 'src_bytes', 'dst_bytes', 'Land', 'wrong_fragment', 'Urgent', 'Hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'Count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate','Class'])


#row_count is used to do the test with complete or partial data
row_count = int(len(class_train) * 1)
#row_count = 20000


#data size is defined with row count 
class_train = class_train[:row_count]
class_test = class_test[:row_count]
   
#data frames are initialized and NaN fields are filled with 0
class_train = class_train.fillna(0)
class_test = class_test.fillna(0)
         

""" Change of train classes by tagging normal or attack """
class_train.loc[(class_train['Class'] !='normal.'),'Class'] = 'attack'
class_train.loc[(class_train['Class'] =='normal.'),'Class'] = 'normal'
""" Change of test classes of tagging normal or attack """
class_test.loc[(class_test['Class'] !='normal.'),'Class'] = 'attack'
class_test.loc[(class_test['Class'] =='normal.'),'Class'] = 'normal'
   
#list and dictionaries to be used in ordering the features by importance
list = []
dic = {}
odic= {}
   
#non numeric protocol types are vectorized
protocol_dict = {'tcp':1, 'udp':2, 'icmp':3}
class_train['protocol_type'].replace(protocol_dict, inplace=True)
class_test['protocol_type'].replace(protocol_dict, inplace=True)

#non numeric service types are vectorized
service_dict ={'http':1,
               'ecr_i':2,
               'smtp':3,
               'domain_u':4,
               'ftp_data':5,
               'private':6,
               'eco_i':7,
               'finger':8,
               'ftp':9,
               'ntp_u':10,
               'telnet':11,
               'auth':12,
               'pop_3':13,
               'other':14,
               'time ':15,
               'domain':16, 
               'rje':17,
               'gopher':18, 
               'ssh':19,
               'mtp':20,
               'login':21,
               'link':22,
               'nntp':23,
               'name':24,
               'imap4':25,
               'whois':26,
               'remote_job':27,
               'daytime':28,
               'ctf':29,
               'time':30,
               'X11':31,
               'IRC':32,
               'tim_i':33,
               'http_443':34,
               'systat':35,
               'ldap':36,
               'printer':37,
               'sunrpc':38,
               'urp_i':39,
               'Z39_50':40,
               'bgp':41,
               'vmnet':42,
               'sql_net':43,
               'netbios_dgm':44,
               'netbios_ssn':45,
               'netbios_ns':46,
               'uucp_path':47,
               'pop_2':48,
               'csnet_ns':49,
               'iso_tsap':50,
               'hostnames':51,
               'supdup':51,
               'netstat':52,
               'discard':53,
               'echo':54,
               'klogin':55,
               'kshell':56,
               'uucp':57,
               'courier':58,
               'efs':59,
               'shell':60,
               'exec':61,
               'nnsp':62,
               'red_i':63,
               'tftp_u':64,
               'pm_dump':65,
               'urh_i':66,
               'icmp':67}

class_train['Service'].replace(service_dict, inplace=True)
class_test['Service'].replace(service_dict, inplace=True)

#non numeric flag types are vectorized
flag_dict ={'SF':1,
            'REJ':2,
            'SH':3,
            'RSTR':4,
            'RSTO':5,
            'S1':6,
            'S0':7,
            'RSTOS0':8,
            'S2':9,
            'S3':10,
            'OTH':11}

class_train['Flag'].replace(flag_dict, inplace=True)
class_test['Flag'].replace(flag_dict, inplace=True)

#non numeric class types are vectorized
class_dict ={'normal':0, 'attack':1}
class_train['Class'].replace(class_dict, inplace=True)
class_test['Class'].replace(class_dict, inplace=True)


# In[25]:

f5  = [6, 11, 22, 23, 38]
f15 = [6, 11, 22, 23, 38, 37, 24, 25, 4, 35, 1, 3, 28, 26, 40]
f25 = [6, 11, 22, 23, 38, 37, 24, 25, 4, 35, 1, 3, 28, 26, 40, 27, 33, 12, 32, 39, 7, 34, 31, 36, 2]
f30 = [6, 11, 22, 23, 38, 37, 24, 25, 4, 35, 1, 3, 28, 26, 40, 27, 33, 12, 32, 39, 7, 34, 31, 36, 2, 5, 10, 30, 16, 29]
f41 = [6, 11, 22, 23, 38, 37, 24, 25, 4, 35, 1, 3, 28, 26, 40, 27, 33, 12, 32, 39, 7, 34, 31, 36, 2, 5, 10, 30, 16, 29, 13, 17, 9, 0, 15, 21, 18, 8, 14, 19, 20]

features = f5
class_train_reduced  = class_train.iloc[:, features]
class_test_reduced = class_test.iloc[:, features]
print ("reduced training set size: ",class_train_reduced.shape)
print("reduced test set size: ",class_test_reduced.shape)
#print(class_test_reduced.head)
# In[30]:

x_train = class_train.values[:,:-1]
y_train = class_train.values[:,-1]

x_test= class_test.values[:,:-1]
y_test = class_test.values[:,-1]

#class_test.values[:,:-1].shape
#class_train.iloc[:,-1] 

fx_train = class_train_reduced.values[:,:-1]
fy_train = class_train.values[:,-1]

fx_test= class_test_reduced.values[:,:-1]
fy_test = class_test.values[:,-1]

# In[40]:  
NB_start_time = time.time()
print('NB egitim baslama zamani: ',
    datetime.datetime.fromtimestamp(
        NB_start_time).strftime('%Y-%m-%d %H:%M:%S'))


nb = GaussianNB()
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)

NB_stop_time = time.time()
print 'NB egitim toplam gecen zaman: ', int(NB_stop_time - NB_start_time)



fNB_start_time = time.time()
print('fNB egitim baslama zamani: ',
    datetime.datetime.fromtimestamp(
        fNB_start_time).strftime('%Y-%m-%d %H:%M:%S'))

fnb = GaussianNB()
fnb.fit(fx_train, fy_train)
fy_pred = fnb.predict(fx_test)

fNB_stop_time = time.time()
print 'FS + NB egitim toplam gecen zaman: ', int(fNB_stop_time -fNB_start_time)


# In[50]:  

test_acc = sum(y_pred == y_test)/len(x_test)
print ("NB Accuracy :",test_acc)

ftest_acc = sum(fy_pred == fy_test)/len(fx_test)
print ("FS+NB Accuracy :",ftest_acc)

# In[60]: 
# Making the Confusion Matrix
print("")
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print "NB Results"
print ("TP", tp,"%",tp/len(y_test)*100)
print ("TN", tn,"%",tn/len(y_test)*100)
print ("FP", fp,"%",fp/len(y_test)*100)
print ("FN", fn,"%",fn/len(y_test)*100)
print ("toplam y_test", len(y_test))
print("tp+tn+fp+fn",tp+tn+fp+fn)
print("toplam oran",tp/len(y_test)+tn/len(y_test)+fp/len(y_test)+fn/len(y_test))
# In[70]: 
print("")
print "FS + NB Results"
from sklearn.metrics import confusion_matrix
ftn, ffp, ffn, ftp = confusion_matrix(fy_test, fy_pred).ravel()
print ("TP", ftp,"%",ftp/len(fy_test)*100)
print ("TN", ftn,"%",ftn/len(fy_test)*100)
print ("FP", ffp,"%",ffp/len(fy_test)*100)
print ("FN", ffn,"%",ffn/len(fy_test)*100)
print ("toplam fy_test", len(fy_test))
print("tp+tn+fp+fn",ftp+ftn+ffp+ffn)
print("toplam oran",ftp/len(fy_test)+ftn/len(fy_test)+ffp/len(fy_test)+ffn/len(fy_test))