import pandas as pd


def dataprocess(df):
    #df=pd.read_csv('../dataset/happiness_train.csv')
    df=df.drop(['work_status','work_yr','work_type','work_manage','survey_time'],axis=1)
    df=df.fillna(0)#可以指定value=dict,serise,dataframe,反正就是key-value对
    #数据归约没有做
    label=df['happiness']
    feature=df.drop('happiness',axis=1)
    print(feature.isnull().sum())
    return  label,feature
def dataprocesst(df):
    #df=pd.read_csv('../dataset/happiness_train.csv')
    df=df.drop(['work_status','work_yr','work_type','work_manage','survey_time'],axis=1)
    feature=df.fillna(0)#可以指定value=dict,serise,dataframe,反正就是key-value对
    #数据归约没有做
    return feature

def  modeling(label_tr,feature_tr,feature_t):
    from sklearn.metrics import accuracy_score,recall_score,f1_score,mean_squared_error
    from sklearn.linear_model import LinearRegression
    models=[]
    models.append(('LR',LinearRegression()))
    #转换为dict,再用value选择值
    '''
    models=dict(models)
    modelu=models[method]
    '''
    for clsn,clsf in models:
        clsf.fit(feature_tr,label_tr)
        pdi=clsf.predict(feature_t)
        output=pd.DataFrame(pdi,index=[x for x in range(8001,len(feature_t)+8001)],columns=['output'])#告诉程序数据结构是什么很重要，关系到内存
        pd.DataFrame.to_csv(output,'D:\Tianchi1\dataset\happiness_submit.csv')

def main():
    dftr=pd.read_csv('D:/Tianchi1/dataset/happiness_train_abbr.csv')
    dft=pd.read_csv('D:/Tianchi1/dataset/happiness_test_abbr.csv')
    labeltr,featuretr=dataprocess(dftr)
    featuret=dataprocesst(dft)
    modeling(labeltr,featuretr,featuret)


if __name__=='__main__':
    main()