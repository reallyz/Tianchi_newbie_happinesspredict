import pandas as pd


#indies
di_edu={1: '没有受过任何教育',
 2: '私塾、扫盲班',
 3: '小学',
 4: '初中',
 5: '职业高中',
 6: '普通高中',
 7: '中专',
 8: '技校',
 9: '大学专科（成人高等教育）',
 10: '大学专科（正规高等教育）',
 11: '大学本科（成人高等教育)',
 12: '大学本科（正规高等教育）',
 13: '研究生及以上',
 14: '其他'}

ls_edu=['初中','小学','没有受过任何教育','普通高中','本科全日制', '专科全日制','中专','专科成教',
 '本科成教', '职业高中','研究生及以上','私塾、扫盲班','技校','无法回答','其他']
#TODO
#空值，异常值处理， 数据可视化，不建模型
df=pd.read_csv('../dataset/happiness_train_abbr.csv')


