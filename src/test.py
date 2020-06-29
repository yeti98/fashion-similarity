import os
import pandas as pd
path = '/home/ddragon/Downloads/175990_396802_bundle_archive/myntradataset/styles.csv'
data = pd.read_csv(path, error_bad_lines=False)

print(data.articleType.unique())

cate = ['Shirts', 'Trousers', 'Dresses', 'Tshirts']

sizes = {}

for c in cate:
    sizes[c]  = 0
pre = '/home/ddragon/Downloads/175990_396802_bundle_archive/myntradataset/images/'
for index, row in data.iterrows():

    file_name = pre+str(row['id'])+'.jpg'
    if (row['articleType'] in cate):
        print(row['id'], row['articleType'])
    else:
        try:
            os.remove(file_name)
        except :
            pass
