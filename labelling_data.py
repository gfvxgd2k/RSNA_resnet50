import csv
import shutil
import mritopng #convert dcm to png

mritopng.convert_folder('dcm/','png/')

f = open('stage_2_train_labels.csv','r',encoding='utf-8')
rdr = csv.reader(f)
count = 1
src = 'png/'
for line in rdr:
    print(count, line[0], line[5])
    filename = str(line[0]) + '.dcm.png'
    filename_1 = str(line[0]) + '.png'
    print(filename)
    if line[5] == '0':
        dir = 'train/0/'
        try:
            shutil.move(src + filename, dir + filename_1)
        except:
            continue
    elif line[5] == '1':
        dir = 'train/1/'
        try:
            shutil.move(src + filename, dir + filename_1)
        except:
            continue
    count+=1
f.close()