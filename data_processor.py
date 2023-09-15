
import csv

filename1='data/raw/train_binary.txt'
filename2='data/raw/test_binary.txt'
outfile1='data/text_generated.txt'
outfile2='data/corpus/text_generated.txt'
with open(filename1,mode='r',encoding='utf-8') as csvfile_train,open(filename2,mode='r',encoding='utf-8') as csvfile_eval,open(outfile1, mode='w',encoding='utf-8') as targetfile1,open(outfile2, mode='w',encoding='utf-8') as targetfile2:
    csv_reader1 = csv.reader(csvfile_train)  # 使用csv.reader读取csvfile中的文件
    csv_reader2 = csv.reader(csvfile_eval)

    count1 = 6735


    count2 = 750
    #header = next(csv_reader)        # 读取第一行每一列的标题
    index = 0
    for row in csv_reader1:
        if index == count1:
            break
        targetfile1.write(f"{index}\ttrain\t{row[0]}\n")
        targetfile2.write(f"{row[1]}\n")
        index += 1
    for row in csv_reader2:
        if index == count1 + count2:
            break
        targetfile1.write(f"{index}\ttest\t{row[0]}\n")
        targetfile2.write(f"{row[1]}\n")
        index += 1