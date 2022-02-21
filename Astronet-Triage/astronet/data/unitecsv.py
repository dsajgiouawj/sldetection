import sys
import os
import shutil
import csv


if __name__ == '__main__':
    print('start process')
    sys.path.append('D:\\SynologyDrive\\Univ\\kenkyuu\\Astronet-Triage')
    args=sys.argv
    input_csv_name1=args[1]
    print('input_csv_name1 is '+input_csv_name1)
    input_csv_name2=args[2]
    print('input_csv_name2 is '+input_csv_name2)
    output_csv_name=args[3]
    print('output_csv_name is ' +output_csv_name)
    
    dictionary={}
    with open(input_csv_name2,mode='r') as f2:
        lines=f2.readlines()
        for line in lines:
            id,score=line.split()
            score=float(score)
            dictionary[id]=score

    output_csv=open(output_csv_name,'w',newline="")
    writer=csv.writer(output_csv)
    writer.writerow(['tic_id','score','Epoc','Period','Duration','Sectors'])

    with open(input_csv_name1,mode='r') as f1:
        reader=csv.reader(f1)
        l=[row for row in reader]
        for i in range(1,len(l)):
            id=l[i][1]
            score=dictionary[id]
            if score>=0.1:
                writer.writerow([id,score,l[i][5],l[i][6],l[i][7],l[i][9]])
    output_csv.close()