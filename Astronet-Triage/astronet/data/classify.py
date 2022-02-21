import sys
import os
import shutil


if __name__ == '__main__':
    print('start process')
    sys.path.append('D:\\SynologyDrive\\Univ\\kenkyuu\\Astronet-Triage')
    args=sys.argv
    input_csv_name=args[1]
    print('input_csv_name is '+input_csv_name)
    output_directory=args[2]
    print('output directory is ' +output_directory)
    source_directory=args[3]
    print('source directory is ' +source_directory)
    prefix=args[4]
    print('prefix is ' +prefix)

    subdirectory=['-01','-02','-03','-04','-05','-06','-07','-08','-09','-10']
    with open(input_csv_name,mode='r') as f:
        lines=f.readlines()
        for line in lines:
            id,score=line.split()
            score=float(score)
            subdir='-01'
            for i in range(10):
                if score>i*0.1:
                    subdir=subdirectory[i]
            dest=os.path.join(output_directory,subdir,prefix+id+'.png')
            source=os.path.join(source_directory,id+'.png')
            shutil.copy(source,dest)