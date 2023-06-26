import os
import subprocess

def find_csv_files(directory):
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('ake_updated.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def process_files(directory):
    csv_files = find_csv_files(directory)
    cnt = 0
    for file in csv_files:
        print(file)
        subprocess.run(['python3', './inserter.py', '-f', file, '-c', 'ake', '-s', ','])
        cnt += 1
    print(cnt)
        
if __name__ == '__main__':
    directory = './AKE/' # input('Enter the directory path to search for .txt files: ')
    process_files(directory)

