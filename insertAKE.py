import os
import subprocess

def find_txt_files(directory):
    txt_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files

def add_line_to_file(file_path, line):
    with open(file_path, 'r+', encoding='utf-8-sig') as file:
        content = file.read()
        file.seek(0, 0)
        file.write(line.rstrip('\r\n') + '\n' + content)

def process_files(directory):
    txt_files = find_txt_files(directory)
    for file in txt_files:
        subprocess.run(['python3', './inserter.py', '-f', file, '-c', 'ake', '-s', ';'])

        
if __name__ == '__main__':
    directory = './AKE/'#input('Enter the directory path to search for .txt files: ')
    process_files(directory)

