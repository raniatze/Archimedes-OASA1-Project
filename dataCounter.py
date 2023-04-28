import os

def find_txt_files(directory):
    txt_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files

def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        content = file.readlines()
    return content

def process_files(directory):
    txt_files = find_txt_files(directory)
    counts = {}
    for file in txt_files:
        content = read_txt_file(file)
        for line in content:
            # Get the string before the " - " token
            string = line.split(' - ')[0]
            # Increment the count for this string
            counts[string] = counts.get(string, 0) + 1

    # Print the counts for each string
    for string, count in counts.items():
        print(f'{string}: {count}')

if __name__ == '__main__':
    directory = './AKE/'#input('Enter the directory path to search for .txt files: ')
    process_files(directory)

