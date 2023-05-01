import os

def find_txt_files(directory):
    txt_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files

def process_file(file_path, line):
    with open(file_path, 'rb') as file:
        content = file.read()

    if content.startswith(b'\xef\xbb\xbf'):
        content = content[3:]

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(line.rstrip('\r\n') + '\n')
        file.write(content.decode('utf-8'))
        
    # Split the file path into directory and filename components
    dir_path, filename = os.path.split(file_path)

    # Replace the file extension with ".csv"
    new_filename = os.path.splitext(filename)[0] + '.csv'

    # Construct the new file path by joining the directory and new filename components
    new_file_path = os.path.join(dir_path, new_filename)

    # Rename the file
    os.rename(file_path, new_file_path)

def process_files(directory):
    txt_files = find_txt_files(directory)
    for file in txt_files:
        # Add the line at the start of the file and rename it
        process_file(file, 'Line_descr;Rtype;Direction;Sched;S_order;Vehicle_no;Arrival_datetime;Stop_id;Stop_descr;Stop_metric;T_pas_in;T_pas_out;T_pa_in_veh')

if __name__ == '__main__':
    directory = '/home/raniatze/AKE/' #input('Enter the directory path to search for .txt files: ')
    process_files(directory)

