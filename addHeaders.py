import os

def find_txt_files(directory):
    txt_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files

"""
def add_line_to_file(file_path, line):
    content = ''
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        content = file.read()
        file.seek(0, 0)
        file.write(line.rstrip('\r\n') + '\n' + content)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.seek(0, 0)
        file.write(line.rstrip('\r\n') + '\n' + content)
"""

def add_line_to_file(file_path, line):
    with open(file_path, 'rb') as file:
        content = file.read()

    if content.startswith(b'\xef\xbb\xbf'):
        content = content[3:]

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(line.rstrip('\r\n') + '\n')
        file.write(content.decode('utf-8'))


def process_files(directory):
    txt_files = find_txt_files(directory)
    for file in txt_files:
        # Add the line at the start of the file
        add_line_to_file(file, 'Line_descr;Rtype;Direction;Sched;S_order;Vehicle_no;Arrival_datetime;Stop_id;Stop_descr;Stop_metric;T_pas_in;T_pas_out;T_pa_in_veh')

if __name__ == '__main__':
    directory = './AKE/'#input('Enter the directory path to search for .txt files: ')
    process_files(directory)

