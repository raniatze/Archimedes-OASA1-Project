import os
import pandas as pd
import csv

def find_txt_files(directory):
    txt_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files

def combine_dicts(dict1, dict2):
    combined_dict = dict1.copy()

    for key, values in dict2.items():
        if key in combined_dict:
            combined_dict[key]['missing_values'] += values['missing_values']
            combined_dict[key]['occurrences'] += values['occurrences']
            combined_dict[key]['missing_dates'] += values['missing_dates']
        else:
            combined_dict[key] = values

    return combined_dict


def process_files(directory):
    txt_files = find_txt_files(directory)
    counts = {}
    for file in txt_files:
        #content = read_txt_file(file)
        df = pd.read_csv(file, sep = ';')

        grouped_data = df.groupby('Line_descr')

        stats_by_route = grouped_data.agg(
            missing_values=('T_pa_in_veh', lambda x: x.isnull().sum()),
            missing_dates=('Arrival_datetime', lambda x: x.isnull().sum()),
            occurrences=('Line_descr', 'count')
        )

        counts = combine_dicts(counts, stats_by_route.to_dict('index'))

        '''
        for line in content:
            # Get the string before the " - " token
            string = line.split(' - ')[0]
            # Increment the count for this string
            counts[string] = counts.get(string, 0) + 1
        '''

    data = []

    for string, count in counts.items():
        line = string.split(' - ')[0]
        count['missing_pass_perc'] = count['missing_values'] / count['occurrences']
        count['missing_date_perc'] = count['missing_dates'] / count['occurrences']
        
        count['line'] = line

        data.append(count)

        #print(f'{line}: {count}')

    # Open the file in write mode and write the dictionaries to the file
    csv_file = './nulls.csv'
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
        
        # Write the header (keys) to the CSV file
        writer.writeheader()
        
        # Write the data (values) to the CSV file
        for row in data:
            writer.writerow(row)

if __name__ == '__main__':
    directory = './AKE/'#input('Enter the directory path to search for .txt files: ')
    process_files(directory)
