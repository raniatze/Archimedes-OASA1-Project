import os
import pandas as pd
import csv

def find_csv_files(directory):
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files
    
def clean_line_descr(line_descr):
    return ' '.join(s.strip() for s in line_descr.split('-'))

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
    csv_files = find_csv_files(directory)
    counts = {}
    for file in csv_files:
        #content = read_csv_file(file)
        df = pd.read_csv(file, sep = ';')
        
        df['clean_line_descr'] = df['Line_descr'].apply(clean_line_descr)


        grouped_data = df.groupby('clean_line_descr')

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
        line_descr = string.split(';')[0]
        new_line_descr = ' '.join(s.strip() for s in line_descr.split('-'))
        count['missing_pass_perc'] = count['missing_values'] / count['occurrences']
        count['missing_date_perc'] = count['missing_dates'] / count['occurrences']
        
        count['line'] = new_line_descr

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
    directory = './AKE/' # input('Enter the directory path to search for .csv files: ')
    process_files(directory)
