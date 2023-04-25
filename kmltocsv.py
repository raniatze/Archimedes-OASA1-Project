import xml.etree.ElementTree as ET
import csv

# Replace the file path below with the path to your KML file
kml_file = './stops_230306.kml'

# Parse the KML file with ElementTree
tree = ET.parse(kml_file)
root = tree.getroot()

# Find all the Placemark elements
placemarks = root.findall('.//{http://www.opengis.net/kml/2.2}Placemark')

# Create a CSV file to write the data to
# Replace the file path below with the path where you want to save the CSV file
csv_file = 'staseis_dimoi.csv'
csv_writer = csv.writer(open(csv_file, 'w'), delimiter=';')

# Write the header row to the CSV file
csv_writer.writerow(['stop_id', 'stop_descr', 'stop_desr_matrix', 'stop_street', 'dimos', 'perioxi','stop_code', 'stop_url'])

# Loop through each Placemark and extract the data between <ExtendedData> tags
for placemark in placemarks:
    data = ''
    extended_data = placemark.find('.//{http://www.opengis.net/kml/2.2}ExtendedData')
    if extended_data is not None:
        for child in extended_data:
            if child.tag.endswith('Data'):
                try:
                    data += child.find('{http://www.opengis.net/kml/2.2}value').text + ';'
                except:
                    data += 'NULL;'
    # Write the data to the CSV file
    csv_writer.writerow([data.rstrip(';')])

with open(csv_file, 'r') as f:
    lines = f.readlines()

# Remove the first and last character of each line
modified_lines=[lines[0]]
modified_lines.extend([line[1:-2]+'\n' for line in lines[1:]])
# Write the modified lines back to the file
with open(csv_file, 'w') as f:
    f.writelines(modified_lines)

