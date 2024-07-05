import csv

csv.field_size_limit(1000000)  # or any larger number you need
with open('VCR-wiki-zh-easy-test-500.tsv', 'r', newline='') as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        print(row)
