import pandas as pd
import random
import ast

def copy_csv(input_csv, output_csv):
    data = pd.read_csv(input_csv)
    data.to_csv(output_csv, index=False)
    return data

# Function to safely evaluate string literals
def safe_literal_eval(node):
    try:
        return ast.literal_eval(node)
    except ValueError:
        return []  # Return an empty list if there is an error

# Paths to the CSV files
train_label_path = 'data/wayfast/train_labels.csv'
sam_train_label_path = 'data/wayfast/sam_train_labels.csv'
output_csv_path = 'data/wayfast/output.csv'

# Copy train_labels.csv to output.csv and load data
train_labels = copy_csv(train_label_path, output_csv_path)

# Load sam_train_labels.csv
sam_train_labels = pd.read_csv(sam_train_label_path)

# Convert labels from string to list
sam_train_labels['labels'] = sam_train_labels['labels'].apply(lambda x: x.split(';'))
#=================================================================================
selected_num = 3 # How many data you want to extract

for index, row_data in sam_train_labels.iterrows():
    sam_data_list = [ast.literal_eval(item) for item in row_data['labels']]  # Convert string to list
    # print(f"Label {index}: {sam_data_list}")
    selected_data = random.sample(sam_data_list, selected_num)
    # print(f"Label {index}: {selected_data}")
    # print(selected_data)
    formatted_strings = [';'.join([str(sublist).replace(" ", "") for sublist in selected_data])]
    # print(formatted_strings, type(formatted_strings))
    if len(train_labels) > 0:
        existing_labels = train_labels.at[index, 'labels']
        # print(existing_labels, type(existing_labels))
        # update column of labels
        for string in formatted_strings:
            train_labels.at[index, 'labels'] = existing_labels + ";" + string 
            train_labels.at[index, 'labels'].replace(" ", "")
        
    train_labels.to_csv(output_csv_path, index=False)

print("Updated labels added to output.csv.")