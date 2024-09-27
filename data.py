import pandas as pd

def generate_dataset():
    dataset = pd.DataFrame(columns=['first_name', 'last_name', 'email', 'phone_number', 'other_info'])

    data = [
        ['John', 'Doe', 'johndoe@example.com', '555-1234', 'Customer since 2015'],
        ['Jane', 'Smith', 'janesmith@example.com', '555-5678', 'Preferred customer'],
        ['Michael', 'Johnson', 'michaeljohnson@example.com', '555-9876', 'VIP customer'],
        ['Emily', 'Brown', 'emilybrown@example.com', '555-4321', 'Customer since 2018'],
        ['David', 'Davis', 'daviddavis@example.com', '555-8765', 'Loyal customer'],
        ['Sarah', 'Miller', 'sarahmiller@example.com', '555-2468', 'Customer since 2017']
    ]

    for record in data:
        dataset = pd.concat([dataset, pd.DataFrame([record], columns=dataset.columns)], ignore_index=True)

    return dataset

# Generate dataset 1
dataset1 = generate_dataset()
dataset1.to_csv('dataset1.csv', index=False)

# Generate dataset 2 with some overlapping records
dataset2 = generate_dataset()
dataset2 = dataset2.sample(frac=0.8)  # Keep 80% of the records
dataset2 = pd.concat([dataset2, dataset1.sample(frac=0.2)], ignore_index=True)  # Add 20% of dataset1 records
dataset2 = dataset2.sample(frac=1)  # Shuffle the records
dataset2.to_csv('dataset2.csv', index=False)
