import csv
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("ServiceAccountKey.json")
app = firebase_admin.initialize_app(cred)

store = firestore.client()

# Option 1: Double backslashes
file_path = "garments_workers.csv"

collection_name = "productivity"

def batch_data(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

data = []
headers = []

try:
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                for header in row:
                    headers.append(header)
                line_count += 1
            else:
                obj = {}
                for idx, item in enumerate(row):
                    obj[headers[idx]] = item
                data.append(obj)
                line_count += 1
        print(f'Processed {line_count} lines.')
except Exception as e:
    print(f"An error occurred while reading the CSV file: {e}")

print("Data loaded:", data)

try:
    for batched_data in batch_data(data, 499):
        batch = store.batch()
        for data_item in batched_data:
            doc_ref = store.collection(collection_name).document()
            batch.set(doc_ref, data_item)
        batch.commit()
        print("Batch committed successfully.")
except Exception as e:
    print(f"An error occurred while writing to Firestore: {e}")

print('Done')
