import pandas as pd
import json
import sys

def csv_to_json(csv_path, json_path, n):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Take the  nth row
        df_subset = df.iloc[[n]]
        
        # Remove output/target columns if they exist
        cols_to_remove = ['is_fraud', 'predicted_prob', 'prediction']
        df_subset = df_subset.drop(columns=[c for c in cols_to_remove if c in df_subset.columns])
        
        # Convert to list of dictionaries
        data = df_subset.to_dict(orient='records')
        
        # If n=1, we might want just the object, but usually for a "set of data" a list is better.
        # Given the request "nth row", a list is most appropriate.
        
        with open(json_path, 'w') as f:
            json.dump(data[0], f, indent=4)
        
        print(f"Successfully converted {len(data)} rows from {csv_path} to {json_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    csv_file = "found_fraud_cases.csv"
    json_file = "sample_input.json"
    
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == 'all':
            n = None
        else:
            try:
                n = int(sys.argv[1])
            except ValueError:
                print("Usage: python csv_to_json.py <number_of_rows | 'all'>")
                sys.exit(1)
    else:
        # Default to all if not specified
        n = None
        
    csv_to_json(csv_file, json_file, n)
