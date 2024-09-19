import sys
import os
import pandas as pd



if __name__ == "__main__":
    data_path = sys.argv[1]
    k_value = sys.argv[2]
    output_path = sys.argv[3]

    # Print the input arguments with clear labels
    print(f"Data path: {data_path}")
    print(f"K value: {k_value}")
    print(f"Output path: {output_path}")
    

    # ... rest of your code ...
    df = pd.read_csv(data_path)
    print(df.head())
    
    df.to_csv(output_path)
    

    # Print a completion message or any other relevant information
    print("Processing completed successfully!")

 
    
    
    
    