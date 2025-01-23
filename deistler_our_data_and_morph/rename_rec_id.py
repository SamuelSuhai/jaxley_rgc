import sys
import pandas as pd

# Ensure the script is executed with a file name
if len(sys.argv) != 2:
    print("Usage: python script.py <file_name.pkl>")
    sys.exit(1)

# Get the file name from the command line
file_name = sys.argv[1]

# Check if the file has the correct extension
if not file_name.endswith(".pkl"):
    print("Error: The file must have a .pkl extension.")
    sys.exit(1)

try:
    # Load the .pkl file into a DataFrame
    df = pd.read_pickle(file_name)

    # Check if the 'rec_id' column exists
    if "rec_id" in df.columns:
        # Check the type of values in the 'rec_id' column
        if df["rec_id"].apply(type).eq(str).all():
            # Convert the last character of each string to an integer
            df["rec_id"] = df["rec_id"].apply(lambda x: int(x[-1]))

        elif df["rec_id"].apply(type).eq(int).all():
            print("The 'rec_id' column already contains integers. No modification needed.")
        else:
            print("Error: The 'rec_id' column contains mixed or unsupported types.")
            sys.exit(1)

        # Save the modified DataFrame back to the same file
        df.to_pickle(file_name)
        print(f"File '{file_name}' has been successfully updated.")
    else:
        print(f"The 'rec_id' column does not exist in the DataFrame. Only columns are: {df.columns}")

except FileNotFoundError:
    print(f"Error: File '{file_name}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")
