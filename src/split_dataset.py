import pandas as pd
from sklearn.model_selection import train_test_split

# Load the TSV file into a DataFrame
df = pd.read_csv('/storage/1008ljt/DL-exp5/data/news-commentary-v15.en-zh.tsv', sep='\t', header=None)

# Split the DataFrame into training and temp DataFrames
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)

# Split the temp DataFrame into validation and testing DataFrames
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save the DataFrames to TSV files
train_df.to_csv('/storage/1008ljt/DL-exp5/data/news-commentary-v15/train.tsv', sep='\t', header=False, index=False)
val_df.to_csv('/storage/1008ljt/DL-exp5/data/news-commentary-v15/val.tsv', sep='\t', header=False, index=False)
test_df.to_csv('/storage/1008ljt/DL-exp5/data/news-commentary-v15/test.tsv', sep='\t', header=False, index=False)
