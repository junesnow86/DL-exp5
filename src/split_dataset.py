import pandas as pd
from sklearn.model_selection import train_test_split

# news-commentary-v15.en-zh
# Load the TSV file into a DataFrame
df = pd.read_csv('/home/ljt/DL-exp5/data/news-commentary-v15/news-commentary-v15.en-zh.tsv', sep='\t', header=None)

# Sample half of the data
df = df.sample(frac=0.5)

# Split the DataFrame into training and temp DataFrames
train_df, temp_df = train_test_split(df, test_size=0.1)

# Split the temp DataFrame into validation and testing DataFrames
val_df, test_df = train_test_split(temp_df, test_size=0.5)

# Save the DataFrames to TSV files
df.to_csv('/home/ljt/DL-exp5/data/news-commentary-v15/news-commentary-v15_sample.en-zh.tsv', sep='\t', header=False, index=False)
train_df.to_csv('/home/ljt/DL-exp5/data/news-commentary-v15/train.tsv', sep='\t', header=False, index=False)
val_df.to_csv('/home/ljt/DL-exp5/data/news-commentary-v15/val.tsv', sep='\t', header=False, index=False)
test_df.to_csv('/home/ljt/DL-exp5/data/news-commentary-v15/test.tsv', sep='\t', header=False, index=False)


# back-translation
def count_lines(filename):
    with open(filename, 'r') as f:
        return sum(1 for _ in f)

def read_samples(en_filename, zh_filename, percentage=0.005):
    num_lines = count_lines(en_filename)
    num_samples = int(num_lines * percentage)
    samples = []

    with open(en_filename, 'r') as en_file, open(zh_filename, 'r') as zh_file:
        for _, (en_line, zh_line) in zip(range(num_samples), zip(en_file, zh_file)):
            samples.append({'English': en_line.strip(), 'Chinese': zh_line.strip()})

    return pd.DataFrame(samples)

def split_samples(df, test_size=0.1):
    train_df, val_test_df = train_test_split(df, test_size=test_size)
    val_df, test_df = train_test_split(val_test_df, test_size=0.5)
    return train_df, val_df, test_df

en_filename = '/home/ljt/DL-exp5/data/back-translation/news.en'
zh_filename = '/home/ljt/DL-exp5/data/back-translation/news.translatedto.zh'
df = read_samples(en_filename, zh_filename)
train_df, val_df, test_df = split_samples(df)

df.to_csv('/home/ljt/DL-exp5/data/back-translation/news.en-zh.tsv', sep='\t', header=False, index=False)
train_df.to_csv('/home/ljt/DL-exp5/data/back-translation/train.tsv', sep='\t', header=False, index=False)
val_df.to_csv('/home/ljt/DL-exp5/data/back-translation/val.tsv', sep='\t', header=False, index=False)
test_df.to_csv('/home/ljt/DL-exp5/data/back-translation/test.tsv', sep='\t', header=False, index=False)
