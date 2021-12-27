import pandas as pd
import sys

# read argument
threshold = float(sys.argv[1])
input_data_path = sys.argv[2]
output_data_path = sys.argv[3]

data = pd.read_csv(input_data_path)
def filter_unknown(label):
    label = str(label).replace("###", '')
    return label

data["Label_data"] = data["Label_data"].apply(filter_unknown)

def check_only_symbol(label):
    symbolic = '#{}()[].,:;+-*/\\&|<>=~、%$\'"'
    for i in str(label):
        if i not in symbolic:
            return False

    return True
data.drop(data[data["Label_data"].apply(check_only_symbol)].index, inplace=True)

data["Label_data"][data["Label_data"].isna()] = "###"
data["Label_data"][data["Label_data"]==""] = "###"

data.loc[:, ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3']] = data.loc[:, ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3']].astype(int)
data = data[(data["rec_confidence"] > threshold)]

def changeName(name):
    import re
    pattern = "(img_\d+)-\d+.jpg"
    pattern = re.compile(pattern)

    return pattern.search(name)[1]
data["imgName"] = data["imgName"].apply(changeName)

data["Label_data"][data["Label_data"].isna()] = "###"
data["Label_data"][data["Label_data"]==""] = "###"
data.to_csv(output_data_path, columns=['imgName', 'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'Label_data'], index=False, header=False)