import sys
import pandas as pd
import random

def gen_seq(df):
    item = []
    for idx, row in df.iterrows():
        item.append(str(row["date"])+","+str(row["HUFL"])+","+str(row["HULL"])
                                             +","+str(row["MUFL"])+","+str(row["MULL"])
                                             +","+str(row["LUFL"])+","+str(row["LULL"]) + "," + str(row["OT"]))
    return ";".join(item)


def gen_label(df):
    return str(max(df['OT']))


def sample(df):
    line = []
    for i in range(0, len(df), 30):
        offset = random.randint(0, 25)
        if offset+i+192 >= len(df) or offset+i+240>=len(df):
            continue
        seq = df[offset+i: offset+i+192]
        label = df[offset+i+192: offset+i+240]
        feat = gen_seq(seq)
        lb = gen_label(label)
        line.append(feat + "\t" + lb)
    return line


def process(input_file, output_file):
    df = pd.read_csv(input_file)
    length = len(df)
    threhold = int(length*0.85)
    train_df = df[: threhold]
    test_df = df[threhold:]

    lines = sample(train_df)
    fd=open(output_file+".train", "w")
    fd.write("\n".join(lines))
    fd.close()

    lines = sample(test_df)
    fd=open(output_file+".test", "w")
    fd.write("\n".join(lines))
    fd.close()


if __name__ == "__main__":
    process(sys.argv[1], sys.argv[2])