
from Bio.Blast.Applications import NcbipsiblastCommandline
import os
import joblib
import argparse
import pandas as pd
import numpy as np


def write_fasta(fasta_name, keys, sequences):
    with open(fasta_name, "w") as f:
        for i, key in enumerate(keys):
            seq = sequences[i]
            f.write(f">{key}\n")
            f.write(f"{seq}\n")


def process_input_user(df,dir):
    for i, row in df.iterrows():
        uniprot_id = row.uniprot_id
        sequence = row.Sequence
        print(f"Processing protien {uniprot_id}")
        pssmfile=dir+uniprot_id+"_pssm.txt"
        inputfile=dir+uniprot_id+"_tem.fasta"

        if not os.path.exists(pssmfile):
            if os.path.exists(inputfile):
                print(inputfile)
                os.remove(inputfile)
            write_fasta(inputfile, [uniprot_id], [sequence])
            psiblast_cline = NcbipsiblastCommandline(query=inputfile, db='./db/swissprot/swissprot', num_iterations=3,
                                    evalue=0.001, out_ascii_pssm=pssmfile, num_threads=4)
            stdout, stderr = psiblast_cline()
            try:
                psiblast_cline = NcbipsiblastCommandline(query=inputfile, db='./db/swissprot/swissprot', num_iterations=3,
                                                        evalue=0.001, out_ascii_pssm=pssmfile, num_threads=4)
                stdout, stderr = psiblast_cline()
                print(stderr)
                os.remove(inputfile)
            except:
                print("invalid protein: " + uniprot_id)  

    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="SearchFirst config file path")
    argparser.add_argument("--csv", default="./data/uniprot_trainset.csv", help="path to data file")
    argparser.add_argument("--dir", default="./test_pssm/", help="directory to save pssm files")
    argparser.add_argument("--n_cores", type=int, default=1, help="path to data file")

    args = argparser.parse_args()
    csv = args.csv
    directory = args.dir
    n_cores = args.n_cores

    df = pd.read_csv(csv)
    split_dfs = np.array_split(df, n_cores)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    joblib.Parallel(n_jobs=n_cores)(
                joblib.delayed(process_input_user)(split_df, directory)
                for split_df in split_dfs
            )
