from pathlib import Path
import pandas


from argparse import ArgumentParser

from alinemol.utils.split_utils import split_molecules_train_test, split_molecules_train_val_test
from alinemol.utils.utils import increment_path


def parse_args():
    parser = ArgumentParser('Splitting molecules into train and test sets')
    parser.add_argument('-f', '--file-path', type=str, required=True,
                        help='Path to a .csv/.txt file of SMILES strings')
    parser.add_argument('-sp', '--splitter', type=str, default='random',
                        help='The name of the splitter to use')
    parser.add_argument('-tr', '--train-size', type=float, default=0.9,
                        help='The size of the train set')
    args = vars(parser.parse_args())
    return args



if __name__ == '__main__':
    args = parse_args()
    file_path = Path(args['file_path'])
    splitter = args['splitter']
    train_size = args['train_size']

    if file_path.suffix == '.csv':
        df = pandas.read_csv(file_path)
    elif file_path.suffix == '.txt':
        df = pandas.read_csv(file_path, sep='\t')
    else:
        raise ValueError('File must be a .csv or .txt file')

    split_folder = (file_path.parent/"split").mkdir(parents=True, exist_ok=True)
    split_path =  (split_folder/splitter).mkdir(parents=True, exist_ok=True)

    train, external_test = split_molecules_train_test(df, splitter, train_size=train_size)

    print("percentage of actives in the train set:", train['label'].sum() / train['label'].shape[0])
    print("percentage of actives in the external test set:", external_test['label'].sum() / external_test['label'].shape[0])

    train.to_csv(increment_path(split_path/'train.csv'), index=False)
    external_test.to_csv(increment_path(split_path/'external_test.csv'), index=False)
    

