import glob, os
import pandas as pd
import argparse
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def parse_csv(root: str, rel_label_file: str, rel_test_list: str, frac: float):
    labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
              'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
              'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
              'Pneumothorax', 'none']

    _data = pd.read_csv(
        os.path.join(root, rel_label_file),
        usecols=['Image Index', 'Finding Labels', 'Patient ID']
    )

    if(frac < 1):
        _data = _data.sample(frac=frac)

    _data.rename(columns = {
        'Image Index': 'idx',
        'Finding Labels': 'findings',
        'Patient ID': 'patient'
    }, inplace = True)

    # replace 'No Finding' with none
    _data['findings'] = _data['findings'].map(lambda x: x.replace('No Finding',
                                                                  'none'))

    # | split labels to list
    _data['findings'] = _data['findings'].map(lambda x: x.split('|')).tolist()

    for label in labels:
        _data[label] = _data['findings'].map(lambda finding: 1.0 if label in finding else 0.0)

    _test_files = pd.read_csv(
        os.path.join(root, rel_test_list),
        header=None,
        squeeze=True
    )
    # split test/train data
    test_filter = pd.Index(_data['idx']).isin(_test_files)
    _data_test  = _data.loc[test_filter].reset_index(drop=True)
    _data_train = _data.loc[[not x for x in test_filter]].reset_index(drop=True)

    return (_data_test, _data_train)


def preload(df: pd.DataFrame, img_dir: str):
    _labels = []
    _data   = []

    for idx, row in df.iterrows():
        print('Progress: {}'.format(idx))
        img_path = os.path.join(img_dir, row[0])
        img_path = glob.glob(img_path)
        img = Image.open(img_path[0]).convert('RGB').resize((224, 224))
        img_tr = transform(img)

        _data.append(np.array(img_tr))
        _labels.append(row[2:18].values)

    return np.array(_labels), np.array(_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--file-labels', type=str, default='Data_Entry_2017.csv')
    parser.add_argument('--file-test',   type=str, default='test_list.txt')
    parser.add_argument('--frac',        type=float, default=1)
    parser.add_argument('--store-dir',   type=str, default='./')
    args = parser.parse_args()

    test_file  = os.path.join(args.store_dir, 'test_224.npy')
    train_file = os.path.join(args.store_dir, 'train_224.npy')

    test, train = parse_csv(root           = args.data_dir,
                            rel_label_file = args.file_labels,
                            rel_test_list  = args.file_test,
                            frac           = args.frac)

    np_test_labels, np_test_data = preload(test,
                                           os.path.join(args.data_dir,
                                                        'images_*/images'))
    with open(test_file, 'wb') as f:
        np.save(f, np_test_data)
        np.save(f, np_test_labels)
    del np_test_data
    del np_test_labels

    np_train_labels, np_train_data = preload(train,
                                             os.path.join(args.data_dir,
                                                          'images_*/images'))
    with open(train_file, 'wb') as f:
        np.save(f, np_train_data)
        np.save(f, np_train_labels)

    del np_train_data
    del np_train_labels



if __name__ == "__main__":
    main()
