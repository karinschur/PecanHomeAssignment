from utils import utils, constants


class DataHolder:

    def __init__(self, *, path_to_data: str):
        self.train = utils.read_csv(path=f'{path_to_data}/train.csv',
                                    schema=constants.schema)
        self.val = utils.read_csv(path=f'{path_to_data}/val.csv',
                                  schema=constants.schema)
        self.test = utils.read_csv(path=f'{path_to_data}/test.csv',
                                   schema=constants.schema)


if __name__ == '__main__':
    import os
    from pathlib import Path
    path = os.path.join(Path(os.getcwd()).parent, 'data')
    dh = DataHolder(path_to_data=path)
    print()
