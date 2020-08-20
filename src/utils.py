import pickle

def save_data(data, path):
    """load file using pickle"""
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def read_data( path):
    """read file using pickle"""
    with open(path, "rb") as fr:
        return pickle.load(fr)
