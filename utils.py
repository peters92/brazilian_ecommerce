import pickle 

def save_data(data, path):
        with open(path, 'wb') as file:
            pickle.dump(data, file)
            file.close()

def load_data(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def trim_id(string_input, trim_length=5):
    # Trims input to given length
    # Used for making customer and other id's easier to read
    return string_input[:trim_length]