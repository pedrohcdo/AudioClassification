
class Dataset():

    class Data():
        def __init__(self, label, data, fs):
            self.label = label
            self.data = data
            self.fs = fs

    def __init__(self, classes):
        self.datas = []
        self.classes = classes
    
    def add_data(self, label, data, fs):
        self.datas.append(Dataset.Data(label, data, fs))

    def __iter__(self):
        return iter(self.datas)
