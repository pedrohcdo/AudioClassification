
class Dataset():

    class Data():
        def __init__(self, label, data):
            self.label = label
            self.data = data

    def __init__(self, classes):
        self.datas = []
        self.classes = classes
    
    def add_data(self, label, data):
        self.datas.append(Dataset.Data(label, data))

    def __iter__(self):
        return iter(self.datas)
