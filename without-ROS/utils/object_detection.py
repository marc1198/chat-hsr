import pathlib

class ObjectDetection:
    def __init__(self, relative_objects_file="../objects_available.txt"):
        self.folder = pathlib.Path(__file__).parent.resolve()
        self.objects_file = f"{self.folder}/{relative_objects_file}"

    def detect_objects(self, removed_objects):
        #Object detection of available objects on the table (read from txt file)
        with open(self.objects_file, 'r', encoding='utf-8') as all_objects:
            objects = all_objects.read().replace("\n", "").split(',')
            objects = [obj.strip(' " ') for obj in objects]
            objects = [obj for obj in objects if obj not in removed_objects]
        
        print("Object detection done")
        return objects