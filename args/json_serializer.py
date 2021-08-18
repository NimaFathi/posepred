from json import JSONEncoder


class JSONEncoder_(JSONEncoder):
    def default(self, o):
        return o.__dict__
