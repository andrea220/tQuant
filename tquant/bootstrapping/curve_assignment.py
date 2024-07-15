from tquant.bootstrapping.id3 import id3, classify


class CurveAssignment:
    def __init__(self, data, attributes):
        self.tree = id3(data, attributes)
        self.attributes = attributes

    def get_curve_name(self, instance):
        return classify(instance, self.tree, self.attributes)


if __name__ == "__main__":
    data = [["EUR", "DISCOUNT", "", "EUR EONIA"]]
    attributes = ["CCY", "USAGE", "INDEX_TENOR"]
    curve_assignment = CurveAssignment(data, attributes)

    instance = {"CCY": "EUR", "USAGE": "DISCOUNT"}
    print(curve_assignment.get_curve_name(instance))


