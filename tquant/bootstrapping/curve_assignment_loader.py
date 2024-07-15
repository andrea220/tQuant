import json

from tquant.bootstrapping.curve_assignment import CurveAssignment


class CurveAssignmentLoader:
    def __init__(self, filename: str):
        self.filename = filename

    def load_curve_assignment(self) -> CurveAssignment:
        with open(self.filename, 'r') as file:
            data = json.load(file)
            attributes = ["CCY", "USAGE", "INDEX_TENOR"]
            curve_assignment = CurveAssignment(data, attributes)
            return curve_assignment


if __name__ == "__main__":
    loader = CurveAssignmentLoader('curve_assignment_data.json')
    curve_assignment = loader.load_curve_assignment()
    instance = {"CCY": "EUR", "USAGE": "DISCOUNT"}
    curve_name = curve_assignment.get_curve_name(instance)
    print(curve_name)
    instance = {"CCY": "GBP", "USAGE": "DISCOUNT"}
    curve_name = curve_assignment.get_curve_name(instance)
    print(curve_name)
    instance = {"CCY": "USD", "USAGE": "FORECAST", "INDEX_TENOR": "1M"}
    curve_name = curve_assignment.get_curve_name(instance)
    print(curve_name)