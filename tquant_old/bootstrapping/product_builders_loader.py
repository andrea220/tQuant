import datetime
import json

from tquant import DepositBuilder
from tquant.bootstrapping.basis_swap_builder import BasisSwapBuilder
from tquant.bootstrapping.fra_builder import FraBuilder
from tquant.bootstrapping.future_builder import FutureBuilder
from tquant.bootstrapping.ois_builder import OisBuilder
from tquant.bootstrapping.swap_builder import SwapBuilder


class ProductBuildersLoader:
    def __init__(self, filename: str):
        self.filename = filename

    def load_product_builders(self):
        with open(self.filename, 'r') as file:
            data = json.load(file)

        products_dict = {}

        for item in data:
            if item["type"] == "deposit":
                obj = DepositBuilder.from_json(item)
            elif item["type"] == "fra":
                obj = FraBuilder.from_json(item)
            elif item["type"] == "ois":
                obj = OisBuilder.from_json(item)
            elif item["type"] == "swap":
                obj = SwapBuilder.from_json(item)
            elif item["type"] == "basis_swap":
                obj = BasisSwapBuilder.from_json(item)
            elif item["type"] == "future":
                obj = FutureBuilder.from_json(item)

            products_dict[item["name"]] = obj

        return products_dict


if __name__ == "__main__":
    builders_loader = ProductBuildersLoader("product_builders_data_test.json")
    obj = builders_loader.load_product_builders()
    print(obj["depo"])
    print(obj["fra-6M"])
    print(obj["ois-6M-1Y"])
    print(obj["swap-6M-1Y"])

    now = datetime.datetime.now()
    product = obj["depo"].build(now, 0.01, "6M")
    print(product)
    product = obj["fra-6M"].build(now, 0.01, "3M-6M")
    print(product)
    product = obj["ois-6M-1Y"].build(now, 0.01, "2Y")
    print(product)
    product = obj["swap-6M-1Y"].build(now, 0.01, "2Y")
    print(product)
    product = obj["basis_swap-6M-1Y"].build(now, 0.01, "2Y")
    print(product)
    product = obj["fut-JUN-24"].build(now, 0.01, "JUN 24")
    print(product)
