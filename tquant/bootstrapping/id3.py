import math


def entropy(data):
    label_counts = {}
    for entry in data:
        label = entry[-1]
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    entropy = 0.0
    for label in label_counts:
        prob = label_counts[label] / len(data)
        entropy -= prob * math.log2(prob)
    return entropy


def information_gain(data, attribute_index):
    attribute_values = {}
    for entry in data:
        attr_val = entry[attribute_index]
        if attr_val in attribute_values:
            attribute_values[attr_val].append(entry)
        else:
            attribute_values[attr_val] = [entry]
    gain = entropy(data)
    for attr_val in attribute_values:
        subset = attribute_values[attr_val]
        prob = len(subset) / len(data)
        gain -= prob * entropy(subset)
    return gain


def id3(data, attributes):
    labels = [entry[-1] for entry in data]
    if len(set(labels)) == 1:
        return labels[0]
    if len(attributes) == 0:
        return max(set(labels), key=labels.count)
    gains = [information_gain(data, i) for i in range(len(attributes))]
    best_attr_index = gains.index(max(gains))
    best_attr = attributes[best_attr_index]

    tree = {best_attr: {}}
    # del attributes[best_attr_index]

    attribute_values = set([entry[best_attr_index] for entry in data])
    for value in attribute_values:
        subset = [entry[:] for entry in data if entry[best_attr_index] == value]
        subtree = id3(subset, attributes[:])
        tree[best_attr][value] = subtree
    return tree


def classify(instance, tree, attributes):
    if isinstance(tree, str):
        return tree
    else:
        attr = list(tree.keys())[0]
        attr_value = ""
        if any(attr in key for key in instance.keys()):
            attr_value = instance[attr]
        subtree = tree[attr][attr_value]
        return classify(instance, subtree, attributes)


if __name__ == "__main__":
    dataset = [
        ["EUR", "DISCOUNT", "", "EUR EONIA"],
        ["EUR", "FORECAST", "6M", "EUR EURIBOR 6M"],
        ["EUR", "FORECAST", "3M", "EUR EURIBOR 3M"],
        ["USD", "DISCOUNT", "", "USD:STD"],
        ["USD", "FORECAST", "1M", "USD LIBOR 1M"],
        ["GBP", "DISCOUNT", "", "GBP:STD"],
        ["GBP", "FORECAST", "3M", "GBP LIBOR 3M"],
        ["JPY", "DISCOUNT", "", "JPY:STD"]
    ]

    attributes = ["CCY", "USAGE", "INDEX_TENOR"]
    tree = id3(dataset, attributes)

    new_instance = {"USAGE": "DISCOUNT", "CCY": "JPY"}
    curve_name = classify(new_instance, tree, attributes)
    print(curve_name)

    new_instance = {"CCY": "EUR", "USAGE": "DISCOUNT"}
    new_instance2 = {"INDEX_TENOR": "3M", "CCY": "EUR", "USAGE": "FORECAST"}
    new_instance3 = {"USAGE": "FORECAST", "INDEX_TENOR": "6M", "CCY": "EUR"}
    new_instance4 = {"CCY": "USD", "USAGE": "DISCOUNT"}
    new_instance5 = {"CCY": "USD", "USAGE": "FORECAST", "INDEX_TENOR": "1M"}
    new_instance6 = {"CCY": "GBP", "USAGE": "DISCOUNT"}
    new_instance7 = {"CCY": "GBP", "USAGE": "FORECAST", "INDEX_TENOR": "3M"}
    new_instance8 = {"USAGE": "DISCOUNT", "INDEX_TENOR": "3M", "CCY": "JPY"}
    #
    curve_name = classify(new_instance, tree, attributes)
    print(curve_name)
    curve_name = classify(new_instance2, tree, attributes)
    print(curve_name)
    curve_name = classify(new_instance3, tree, attributes)
    print(curve_name)
    curve_name = classify(new_instance4, tree, attributes)
    print(curve_name)
    curve_name = classify(new_instance5, tree, attributes)
    print(curve_name)
    curve_name = classify(new_instance6, tree, attributes)
    print(curve_name)
    curve_name = classify(new_instance7, tree, attributes)
    print(curve_name)
    curve_name = classify(new_instance8, tree, attributes)
    print(curve_name)
