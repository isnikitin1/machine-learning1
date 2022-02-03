import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    if np.unique(feature_vector).shape[0] == 1:
        return -np.inf, -np.inf, -np.inf, -np.inf
    else:
        feature_vector = np.array(feature_vector)
        target_vector = np.array(target_vector)
        ind = feature_vector.argsort()
        sorted_feature_ = np.sort(feature_vector)
        sorted_feature, index = np.unique(sorted_feature_[::-1], return_index = True)
        index = sorted_feature_.shape[0] - 1 - index
        index = index[:-1]
        thresholds = (np.sort(sorted_feature)[1:] + np.sort(sorted_feature)[:-1]) / 2
        target_sorted = target_vector[ind]
        R = feature_vector.shape[0]
        R_L = (index + 1)
        R_R = R - R_L
        p_1_L = np.cumsum(target_sorted)[index] / R_L
        p_1_R = (np.cumsum(target_sorted)[-1] - np.cumsum(target_sorted))[index] / R_R
        H_L = 1 - p_1_L**2 - (1 - p_1_L)**2
        H_R = 1 - p_1_R**2 - (1 - p_1_R)**2
        ginis = - R_L / R * H_L - R_R / R * H_R
        threshold_best, gini_best = thresholds[np.argmax(ginis)], np.max(ginis)
        return thresholds, ginis, threshold_best, gini_best
    
    
class DecisionTree:
    def __init__(
        self,
        feature_types,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
    ):
        if np.any(
            list(map(lambda x: x != "real" and x != "categorical", feature_types))
        ):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def get_params(self, deep=False):
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf,
        }

    def _fit_node(self, sub_X, sub_y, node, depth):
        sub_X = np.array(sub_X)
        sub_y = np.array(sub_y)
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        depth_condition = self._max_depth != None and depth > self._max_depth
        split_condition = self._min_samples_split != None and sub_y.shape[0] <= self._min_samples_split
        if depth_condition or split_condition:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(
                    map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1]))
                )
                categories_map = dict(
                    zip(sorted_categories, list(range(len(sorted_categories))))
                )

                feature_vector = np.array(
                    list(map(lambda x: categories_map[x], sub_X[:, feature]))
                )
            else:
                raise ValueError

            if len(feature_vector) == 0:
                break

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(
                        map(
                            lambda x: x[0],
                            filter(lambda x: x[1] < threshold, categories_map.items()),
                        )
                    )
                else:
                    raise ValueError

        if gini_best == -np.inf:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        node_ = node
        x = np.array(x)
        while True:
            if node_["type"] == "terminal":
                return node_["class"]
            else:
                if self._feature_types[node_["feature_split"]] == "real":
                    if x[node_["feature_split"]] <= node_["threshold"]:
                        node_ = node_["left_child"]
                    else:
                        node_ = node_["right_child"]
                else:
                    if x[node_["feature_split"]] in node_["categories_split"]:
                        node_ = node_["left_child"]
                    else:
                        node_ = node_["right_child"]

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 1)

    def predict(self, X):
        predicted = []
        for x in np.array(X):
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)