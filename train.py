import argparse
import os
import ast
import re
import math
import pickle

from typing import Callable, Any, Sequence, Union
from os.path import join
from random import sample
from copy import deepcopy


class LogisticRegression(object):
    """Logistic regression for binary classification"""


    def __init__(
        self, lr: float = 1e-3,
        max_iterations: int = 100,
        stop_criterion: float = 1e-4,
        use_bias: bool = True, 
        l1_coef: float = 0.,
        l2_coef: float = 0., 
        optimization: Union[str, None] = None,
        minibatch_size: int = 100,
        weight_init: Union[Callable[[int], list[float]], None] = None) -> None:


        self.is_trained = False
        self.max_iterations = max_iterations
        self.stop_criterion = stop_criterion
        self.use_bias = use_bias
        self.lr = lr
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.optimization = optimization
        self.minibatch_size = minibatch_size
        self.weight_init = weight_init

    def _init_param(self, features_count: int) -> None:

        if self.use_bias:
            self.features_count = features_count + 1
        else:
            self.features_count = features_count
        
        if self.weight_init != None:
            self._w = self.weight_init(self.features_count)
        else:
            self._w = [0 for _ in range(self.features_count)]
        

    def _preprocess_data(self, X, y) -> tuple[list[list[float]], list[float]]:
        """Called after receiving data"""
        X = deepcopy(X)
        y = deepcopy(y)

        if self.use_bias:
            for x in X:
                x.append(1)
        return X, y

    def fit(self, X:list[list[float]], y:list[float], metrics: list[Callable[[list[float], list[float]], float]] = None) -> None:
        """Fit logistic regression on X and y
            Arguments:
                X::list[list[float]]
                    The matrix of float features
                y::list[float]
                    The vector of float targets
                metrisc::list[Callable[[list[float], list[float]], float]]
                    The list with metrics functions
        """
        assert len(X) == len(y), "Count element X and y must be the same size"


        self.metrics = metrics
        self._init_param(len(X[0]))
        X, y = self._preprocess_data(X, y)
        self._fit(X, y, metrics)
        self.is_trained = True
    
    def _fit(self, X, y, metrics: list[Callable[[list[float], list[float]], float]] = None) -> None:
        """Private function for fit model"""

        def cross_entropy(y_true, y_pred, eps=1e-4):
            """Cross entropy loss function"""
            if 1 - y_pred == 0:
                y_pred -= eps
            elif y_pred == 0:
                y_pred += eps

            if y_true == 0:
                y_true += eps

            return -((y_true * math.log(y_pred)) + (1 - y_true) * math.log(1 - y_pred))

        def gradient_descent_step(X, y):

            all_loss = 0
            gradient = [0 for _ in range(len(self._w))]

            for x, y_true in zip(X, y):
                y_pred = self._sigmoid(sum([x[i] * self._w[i] for i in range(len(self._w))]))
                loss = cross_entropy(y_true, y_pred)
                all_loss += loss
                
                # sum gradient
                gradient = [g_o + g_n for g_o, g_n in zip(gradient, self._compute_gradient(x, y_pred, y_true))]
                
            self._w = [w_i - self.lr * g_i for w_i, g_i in zip(self._w, gradient)]
            return all_loss / len(X)

        for iteration in range(self.max_iterations):
            avg_loss = gradient_descent_step(X, y)
            print(f"Iteration: {iteration+1}/{self.max_iterations}, Loss: {avg_loss}", end="; ")
            if metrics:
                for metric in metrics:
                    print(f"{metric.__name__} - {metric(y, [self._sigmoid(sum([x[i] * self._w[i] for i in range(len(self._w))])) for x in X])}", end=' ')

                print()
            else:
                print()

    def regularization(self):
        pass

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function"""
        return 1. / (1. + math.e**-x)

    def _compute_gradient(self, x, y_pred, y_true) -> list[float]:
        """Computer gradient"""
        g = [(y_pred- y_true) * x_i for x_i in x]
        return g

    def fit_transform(self, X:list[list[float]], y:list[float], metrics: list[Callable[[list[float], list[float]], float]] = None) -> list[list[int]]:
        self.fit(X, y, metrics)
        return self.predict(X)

    def predict_proba(self, X:list[list[float]]) -> list[float]:
        """Make predict and return probability"""
        assert self.is_trained, "Before prediction model must be trained"
        
        X, _ = self._preprocess_data(X, [])

        y = []
        for x in X:
            y_pred = self._sigmoid(sum([x[i] * self._w[i] for i in range(len(self._w))]))
            y.append(y_pred)
        return y

    def predict(self, X: list[list[float]], boundary: float = 0.5) -> list[int]:
        """Make predict and return labels"""
        assert self.is_trained, "Before prediction model must be trained"
        
        X, _ = self._preprocess_data(X, [])

        y = []
        for x in X:
            y_pred = self._sigmoid(sum([x[i] * self._w[i] for i in range(len(self._w))]))
            if y_pred >= boundary:
                y.append(1)
            else:
                y.append(0)
        return y        

class Transformer(object):
    """Unit of pipeline where is transforming data"""

    def __init__(self, *transforms: Sequence[Callable[[Any], Any]]):
        """Object initialization by specifying the order of transformation functions
        
        Arguments:
            *transforms::Sequence[Callable[[Any], Any]]
                A sequence of functions that will be executed in turn, where each subsequent function takes the output of the previous one as an argument.
                Be attentive all functions must to get a list as input
        Returns:
            None
        """
        assert len(transforms) != 0, "The number of passed objects cannot be 0!"
        
        self._transforms = transforms

    def __call__(self, data: list[Any]) -> Any:
        """Call transformer
        
        Arguments:
            data::list[list]
                As input we get list, where every list is example of data
        """
        
        data = data.copy()
        
        for transform in self._transforms:
            for i, example in enumerate(data):
                data[i] = transform(data[i])
        return data


def get_arguments() -> argparse.Namespace:
    """This function get arguments from user.

    Returns:
        parser::argparse.Namespace
            parser with arguments from user.
    """
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('files', type=str, help='Path to files')
    parser.add_argument('plagiat1', type=str, help='Path to plagiat1')
    parser.add_argument('plagiat2', type=str, help='Path to plagiat2')
    parser.add_argument(
        '--model',
        type=str,
        default='model.pkl',
        help='Model name (default: model.pkl)'
    )

    return parser.parse_args()


def get_data(*folders: Sequence[str]) -> list[dict[str, str]]:
    """Return data from folders as dictionaries with keys as name and string as value.
    
    Arguments:
        *folders::str
            Paths to folder of dataset.
    
    Returns:
        corpus::list[dict[str, str]]
            List of dictionary data from folders where key is filename and value is string from file.
    """

    corpus = []

    for folder in folders:
        d = {}
        corpus.append(d)
        for file in os.listdir(folder):
            text = open(join(folder, file), mode='r', encoding='utf8').read()
            d[file] = text
    return corpus


def create_pairs(data_source: dict[str, str], *data: Sequence[dict[str, str]], n_combinations: int) -> tuple[list[list[str]], list[float]]:
    """Creates pairs of combinations with source and other data
    
    Arguments:
        data_source::dict[str, str]
            source data which will be in each pair n_combinations times
        *data::dict[str, str]
            data which will in pair with data source
        n_combinations::int
            count of combinations with one example from data_source and other examples from data
    
    Returns:
        
    """
    corpus = []
    y = []

    for filename, source_text in data_source.items():
        for other_data in data:
            # create pair with similar (transformed) source file from current source
            corpus.append([source_text, data_source[filename]])
            y.append(1)

            # create list of files and remove appended element
            files = list(other_data.keys())
            files.remove(filename)
            
            # sample n_combinations from 
            for other_filename in sample(files, k=n_combinations):
                corpus.append([source_text, other_data[other_filename]])
                y.append(0)
    return corpus, y

# functions for features extraction
def text_to_lower(x: tuple[str, str]) -> tuple[str, str]:
        """Make text in lower register"""
        return x[0].lower(), x[1].lower()

    
def delete_comments(x: tuple[str, str]) -> tuple[str, str]:
    """Delete all coments from code"""
    # delete all one-liners and multilines comments
    return [re.sub(r'#.*', ' ', re.sub(r'\"\"\"[^\"\"\"]*\"\"\"', '', x[0])), re.sub(r'#.*', ' ', re.sub(r'\"\"\"[^\"\"\"]*\"\"\"', '', x[1]))]


def delete_multispace(x: tuple[str, str]) -> tuple[str, str]:
    """Delete multispace"""
    return [re.sub(r'\s\s+', ' ', x[0]), re.sub(r'\s\s+', ' ', x[1])]


def delete_enters(x: tuple[str, str]) -> tuple[str, str]:
    """Delete enters"""
    return [re.sub(r'\n', ' ', x[0]), re.sub(r'\n', ' ', x[1])]


def len_string_ratio(x: tuple[str, str]) -> list[float]:
    """Ratio of number symbols first string to second string"""
    try:
        l = [len(x[0]) / len(x[1])] 
        return l if l[0] <= 3 else [3]
    except ZeroDivisionError:
        return [0]


class MinMaxScaler(object):
    """Min-max scaler for len_string_ratio"""
    def __init__(self):
        self.min = 0
        self.max = 0

    def fit(self, x: list[float]) -> list[float] :
        """Fit min-max scaler"""
        
        if x[0] > self.max:
            self.max = x[0]
        
        if x[0] < self.min:
            self.min = x[0]
    
        return x

    def scale(self, x: list[float]) -> list[float]:
        """Scaling x"""
        return [(x[0]- self.min) / (self.max - self.min)]


def tokenizer(x: tuple[str, str]) -> tuple[list[str], list[str]]:
    """Tokenizing text to tokens"""
    return [x[0].split(), x[1].split()]


def language_constructs_analysis_ast(x: tuple[str, str]) -> tuple[dict[str, int], dict[str, int]]:
    """Functions for analysis language constructs of python code
       Be careful you may use this function when you trust files in syntax errors, else use language_constructs_analysis_re which analysis by regulars expression
    """
    class Analyzer(ast.NodeVisitor):
        """Anylyzer of ast node"""
        def __init__(self) -> None:
            """Ast tree stack init
            If you want to append new feature just visit site and append new method: 
            https://docs.python.org/3/library/ast.html"""
            
            self.stats = {'import': [], 'from': [], 'class': []}
        def visit_Import(self, node) -> None:
            for alias in node.names:
                self.stats['import'].append(alias.name)
            self.generic_visit(node)
        def visit_ImportFrom(self, node) -> None:
            for alias in node.names:
                self.stats['from'].append(alias.name)
            self.generic_visit(node)
        def visit_ClassDef(self, node) -> None:
            self.stats['class'].append(node.name)
    tree_1 = ast.parse(x[0])
    tree_2 = ast.parse(x[1])
    
    analyzer1 = Analyzer()
    analyzer2 = Analyzer()
    analyzer1.visit(tree_1)
    analyzer2.visit(tree_2)
    print(analyzer1.stats)
    print(analyzer2.stats)
    return analyzer1.stats, analyzer2.stats


def language_constructs_analysis_re(x: tuple[str, str]) -> tuple[dict[str, int], dict[str, int]]:
    """Almost language_constructs_analysis but use regulars"""
    class Analyzer(object):
        """Function collect stat about code"""
        def __init__(self, code) -> None:
            self.stats = {}
            self.code = code
        def visit(self) -> None:
            self._visit_import()
            self._visit_class()
            #self._visit_def()

        def _visit_import(self) -> None:
            """Take all library from code"""
            pattern = r'import [^(\n]*'
            self.stats['import'] = [imp.split(' ')[1] for imp in re.findall(pattern, self.code)]

        def _visit_class(self):
            pattern = r'class [^(:]*'
            self.stats['class'] = [cls.split(' ')[1] for cls in re.findall(pattern, self.code)]

        def _visit_def(self) -> None:
            """Take all methods and functions"""
            pattern = r'def [^(]*'
            self.stats['def'] = [fun.split(' ')[1] for fun in re.findall(pattern, self.code)]

    a1 = Analyzer(x[0])
    a2 = Analyzer(x[1])
    a1.visit()
    a2.visit()
    return a1.stats, a2.stats


def check_similarity_constructions(x: tuple[dict[str, int], dict[str, int]]) -> tuple:
    """Function check similirity of python constructions"""
    similary = []
    
    for key, items in x[0].items():
        current_similary = 0
        for item in items:
            if item in x[1][key]:
                current_similary += 1                
        try:
            similary.append(current_similary / len(x[0][key]))
        except ZeroDivisionError:
            similary.append(0)
    return similary


def delete_specsymbols(x: tuple[str, str]) -> tuple[str, str]:
    return [re.sub(r'[^a-zA-Z ]', ' ', x[0]), re.sub(r'[^a-zA-Z ]', ' ', x[1])]


class TokenAnalyzer(object):
    """Analyzer of token"""

    def __init__(self, tokens_len=5000):
        """Params init"""
        self.tokens = {}
        self.dictionary_len = tokens_len
        
        # dictionary 
        self.is_dictionary_create = False
        self.dictionary = {}
        self.tokens_documents_frequency = []

    def collect_tokens(self, x:list[str]) -> list[str]:
        """Function for collecting all tokens"""
        for tokens in x:
            for token in tokens:
                if token in self.tokens.keys():
                    self.tokens[token] += 1
                else:
                    self.tokens[token] = 1
        return x

    def _dictionary_create(self) -> None:
        """Create dictionary"""
        i = 0
        for token, _ in sorted(self.tokens.items(), reverse=True, key=lambda item: item[1])[:self.dictionary_len]:
            # token list
            self.dictionary[token] = i
            i += 1
        self.tokens_documents_frequency = [0. for _ in range(len(self.dictionary))]
        self.is_dictionary_create = True

    def to_bag_words(self, x:list[str]) -> list[list[float]]:
        """Make x from tokens list to bag of words"""
        
        if not self.is_dictionary_create:
            self._dictionary_create()
        
        bags_words = []
        for tokens in x:
            bag_words = [0. for _ in range(len(self.dictionary))]
            t = []
            for token in tokens:
                try:
                    pos = self.dictionary[token]
                    bag_words[pos] += 1.
                    # for tf-idf
                    if pos not in t:
                        self.tokens_documents_frequency[pos] += 1.
                    t.append(pos)
                except KeyError:
                    pass
            bags_words.append(bag_words)
        return bags_words

    def to_tf_idf(self, x:list[list[float]]) -> list[list[float]]:
        """From bag to words to tf-idf"""
        tf_idfs = []
        for bag in x:
            tokens_count = sum(bag)
            tf = [(c + 1) / (tokens_count + 1) for c in bag]
            documents_count = len(self.dictionary) * 2
            idf = [math.log((documents_count + 1) / (self.tokens_documents_frequency[i] + 1)) for i in range(len(bag))]
            tf_idf = [tf_i * idf_i for tf_i, idf_i in zip(tf, idf)]
            tf_idfs.append(tf_idf)
        return tf_idfs
        
    def cosine_similarity(self, x:list[list[float]]) -> list[float]:
        """Cosine Similarity"""
        vec_1, vec_2 = x
        return [sum([x_i * y_i for x_i, y_i in zip(vec_1, vec_2)]) / (math.sqrt(sum([x_i**2 for x_i in vec_1])) * math.sqrt(sum([y_i**2 for y_i in vec_2])))]


def merge(features: list[Sequence[list[Any]]]) -> list[Any]:
    """Merge arrays"""
    values = []
    for feature in features:
        for value in feature:
            values.append(value)
    return values


def get_info(X: list[list[float]]) -> list[list[float]]:
    """The function return stats about each feature in X

    Arguments:
        X::list[list[float]]
            Matrix of features

    Returns:
        X::list[list[float]]
            The same X
    """
    stats = {}
    for i in range(len(X[0])):
        stats[f"avg_{i}"] = 0
        stats[f"min_{i}"] = 0
        stats[f"max_{i}"] = 0
        stats[f"D_{i}"] = 0

    # calculate avg, min, max
    for x in X:
        for i, feature in enumerate(x):
            stats[f"avg_{i}"] += feature / len(X)
            if stats[f'min_{i}'] > feature:
                stats[f'min_{i}'] = feature
            
            if stats[f'max_{i}'] < feature:
                stats[f'max_{i}'] = feature
    
    # calculate dispersion
    for x in X:
        for i, feature in enumerate(x):
            stats[f'D_{i}'] += ((feature - stats[f'avg_{i}'])**2) / len(X)

    print('{:*^40}'.format('Statistics by feature numbers'))
    print('Number feature   Stat name          Value')
    for key, item in stats.items():
        stat, n = key.split("_")
        print("{:>}{:>20}{:>20.2f}".format(n, stat, item))
    return X


def split(X, y, train_size=0.8):
    """Split dataset by train and test part"""
    X_train = X[:int(train_size * len(X))]
    y_train = y[:int(train_size * len(X))]
    X_test = X[int(train_size * len(X)):]
    y_test = y[int(train_size * len(X)):]
    
    X_train_shuffled = []
    y_train_shuffled = []
    X_test_shuffled = []
    y_test_shuffled = []
    for i in sample(range(len(X_train)), len(X_train)):
        X_train_shuffled.append(X_train[i])
        y_train_shuffled.append(y_train[i])
    for i in sample(range(len(X_test)), len(X_test)):
        X_test_shuffled.append(X_test[i])
        y_test_shuffled.append(y_test[i])
    return X_train_shuffled, X_test_shuffled, y_train_shuffled, y_test_shuffled

def accuracy_score(y_true: list[float], y_pred: list[float], boundary: float = 0.5) -> float:
    """Accuracy metric"""
    assert len(y_true) == len(y_pred), 'Count element y_true and y_pred must be the same size'
    
    r = 0
    for y_t, y_p in zip(y_true, y_pred):
        if y_p >= boundary:
            y_p = 1
        else:
            y_p = 0
        if y_t == y_p:
            r += 1
    return r / len(y_true)

if __name__ == '__main__':
    args = get_arguments()
    
    data_files, data_plagiat1, data_plagiat2 = get_data(args.files, args.plagiat1, args.plagiat2)

    pairs, y = create_pairs(data_files, data_plagiat1, data_plagiat2, n_combinations=2)

    scaler = MinMaxScaler()
    len_string_ratio_transformer = Transformer(delete_multispace, delete_enters, len_string_ratio, scaler.fit, scaler.scale)
    len_string_ratio_features = len_string_ratio_transformer(pairs)
    
    language_constructs_transformer = Transformer(delete_comments, text_to_lower, language_constructs_analysis_re, check_similarity_constructions)
    language_constructs_features = language_constructs_transformer(pairs)

    token_analyzer = TokenAnalyzer()
    token_analysis_transformer = Transformer(
                                             delete_comments,
                                             delete_multispace,
                                             delete_enters,
                                             delete_specsymbols, 
                                             text_to_lower,
                                             tokenizer,
                                             token_analyzer.collect_tokens,
                                             token_analyzer.to_bag_words,
                                             token_analyzer.to_tf_idf,
                                             token_analyzer.cosine_similarity)

    cosine_similarity_features = token_analysis_transformer(pairs)

    # merge features
    X = [merge(i) for i in zip(len_string_ratio_features, language_constructs_features, cosine_similarity_features)]
 
    # info block
    info_transform = Transformer(get_info)
    info_transform([X])

    X_train, X_test, y_train, y_test = split(X, y)

    # Model block
    model = LogisticRegression(iterations=30, lr=0.001, use_bias=True)
    model.fit(X_train, y_train, metrics=[accuracy_score])
    y_predict = model.predict(X_test)
    print('Test score: ', accuracy_score(y_test, y_predict))
    
    # save model
    with open(args.model, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model save as: {args.model}")