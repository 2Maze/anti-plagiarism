import argparse
import pickle

from train import *


def get_arguments() -> argparse.Namespace:
    """
    This function get arguments from user
    """

    parser = argparse.ArgumentParser(description='Checks for plagiarism two files')
    parser.add_argument('file_of_pairs', type=str, help='File of pairs')
    parser.add_argument('file_of_scores', type=str, help='Output file of results')
    parser.add_argument(
        '--model',
        type=str,
        default='model.pkl',
        help='Trained model path (default: model.pkl)'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    # load model 
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    
    pairs = []
    for pair in open(args.file_of_pairs, mode="r", encoding='utf-8').read().split("\n"):
        pair = pair.split()
        pairs.append([open(pair[0], mode="r", encoding='utf-8').read(), open(pair[1], mode="r", encoding='utf-8').read()])

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

    X = [merge(i) for i in zip(len_string_ratio_features, language_constructs_features, cosine_similarity_features)]
    y_predict = model.predict_proba(X)

    with open(args.file_of_scores, "w") as file:
        for prediction in y_predict:
            file.write(str(round(prediction, 2)) + "\n")