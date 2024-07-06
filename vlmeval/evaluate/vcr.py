import multiprocessing
import uuid
from difflib import SequenceMatcher as SM
from functools import partial

import evaluate
import spacy
from nltk.util import ngrams
from spacy.cli import download

from vlmeval.smp import *

experiment_id = str(uuid.uuid4())
rouge = evaluate.load("rouge", experiment_id=experiment_id)
# Download the English and Chinese models
try:
    nlp_en = spacy.load("en_core_web_sm")
except:
    download("en_core_web_sm")
    nlp_en = spacy.load("en_core_web_sm")

try:
    nlp_zh = spacy.load("zh_core_web_sm")
except:
    download("zh_core_web_sm")
    nlp_zh = spacy.load("zh_core_web_sm")

nlp = {"en": nlp_en, "zh": nlp_zh}


def rough_filter(answer_text):
    if "I can't" in answer_text:
        return False
    elif "I cannot" in answer_text:
        return False
    elif "sorry" in answer_text.lower():
        return False
    if "无法" in answer_text:
        return False
    elif "抱歉" in answer_text:
        return False
    else:
        return True


def zero_template(crossed_text):
    return {
        "crossed_text": crossed_text,
        "max_sim_val": 0,
        "max_sim_string": "",
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "jaccard": 0,
        "rouge1": 0,
        "exact_match": 0,
    }


def tokenize(text, language):
    """
    Tokenize the text and return the tokens.

    Parameters:
    text (str): The text to tokenize.
    language (str): The language of the text.

    Returns:
    list: The list of tokens.
    """
    assert language in ["en", "zh"]
    nlp_language = nlp[language]
    processed_text = nlp_language(text)
    return [token.text for token in processed_text]


def find_best_match(needle, hay, language, rouge):
    """
    Finds the best matching n-gram in the haystack for the given needle.

    Parameters:
    needle (str): The string to find.
    hay (str): The text to search within.

    Returns:
    tuple: The highest similarity value and the best matching string.
    """

    assert language in ["en", "zh"]
    tokens_hay = tokenize(hay, language)
    tokens_needle = tokenize(needle, language)

    splitter = "" if language == "zh" else " "
    ngrams_ = ngrams(tokens_hay, len(tokens_needle))
    max_sim_val = 0
    max_sim_string = ""
    max_sim_ngram = []
    tokens_needle_set = set(tokens_needle)
    ngrams_hasjoint = [
        ngram for ngram in ngrams_ if not set(ngram).isdisjoint(tokens_needle_set)
    ]

    for ngram in ngrams_hasjoint:
        hay_ngram = splitter.join(ngram)
        similarity = SM(None, hay_ngram, needle).ratio()
        if similarity > max_sim_val:
            max_sim_val = similarity
            max_sim_string = hay_ngram
            max_sim_ngram = ngram

    # Evaluate
    if len(max_sim_ngram) == 0:
        return {
            "crossed_text": needle,
            "max_sim_val": 0,
            "max_sim_string": "",
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "jaccard": 0,
            "rouge1": 0,
            "exact_match": 0,
        }
    pred_set = set(max_sim_ngram)
    ref_set = set(tokens_needle)
    correct_tokens = pred_set.intersection(ref_set)
    len_correct_tokens = len(correct_tokens)

    precision = len_correct_tokens / len(pred_set)
    recall = len_correct_tokens / len(ref_set)
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    union = pred_set.union(ref_set)
    jaccard = len_correct_tokens / len(union) if len(union) > 0 else 0
    rouge_1 = rouge.compute(
        predictions=[max_sim_string],
        references=[needle],
        tokenizer=partial(tokenize, language=language),
        rouge_types=["rouge1"],
    )["rouge1"]
    exact_match = float(list(max_sim_ngram) == list(tokens_needle))
    out = {
        "crossed_text": needle,
        "max_sim_string": max_sim_string,
        "max_sim_val": max_sim_val,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "rouge1": rouge_1,
        "exact_match": exact_match,
    }
    return out


def process_match_single_new(image_id, prediction, answer, language, progress_queue):
    """
    process the inference results for a single image and calculate the metrics

    Parameters:
    image_id (int): The image id (question id).
    prediction (str): The prediction text.
    answer (Union[str, List[str]]): The answer text, or a list of answer texts. The masked n-grams in the image.
    language (str): The language of the text. Can be "en" or "zh".
    rouge (rouge): The rouge metric object.
    progress_queue (multiprocessing.Queue): The progress queue.

    Returns:
    tuple: The image id (question_id, int) and the result per id (dict of dict of dict).
    """
    result_per_id = {image_id: {}}
    if isinstance(answer, str):
        answer = eval(answer)
    assert isinstance(answer, list)
    result = prediction.split("Assistant: ")[-1]
    for i, crossed_text in enumerate(answer):
        if rough_filter(result):
            find_best_match_result = find_best_match(
                crossed_text, result, language, rouge
            )
            if i == 0:
                result_per_id[image_id] = {str(i): find_best_match_result}
            else:
                result_per_id[image_id][str(i)] = find_best_match_result
        else:
            if i == 0:
                result_per_id[image_id] = {str(i): zero_template(crossed_text)}
            else:
                result_per_id[image_id][str(i)] = zero_template(crossed_text)
    progress_queue.put(1)
    return image_id, result_per_id


def vcr_eval(eval_file, language, difficulty, **kwargs):
    vcr_score_list = {"Exact_Match": [], "Jaccard": []}
    vcr_score = {"Exact_Match": 0, "Jaccard": 0}
    logger = get_logger("Evaluation")
    data = load(eval_file)

    lt = len(data)
    lines = [data.iloc[i] for i in range(lt)]

    pool = multiprocessing.Pool()
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    results = []

    overall_results = {str(image_id): {} for image_id in range(len(lines))}

    for instance_id, instance in enumerate(lines):
        results.append(
            pool.apply_async(
                process_match_single_new,
                args=(
                    str(instance_id),
                    instance["prediction"],
                    instance["answer"],
                    language,
                    progress_queue,
                ),
            )
        )
    pool.close()

    # Display progress bar
    for _ in tqdm(range(len(results))):
        progress_queue.get()

    pool.join()

    # Merging results into overall_result
    for result in results:
        image_id, result_per_id = result.get()
        overall_results[str(image_id)].update(result_per_id[image_id])
        for blank_id_str in result_per_id[image_id].keys():
            vcr_score_list["Exact_Match"].append(
                result_per_id[image_id][blank_id_str]["exact_match"]
            )
            vcr_score_list["Jaccard"].append(
                result_per_id[image_id][blank_id_str]["jaccard"]
            )
        vcr_score["Exact_Match"] = np.mean(vcr_score_list["Exact_Match"])
        vcr_score["Jaccard"] = np.mean(vcr_score_list["Jaccard"])
    results_out = {
        k: v for i in range(len(results)) for k, v in results[i].get()[1].items()
    }
    results_with_metrics = {
        "Exact_Match": vcr_score["Exact_Match"],
        "Jaccard": vcr_score["Jaccard"],
        "Predictions": results_out,
    }
    score_pth = eval_file.replace(".xlsx", f"{language}_{difficulty}_score.json")
    dump(results_with_metrics, score_pth)
    logger.info(
        f"VCR successfully finished evaluating {eval_file}, results saved in {score_pth}"
    )
    logger.info("Score: ")
    for key, value in vcr_score.items():
        logger.info("{}:{}".format(key, value))
