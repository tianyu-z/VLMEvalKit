from vlmeval.smp import *
import spacy
from spacy.cli import download
from nltk.util import ngrams
from difflib import SequenceMatcher as SM
from functools import partial
from evaluate import load as loadrouge
import uuid
import multiprocessing

vcr_score = {"Exact_Macth": 0, "Jacard": 0}
experiment_id = str(uuid.uuid4())
rouge = loadrouge("rouge", experiment_id=experiment_id)
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


# def vcr_eval(eval_file, dataset_name):
#     logger = get_logger("Evaluation")
#     if "en" in dataset_name:
#         language = "en"
#     elif "zh" in dataset_name:
#         language = "zh"
#     else:
#         raise ValueError(f"Unsupported language for dataset {dataset_name}")
#     if "easy" in dataset_name:
#         difficulty = "easy"
#     elif "hard" in dataset_name:
#         difficulty = "hard"
#     else:
#         raise ValueError(f"Unsupported difficulty for dataset {dataset_name}")
#     data = load(eval_file)
#     lt = len(data)
#     lines = [data.iloc[i] for i in range(lt)]
#     for i in tqdm(range(len(lines))):
#         line = lines[i]
#         predict = str(line["prediction"])
#         answers = eval(line["answer"])

#         for j in range(len(answers)):
#             find_best_match(answers[j], predict, language, rouge)
#     final_score_dict = {}
#     score_pth = eval_file.replace(".tsv", "_score.json")
#     dump(final_score_dict, score_pth)
#     logger.info(
#         f"VCR successfully finished evaluating {eval_file}, results saved in {score_pth}"
#     )
#     logger.info(f"Score: ")
#     for key, value in final_score_dict.items():
#         logger.info("{}:{}".format(key, value))

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


def process_match_single(
    image_id, dataset, inference_results, model, language, rouge, progress_queue
):
    """
    process the inference results for a single image and calculate the metrics

    Parameters:
    image_id (int): The image id (question id).
    dataset (HF dataset): The dataset loaded from HF.
    inference_results (dict): The dictionary containing the inference results.
    model (str): The model name.
    language (str): The language of the text. Can be "en" or "zh".
    rouge (rouge): The rouge metric object.
    progress_queue (multiprocessing.Queue): The progress queue.

    Returns:
    tuple: The image id (question_id, int) and the result per id (dict of dict of dict).
    """
    result_per_id = {image_id: {}}
    inner_key = "image"
    result = inference_results[str(image_id)].get(inner_key, None)
    if isinstance(result, dict):
        result = ""
    if isinstance(result, list):
        assert len(result) == 1
        result = result[0]
    result = result.split("Assistant: ")[-1]
    for i in range(len(dataset[image_id]["crossed_text"])):
        crossed_text = dataset[image_id]["crossed_text"][i]
        if rough_filter(result):
            find_best_match_result = find_best_match(
                crossed_text, result, language, rouge
            )
            if i == 0:
                result_per_id[image_id][inner_key] = {str(i): find_best_match_result}
            else:
                result_per_id[image_id][inner_key][str(i)] = find_best_match_result
        else:
            if i == 0:
                result_per_id[image_id][inner_key] = {
                    str(i): zero_template(crossed_text)
                }
            else:
                result_per_id[image_id][inner_key][str(i)] = zero_template(crossed_text)
    progress_queue.put(1)
    return image_id, result_per_id


def process_batch_multiprocessing(
    model,
    language,
    difficulty,
    eval_path,
    rouge,
    json_filename,
    dataset_handler,
    inference_results,
    end_index,
):
    """
    Process the batch using multiprocessing.

    Parameters:
    model (str): The model name.
    language (str): The language of the text. Can be "en" or "zh".
    difficulty (str): The difficulty of the text. Can be "easy", or "hard".
    eval_path (str): (Only work when json_filename is None.) The path include the jsons you want to calculate metrics.
    rouge (rouge): The rouge metric object.
    json_filename (str): The JSON filename. If specified, the language and difficulty will be ignored.
                    If not specified, the language and difficulty will be used to find the JSON filename.
    dataset_handler (str): The dataset handler of HF.
    output_path (str): The output path of evaluation result.
    inference_results (dict): The dictionary containing the inference results. If this is not None, the json file will be ignored.
    end_index (int): The end index of the dataset to process.
    """
    dataset = load_dataset(dataset_handler)["test"]
    if inference_results is None:
        # Find the JSON filename that includes all search strings
        if json_filename is None:
            match_strings = [model.replace("/", "-"), language, difficulty]
            json_filename = find_json_filename_includes(eval_path, match_strings)

            if json_filename is None:
                print(f"JSON file not found for {match_strings}")
                return

        # Read JSON into a dictionary
        inference_results = read_json_into_dict(json_filename)

    # Initialize overall_result dictionary
    overall_result = {
        str(image_id): {
            inner_key: {}
            for inner_key in [
                "res_stacked_image",
                "res_only_it_image",
                "res_only_it_image_small",
            ]
        }
        for image_id in range(min(end_index, len(dataset)))
    }
    language = language.replace("_", "")
    difficulty = difficulty.replace("_", "")

    # Parallel processing using multiprocessing
    pool = multiprocessing.Pool()
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    results = []

    for image_id in range(min(end_index, len(dataset))):
        results.append(
            pool.apply_async(
                process_match_single,
                args=(
                    image_id,
                    dataset,
                    inference_results,
                    model,
                    language,
                    rouge,
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
        overall_result[str(image_id)].update(result_per_id[image_id])

    return overall_result


def vcr_eval(
    model_id,
    eval_path,
    output_path,
    json_filename,
    dataset_handler,
    inference_results,
    end_index=None,
):
    """
    Main function to process the batch.

    Parameters:
    model_id (str): The model_id name.
    eval_path (str): (Only work when json_filename is None.) The path include the jsons you want to calculate metrics.
    output_path (str): The output path of evaluation result.
    json_filename (str): The JSON filename. If specified, the language and difficulty will be ignored.
                    If not specified, the language and difficulty will be used to find the JSON filename.
    dataset_handler (str): The dataset handler of HF.
    inference_results (dict): The dictionary containing the inference results. If this is not None, the json file will be ignored.
    end_index (int): The end index of the dataset to process.
    """
    rouge = load("rouge", experiment_id=experiment_id)
    if "en" in dataset_handler:
        language = "en"
    elif "zh" in dataset_handler:
        language = "zh"
    else:
        raise ValueError("Dataset handler must contain either en or zh")

    if "easy" in dataset_handler:
        difficulty = "easy"
    elif "hard" in dataset_handler:
        difficulty = "hard"
    else:
        raise ValueError("Dataset handler must contain either easy or hard")

    overall_result = process_batch_multiprocessing(
        model_id,
        matcher(language),
        matcher(difficulty),
        eval_path,
        rouge,
        json_filename,
        dataset_handler,
        inference_results,
        end_index,
    )
    modelname = model_id.replace("/", "-")
    if end_index is not None:
        filename = (
            f"{modelname}_{language}_{difficulty}_evaluation_result_{end_index}.json"
        )
    else:
        filename = f"{modelname}_{language}_{difficulty}_evaluation_result.json"
    with open(
        os.path.join(output_path, filename),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(overall_result, f, ensure_ascii=False, indent=4)
    return overall_result
