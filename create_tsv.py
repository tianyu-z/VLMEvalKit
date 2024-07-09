import base64
import csv

from datasets import load_dataset
from tqdm import trange


def create_and_update_tsv(file_path):
    # Create and write the header
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['index', 'image', 'question', 'answer'])

    index = 0

    def add_row(image_path, question, answer):
        nonlocal index

        # Encode image to base64
        with open(image_path, 'rb') as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # Prepare the row data
        row = [
            str(index),
            image_base64,
            question,
            str(answer),  # Convert answer to string in case it's a list
        ]

        # Append the row to the TSV file
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(row)

        index += 1

    return add_row


def get_question(language):
    if language == 'en':
        q = ('What is the covered texts in the image? '
             'Please restore the covered texts without outputting the explanations.')
    elif language == 'zh':
        q = '图像中被覆盖的文本是什么？请在不输出解释的情况下还原被覆盖的文本。'
    else:
        raise ValueError(f'Language {language} not supported.')
    return q


for lang in ['en', 'zh']:
    for diff in ['easy', 'hard']:
        dataset_name = f'vcr-org/VCR-wiki-{lang}-{diff}-test-500'
        dataset = load_dataset(f'vcr-org/VCR-wiki-{lang}-{diff}-test-500')['test']
        tsv_file_path = dataset_name.split('/')[1] + '.tsv'
        add_row = create_and_update_tsv(tsv_file_path)
        question = get_question(lang)
        for i in trange(500):
            # save image
            dataset[i]['stacked_image'].save('tmp.png')
            add_row(
                'tmp.png',
                question,
                dataset[i]['crossed_text'],
            )
