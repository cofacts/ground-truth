import argparse
from pathlib import Path
from collections import Counter
import json
import zipfile

# import yaml


# input example
"""
{'createdAt': '2018-05-17T11:37:01.994Z',
 'hyperlinks': '[object Object]',
 'id': '35nydx5axcgn',
 'reference': 'LINE',
 'tags': [8, 10],
 'text': '盡一份力，可以擁有身體健康檢查表，又可有500塊錢的車馬費！\n\n【好康報報！中央研究院在做人體慢性病的研究，需要二十萬份的名額】\n\n條件是：本國人，\n三十至七十歲內，沒有癌症，空腹六小時抽血驗尿，就有五百元全家或是全聯或是7/11禮卷/儲值卡，現場給你喔！\n還有二週後會寄健康檢查表給我們耶！\n中研院為了建立台灣人體生物資料庫，探討疾病與環境因素的交互作用，招募30~70歲:共20萬名共襄盛舉：\n1.免費驗血驗尿，含糖化血色素及高密膽固醇，肺功能檢測，骨鬆檢測。\n2.檢測完提供早餐並補貼每人車馬費7-11儲值卡$500元，利人利己。\n快去報名~\n\nhttp://www.twbiobank.org.tw/new_web/index.php',
 'url': 'https://cofacts.g0v.tw/article/35nydx5axcgn'
 }
"""

# output example
"""
{
  "classificationAnnotations": [{
    "displayName": "8"
    },{
    "displayName": "10"
  }],
  "textContent": '盡一份力，可以擁有身體健康檢查表，又可有500塊錢的車馬費！\n\n【好康報報！中央研究院在做人體慢性病的研究，需要二十萬份的名額】\n\n條件是：本國人，\n三十至七十歲內，沒有癌症，空腹六小時抽血驗尿，就有五百元全家或是全聯或是7/11禮卷/儲值卡，現場給你喔！\n還有二週後會寄健康檢查表給我們耶！\n中研院為了建立台灣人體生物資料庫，探討疾病與環境因素的交互作用，招募30~70歲:共20萬名共襄盛舉：\n1.免費驗血驗尿，含糖化血色素及高密膽固醇，肺功能檢測，骨鬆檢測。\n2.檢測完提供早餐並補貼每人車馬費7-11儲值卡$500元，利人利己。\n快去報名~\n\nhttp://www.twbiobank.org.tw/new_web/index.php',
  "dataItemResourceLabels": {
    "aiplatform.googleapis.com/ml_use": "training|test|validation"
  }
}
"""

# SCHEMA = yaml.load(
#     open(
#         "schema_dataset_ioformat_text_classification_multi_label_io_format_1.0.0.yaml"
#     ),
#     Loader=yaml.FullLoader,
# )


def transform(sample: dict) -> dict:
    result = {}
    result["textContent"] = sample["text"]
    result["classificationAnnotations"] = [{"displayName": t} for t in sample["tags"]]
    return result


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Process a zip file.")
    parser.add_argument(
        "zip_file_path",
        nargs="?",
        type=str,
        help="Path to the zip file",
        default="20211204_14859.zip",
    )
    args = parser.parse_args()

    zip_file_path = Path(args.zip_file_path)

    # Open the .zip file in read mode
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        # zip_ref.namelist()  # show all the files in zip
        sample_files = list(
            filter(
                lambda x: not x.startswith("_") and x.endswith(".json"),
                zip_ref.namelist(),
            )
        )
        samples = [json.load(open(p)) for p in sample_files]

        tag_arrays = [x["tags"] for x in samples]
        # get single file
        # sample = json.load(open("20200324_14908/26054.json"))

        counter = Counter()

        for a in tag_arrays:
            counter.update(a)
        print("Showing counts for each tag number")
        for k, v in sorted(counter.items(), key=lambda x: x[1], reverse=True):
            print(f"Tag {k} appears {v} times")

        results = [transform(s) for s in samples]

        jsonl_file_path = zip_file_path.with_suffix(".jsonl")

        # Write the list of dictionaries to the JSONL file
        with open(jsonl_file_path, "w", encoding="utf-8") as jsonl_file:
            for entry in results:
                # Dump each dictionary as a JSON string and write it to the file
                json.dump(entry, jsonl_file, ensure_ascii=False)
                jsonl_file.write("\n")  # Add a newline character to separate entries

        print(f"Data has been written to {jsonl_file_path}")
