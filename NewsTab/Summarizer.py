import requests
import json



def naverSummarize(title, content):
    if type(title) != str or type(content) != str:
        raise ValueError

    headers = {
        "X-NCP-APIGW-API-KEY-ID": "hg2gkdp7wf", #Client ID
        "X-NCP-APIGW-API-KEY": "29YjzPzPKlb1X9pAzlsGI0YTrKrAgYFckJaHPK2n", #Client Secret
        "Content-Type": "application/json"
    }

    url= "https://naveropenapi.apigw.ntruss.com/text-summary/v1/summarize" 

    data = {
        "document": {
            "title": title,
            "content" : content
        },
        "option": {
            "language": "ko",
            "model": "news", # Model used for summaries (general, news)
            "tone": "2", # Converts the tone of the summarized result. (0, 1, 2, 3)
            "summaryCount" : "3" # This is the number of sentences for the summarized document.
        }
    }

    #print(json.dumps(data, indent=4, sort_keys=True))
    response = requests.post(url, data=json.dumps(data), headers=headers)
    rescode = response.status_code
    if(rescode == 200):
        return response.text[12:-2]
    else:
        print("Error : " + response.text)

##############################################################################################################

import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-summarization")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-summarization").to(device)

def langaugeProcessingSummarize(content : str):
    input_ids = tokenizer.encode(content, return_tensors="pt").to(device)

    summary_text_ids = model.generate(
        input_ids = input_ids,
        bos_token_id = model.config.bos_token_id,
        eos_token_id = model.config.eos_token_id,
        length_penalty = 3.0,
        max_length = 1024,
        min_length = 32,
        num_beams = 4
    )

    return tokenizer.decode(summary_text_ids[0], skip_special_tokens=True).replace('.', '.\n')

#################################################################################################################

def summarize(title : str, content : str):
    return naverSummarize(title, content)