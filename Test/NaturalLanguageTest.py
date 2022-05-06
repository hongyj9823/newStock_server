import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-summarization")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Current device:', device)
model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-summarization").to(device)

news_text =  """[서울=뉴시스] 이인준 기자 = 삼성전자 경계현 DS부문장(사장)이 최근 주요 고객사들을 만나기 위해 미국 출장을 다녀온 것으로 알려졌다.
18일 업계에 따르면 경 사장은 최시영 삼성전자 파운드리(위탁생산) 사업부 사장 등 삼성전자 DS부문 핵심 경영진과 함께 지난주 미국 출장을 소화하고 귀국했다.
경 사장의 이번 출장은 최근 인텔의 파운드리 시장 진출과 대형 투자 선언, 삼성 파운드리 사업부의 수율(합격품 비율) 논란 등으로 고객 확보 경쟁이 치열한 가운데 진행된 것이어서 업계의 주목을 받고 있다.
삼성전자 관계자는 "경영진 일정에 관해 확인해줄 수 없다"고 밝혔으나, 업계에서는 그가 이번 출장길에 대형 반도체 설계 업체를 차례로 방문했을 것으로 보고 있다. 그래픽처리장치(GPU) 회사 엔비디아, 모바일용 칩 강자 퀄컴 등 주요 고객사를 찾아간 것으로 알려졌다.
경 사장이 직접 고객사를 찾아간 배경은 최근 수율 논란에 대해 적극적으로 해명하고, 향후 공정 로드맵에 대한 자신감을 피력하며 협력을 확대하자는 뜻을 제안하기 위한 것으로 풀이된다.
한편 지난 2019년 삼성전자는 '2030 시스템반도체 1위 비전'를 선언하고 파운드리 사업을 미래 먹거리로 육성하고 있다.
현재 파운드리 시장의 절반 이상을 대만의 TSMC가 점유하고 있는 가운데, 삼성전자는 지난해 말부터 차이를 좁히며 추격을 벌이고 있다. 이런 가운데 인텔은 최근 TSMC와 삼성전자보다 한 발 더 빠른 초미세 공정 로드맵을 제시하면서 도전장을 내민 상태다. 이에 앞으로 고객 확보 경쟁은 한층 더 치열해질 것으로 업계는 보고 있다."""

input_ids = tokenizer.encode(news_text, return_tensors="pt").to(device)


summary_text_ids = model.generate(
    input_ids = input_ids,
    bos_token_id = model.config.bos_token_id,
    eos_token_id = model.config.eos_token_id,
    length_penalty = 3.0,
    max_length = 512,
    min_length = 32,
    num_beams = 4
)

print(tokenizer.decode(summary_text_ids[0], skip_special_tokens=True))
