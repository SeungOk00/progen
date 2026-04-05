1. 나는 지금 Qlora, lora 방법으로 finetuning 방법 사용 
huggingface에서 모델을 불러올거임
data는 pfam의 PF02763, PF09009,PF03494 해당코드 단백질 서열을 이용할거임
이때, 중복되는 서열은 제거하고, clustering을 진행할거임, 이거를 데이터로 확인할거야 ->cd-hit으로 clustering, heatmap 그릴거임
또한, PF02763,PF09009 단백질이 서열이 메인이고, 데이터 수가 제한적으로 PF03494가 보조임으로, 중복데이터를 제외한 나머지 PF02763,PF09009 의 단백질 서열과 같은 개수의 서열로 데이터를 만듦, 데이터는 우선 500개
서열의 헤더를
 Header
: sequence accession number,  Pfam entry ID, protein name, molecular function(EC Number)
버전1이 있고,
서열의 헤더릉 단순히 Diphtheria toxin  이렇게 변경한거
Exotoxin A catalytic ,NAD+-diphthamide ADP-ribosyltransferase activity
이렇게 각각 하나씩만 넣은 v2-v4까지 만들어줘
기존 소스코드를 활용해서 어떻세 하면좋은지 알려주렴
어떤 분석 그래프를 빼면 좋은지