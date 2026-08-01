코드 공개 체크리스트

- 학습에 사용한 데이터셋
    - 각 이미지와 캡션들
    - 네거티브샘플 생성 코드
    - 전처리 코드 (구체적으론 모르지만 아마 학습 코드에 포함??)
- 학습 코드
    - tasa 프레임워크의 각 과정은 논문과 연결지어 주석 필요
    - random seed 등 랜덤성에 관여하는 하이퍼파라미터들은 appendix에서 언급 필요
- 평가 코드
- 계산 환경. GPU/CPU models; amount of memory; operating system; names and versions of relevant software libraries and frameworks
- 하이퍼파라미터 설정

깃허브

GitHub - leewoojye/MedCLIP-SAMv2: Improving tumor segmentation performance using DPO based on MedCLIP-SAMv2

1. $I$ 입력하여 $I^{-k}$ 생성하는 방법
- [어떤 모델 사용하였는지, 어떤 프롬프트(어떤 논문 참고하였는지) 사용하였는지, 메타데이터 입력 등]
- 도메인 마다 다르면, 방법마다 작성

(ex. A의 경우 B모델 사용하였으며, C연구와 같은 파이프라인으로 생성하였다. D는…)

- Breast tumor (BUS-CoT, UDIAT)
    
    full dataset: 
    
    - UDIAT: https://huggingface.co/datasets/WOOJYE/udiat/tree/main
    - BUS-CoT: https://huggingface.co/datasets/WOOJYE/bus_cot_preprocessed/tree/main
    
    유방초음파 음성 생성 설정
    
    - 스크립트: experiments/bus_cot_image2prompt_trial/generate_sd_ipc_batch.py
    
    ```json
      - 기본 생성 모델: stable-diffusion-v1-5/stable-diffusion-inpainting, float16.
      - 스케줄러: 기존 스케줄러 설정을 유지한 DDIMScheduler.
      - inference steps: 50
      - CFG guidance scale: 5.0
      - safety checker: 비활성화.
      - 배치: GPU당 batch size 2, 3개 shard. 초기 519장과 resume 387장을 합쳐 최종 906장이 생성됨. (testset 기준)
      - 난수: 각 정렬된 이미지의 전역 index에 대해 20260728 + index; shard/resume 여부와 무관하게 동일 이미지에는
        동일 seed를 사용.
    ```
    
    - 정상 조건 이미지: 고정된 BUSI 정상 이미지 1장 assets/busi_normal_1.png(BUSI normal 1번 이미지). 입력 이미지마다 새 정상 이미지를 고 르지 않았음.
    
    정상 이미지를 텍스트 조건으로 바꾸는 부분(image2prompt) 설정
    
    - 레퍼런스 논문: The CLIP Model is Secretly an Image-to-Prompt Converter (NeurIPS 2023) https://arxiv.org/abs/2305.12716
    
    ```json
      - OpenAI CLIP openai/clip-vit-large-patch14의 vision CLS embedding(1024차원)을 사용.
      - W_map = pinv(W_text) @ W_visual의 closed-form 변환을 적용.
      - SVD cutoff는 0.3, 실제 유지된 singular value 수는 607.
      - 결과는 SD CLIP text hidden space의 768차원 token임.
      - SD의 native empty prompt에서 시작 token 위치 0은 유지하고, 위치 1–76은 같은 정상 이미지-derived token으로
        반복함. 즉 자연어 텍스트 프롬프트는 사용하지 않았음.
    
      - negative prompt도 빈 문자열의 native SD embedding임.
      - Stable Diffusion 구조 자체는 변경하지 않았음.
    ```
    
    마스크·해상도 설정
    
    ```json
      - 원본 lesion mask를 grayscale로 읽고, dilation_px=9를 적용.
      - 여기서 dilation=9는 19×19 square MaxFilter이므로 각 방향으로 최대 9픽셀 확장임.
      - 원본은 Lanczos로 512×512, 마스크는 nearest-neighbor로 512×512로 맞춘 뒤 inpainting함.
      - 생성 결과는 다시 원본 크기로 Lanczos resize함.
      - 출력 이름은 원본 이름을 유지함. JPEG quality/subsampling은 코드에서 별도 지정하지 않아 Pillow 기본값을
        따름.
    ```
    
- Brain MRI (MMNeuro, Figshare)
    
    full dataset: https://huggingface.co/datasets/WOOJYE/bus_cot_preprocessed/tree/main
    
    pre-trained diffusion model ckpt: https://huggingface.co/WOOJYE/ssldiffusion_brainmri/tree/main/checkpoints
    
    Figshare Brain MRI 음성 생성 설정
    
    - DDPM/run_figshare_test_dilation9_3gpu.sh DDPM/sample_figshare_dilated_sharded.py 설정으로 생성
    
    ```json
      - 대상: Figshare test image/mask pair 3,064개. (testset 기준)
      - 체크포인트: checkpont/step_122000.pt
      - 사용 가중치: raw model이 아닌 EMA shadow weight.
      - 실제 checkpoint global step: 122,000
      - EMA decay: 0.9999
      - 난수: 정렬된 데이터셋 index별 42 + index.
      - 3개 독립 shard로 index rank, rank+3, ...를 나눴습니다. 로그상 실제 GPU는 0번 RTX Pro 5000, 1·2번 RTX A6000
        임.
    ```
    
    DDPM/UNet 및 diffusion 설정
    
    ```json
      - 입력 crop: 224×224
      - UNet 조건 입력 3채널: [masked baseline, lesion mask, Gaussian noise channel]
      - 출력: 1채널 epsilon prediction
      - base channels 128, residual block 2, channel multiplier (1, 1, 2, 2, 4, 4)
      - attention: downsample rate 16
      - dropout 0, learned variance 비활성화, 총 파라미터 약 113.67M
      - 1,000 diffusion step, linear beta schedule (β=0.0001 → 0.02)
      - epsilon prediction, fixed-small variance, MSE loss, timestep rescaling 없음
      - 추론도 DDIM이 아니라 1,000-step DDPM reverse sampling p_sample_loop_known을 사용.
      - clip_denoised=True
    ```
    
    학습 체크포인트의 설정
    
    ```json
      - batch size 8, learning rate 1e-4, AMP 활성화, seed 42
      - 계획 학습 step은 2,850,000이지만 사용 checkpoint는 122,000 step.
      - training mask mode는 synthetic_healthy: 종양 모양을 이동/회전/반전해 종양과 겹치지 않는 정상 뇌 영역에
        synthetic hole을 만듦.
    
      - 실제 종양은 context에서 숨기고(hide_tumor_in_context=True), tumor loss weight는 0; hole loss 1.0, context
        loss 0.05임.
    ```
    
    Figshare 추론(훈련된 diffusion model로 음성 생성하는 단계) 시 전처리와 합성
    
    ```json
      - grayscale 입력을 0.1–99.9 percentile clipping 후 [0,1] min–max 정규화합니다.
      - 실제 종양 mask의 bounding box로 ROI를 잡습니다. 한 변은 max(224, bbox 높이+32, bbox 너비+32)이고 context
        margin은 16px입니다.
    
      - 그 뒤 최종 적용 mask에 dilation_size=9를 적용합니다. 이는 9×9 MaxFilter, 즉 각 방향 최대 4픽셀 확장입니다.
      - 따라서 유방초음파의 dilation_px=9(19×19, 반경 9px)과 Brain MRI의 dilation_size=9(9×9, 반경 4px)은 의미가 다
        릅니다.
    
      - ROI는 bilinear로 224×224, mask는 nearest-neighbor로 224×224로 변환합니다.
      - 생성 prediction에는 PIL GaussianBlur radius 1.075를 적용한 뒤 원래 ROI 크기로 bilinear 복원합니다.
      - raw는 ROI 전체를 prediction으로 교체하고, blended는 dilation된 마스크 내부만 hard replacement합니다. AUROC/
        FPR 평가 CSV의 음성샘플은 blended/의 3,064장을 사용했습니다.
    ```
    
- Chest CT (Kaggle)
    
    full dataset (testset): https://huggingface.co/datasets/WOOJYE/chest_ct/tree/main
    
    가우시안 노이즈 샘플링 설정
    
    - 해상도: 512×512. 원본은 grayscale(L) JPEG이며, 생성 전에 RGB로 복제 변환
    - 노이즈: 선택된 픽셀의 RGB 채널 각각에 독립적으로 N(μ=0, σ=50) additive Gaussian noise를 적용함. 분산은 2500.
        - 다만 전체 이미지를 JPEG로 다시 저장하므로, 마스크 밖에는 새 노이즈가 더해지지 않더라도 JPEG 재압축에 따른 미세한 변화는 생길 수 있음.
    - alpha blending: 없음. 원본 RGB + noise 뒤 [0,255]로 clipping하고 uint8로 변환
    - 난수 시드: 기본 시드 20250728
        - test_i bronchus: 20250728 + 10*i
        - heart: 위 값 +1
        - lung: 위 값 +2 따라서 이미지·부위마다 서로 다른 독립 노이즈가 생성
    - dilation / erosion / feathering: 없음. 선택된 마스크는 hard boundary
    - 저장: RGB JPEG, quality 95, chroma subsampling 4:2:0
- Natural Image (COD10K, MAS3K)
    
    full dataset:
    
    - COD10K: https://huggingface.co/datasets/WOOJYE/COD10K/tree/main
    - MAS3K: https://huggingface.co/datasets/WOOJYE/gmpo_mas3k/tree/main
    
- example (재윤)
    
    
    ```
    noise = N(μ=0, σ=50)       # RGB 채널별 독립 난수
    negative = clip(original + noise, 0, 255)
    ```
    
    - 원시 Gaussian: `μ ≈ 0`, `σ ≈ 50`, 분산 `≈ 2500`
    - alpha blending: 해당 없음. 원본에 대한 회귀 기울기가 0.85–0.97이며, clipping을 고려하면 additive 모델과 일치합니다.
    - 채널: 원시 RGB별 독립 noise입니다. 저장 JPEG의 4:2:0 chroma subsampling 때문에 파일을 직접 보면 채널 상관이 약 `0.80–0.84`로 높아 보입니다.
    - 공간 특성: 픽셀 독립 white noise. 잔차의 이웃 픽셀 상관은 `0.05–0.15` 수준으로 Gaussian blur 같은 공간 상관 노이즈가 아닙니다.
    - 적용 영역:
        - 폐: 파란 마스크
        - 심장: 초록 마스크
        - 기관지: 빨간 마스크
    - 마스크 내부에만 적용됐습니다. `|Δ| ≥ 20` 기준으로 마스크 밖 변화는 폐 0.3%, 심장 0.5%, 기관지 2.5%뿐이며, 모두 경계에서 4px 이내 JPEG DCT artifact입니다.
    - dilation/erosion/feathering: 확인되지 않았습니다. 실질적 경계는 hard boundary입니다.
    - clipping: `[0, 255]` uint8 clipping. 어두운 영역에서 평균 변화가 양수로 치우치고 분산이 작아지는 현상이 정확히 관찰됩니다.
    - 저장 형식: 512×512, 8-bit RGB JPEG, quality 95 수준의 양자화 테이블, 4:2:0 subsampling.
    - 모든 구조와 이미지에 동일한 noise strength가 적용된 것으로 보입니다. 100개 이미지를 대조했을 때 noise pattern 재사용 상관은 `~0`이었습니다.

1. $T$ 와 $T^{-k}$ 생성하는 방법
- basecaption: [어떤 모델 사용하였는지, 어떤 프롬프트(어떤 논문 참고하였는지) 사용하였는지, 메타데이터 입력 등]
    - Breast Ultrasound (BUS-CoT)
        - GPT를 사용하여 병변 정보를 포함하지 않는 영문 base caption 100개를 생성하였다. 각 caption은 “Breast ultrasound image”로 시작하며, 일반적인 breast tissue, parenchyma, background echotexture, tissue composition, sonographic appearance 또는 anatomy만 기술한다. 생성된 100개 caption은 순환 배정하였다.
        - 양성 caption의 대괄호 안 slot은 BUS-CoT metadata의 병변명을 그대로 사용하고, 양성 및 음성 caption은 동일한 base caption을 공유하였다. 즉, 각 쌍은 “[positive metadata] {base caption}” 및 “[negative slot] {same base caption}”의 형태이다.
        - Base caption 생성을 위해 GPT에 사용한 프롬프트는 다음과 같다.

          > Generate exactly 100 diverse, one-sentence English base captions for breast ultrasound images.
          >
          > Each caption must begin with “Breast ultrasound image” and describe only general breast tissue, parenchyma, background echotexture, tissue composition, sonographic appearance, or anatomy.
          >
          > The captions will be shared by both positive and negative DPO pairs. Therefore, do not mention any lesion, mass, tumor, cyst, malignancy, diagnosis, pathology, BI-RADS category, measurement, location, or patient information.
          >
          > Use neutral professional wording. End every caption with a period.
          >
          > Return only the 100 captions, one caption per line, without numbering, bullet points, quotation marks, or explanations.
        
    - Brain MRI (Figshare)
        
        
    - Chest CT (Kaggle)
        
        DPO 훈련데이터 캡션 생성 방식과 동일
        
    - Natural Image (COD10K, MAS3K)
        
        DPO 훈련데이터 캡션 생성 방식과 동일
        

- 존재 부재 캡션: [어떤 모델 사용하였는지, 어떤 프롬프트(어떤 논문 참고하였는지) 사용하였는지, 메타데이터 입력 등]
    - Breast Ultrasound (BUS-CoT)
        - GPT를 사용하여 정상ㆍ무소견 또는 병변 부재를 나타내는 서로 다른 negative slot 50개를 생성하였다. 각 slot은 대괄호 안에 삽입한 짧은 임상 표현이며, 50개 slot을 train/validation 쌍에 균등하게 무작위 배정하였다.
        - Negative slot 생성을 위해 GPT에 사용한 프롬프트는 다음과 같다.

          > Generate exactly 50 distinct short English negative slot phrases for breast-ultrasound DPO caption pairs.
          >
          > Each phrase will be inserted verbatim inside square brackets before a shared generic breast-ultrasound base caption. Produce only the slot phrase itself; do not include square brackets, numbers, bullets, quotation marks, or explanations.
          >
          > Each slot should express a normal, unremarkable, preserved, or lesion-absent breast ultrasound finding. Use concise clinical fragments rather than full sentences. Use lower case except for standard notation such as “BI-RADS 1 negative”.
          >
          > Include varied wording based on expressions such as no suspicious finding, no focal lesion, normal-appearing tissue, preserved tissue/parenchymal appearance, unremarkable sonographic appearance, and negative ultrasound finding. Do not name a positive lesion or pathology.
          >
          > Return exactly 50 phrases, one per line.
    - Brain MRI (Figshare)
    - Chest CT (Kaggle)
        
        DPO 훈련데이터 캡션 생성 방식과 동일
        
    - Natural Image (COD10K, MAS3K)
        
        DPO 훈련데이터 캡션 생성 방식과 동일
        

### Breast tumor (UDIAT; 326 rows)

- 라벨 원천: 원본 UDIAT 폴더 Benign, Malignant 및 생성 음성샘플 폴더를 각각 benign, malignant, normal로 지정. 원본 종양 영상 163장과 음성샘플 163장
- 고정 T
    - benign → There is a benign tumor.
    - malignant → There is malignant tumor.
    - normal → There is no visible tumor.
- a1_inference/build_udiat_classwise_eval_csv.py의 CLASSES 상수로 직접 지정.

### Brain MRI (Figshare; 6,128 rows)

- 라벨 원천: 원본 3,064장의 종양 유형은 data/Figshare/test_figshare3064/figshare_tumor_type_metadata0.json의 label_name에서 읽고, 같은 ID의 DDPM dilation=9 blended 음성샘플 3,064장은 normal로 지정
- 고정 T
    - meningioma → There is a meningioma tumor.
    - glioma → There is a glioma tumor.
    - pituitary → There is a pituitary tumor.
    - normal → There is no visible tumor.
- 이 네 문장은 a1_inference/build_figshare_classwise_eval_csv.py의 CAPTIONS 사전으로 고정. 종양 유형은 메타데이터에서만 얻음

### Chest CT (800 rows)

- 폐+기관지/기관지+심장/심장+폐/기관지+심장+폐 공유 캡션은 민성’s caption을 참고
- 노이즈 클래스의 문장은 노이즈가 적용되지 않은 두 구조를 기술하고, 원본은 세 구조가 모두 보존되었다고 기술

### Natural images (COD10K and MAS3K)

- 라벨 원천: COD10K는 테스트 메타데이터의 69개 camouflage subclass와 no camouflaged animal, MAS3K는 33개 원본 객체 클래스와 같은 no-object 클래스를 사용
- 최종 scene-caption CSV의 고정 T
    - 원본 객체 행: There is an [class name] present in the scene.
    - no-object 행: There is no camouflaged animal present in the scene.
    - 관사 an은 클래스명에 따라 교정하지 않고 모든 객체 클래스에 문자 그대로 유지
- 평가 파일
    - COD10K: data/COD10K/test/cod10k_original2026_no_object2026_subclass70_scene_caption_pairs_paths_fixed.csv — 원본 2,026장 + no-object 2,026장 = 4,052행, 70클래스
    - MAS3K: data/MAS3K/mas3k_original582_no_object582_classwise_scene_caption_pairs_paths_fixed.csv — 원본 582장 + no-object 582장 = 1,164행, 34클래스

### 재현 절차

각 CSV 행에 image_path, class_label, caption을 기록하고 평가 시 클래스별 하나의 고유 캡션만 텍스트 인코더에 입력한다. 이어 a1_inference/clip4retrofit_ovr_metrics.py가 클래스별 one-vs-rest AUROC와 ROC 곡선에서 처음으로 TPR ≥ 0.95가 되는 지점의 FPR을 계산한다. 최종 macro 값은 클래스별 값을 동일 가중치로 산술평균한다. 실제 캡션 생성 단계에는 seed, temperature, top-p, 모델 버전 같은 생성형 언어모델 하이퍼파라미터가 존재하지 않는다. 위 역구성 프롬프트는 새 CSV를 같은 규칙으로 다시 만들 때에만 사용할 수 있다.

1. 학습 파라미터 
- 학습 파라미터

- GPU

1. 학습데이터 및 테스트셋 구분[몇대몇인지, 랜덤샘플링인지, 이미 분할되있는 경우 그렇게썼다고 작성]

한글) 표 1의 AUROC와 macro FPR@95% TPR은 이미지 임베딩과 클래스별 고정 캡션 임베딩의 L2-정규화 코사인 유사도를 사용하여 one-vs-rest 방식으로 산출하였다. 공개 사전학습 모델은 별도 미세조정 없이 zero-shot으로 평가하였다. 유방 초음파와 뇌 MRI에서는 OpenAI CLIP ViT-L/14 및 ViT-B/32, SigLIP Base Patch16-224, MedSigLIP-448, BiomedCLIP PubMedBERT-256 ViT-B/16을 사용하였고, 의료 도메인의 GMPO global_ind 체크포인트는 BiomedCLIP 구조에 strict load하여 평가하였다. 또한 BiomedCoOp은 BiomedCLIP 기반의 공식 16-shot 프롬프트 학습 체크포인트를 사용하였다. 즉 UDIAT에는 BUSI 16-shot 체크포인트, Figshare 뇌 MRI에는 BTMRI 16-shot 체크포인트를 적용했으며, 평가 CSV 자체로 추가 학습은 수행하지 않았다. Chest CT에서는 OpenAI CLIP, SigLIP, MedSigLIP, BiomedCLIP 및 CT GMPO 체크포인트를 평가하였다. COD10K와 MAS3K 자연영상에서는 OpenAI CLIP, SigLIP, BioCLIP, BioCLIP 2를 zero-shot으로 평가하고, OpenAI CLIP ViT-B/32 구조에 로드한 각 데이터셋의 GMPO global_ind 체크포인트를 함께 비교하였다.

영어) The AUROC and macro FPR@95% TPR values in Table 1 were computed in a one-vs-rest manner from the L2-normalized cosine similarity between image embeddings and the embedding of each fixed class caption. Public pretrained models were evaluated zero-shot, without additional fine-tuning. For breast ultrasound and brain MRI, we evaluated OpenAI CLIP ViT-L/14 and ViT-B/32, SigLIP Base Patch16-224, MedSigLIP-448, and BiomedCLIP PubMedBERT-256 ViT-B/16; medical-domain GMPO global_ind checkpoints were strictly loaded into the BiomedCLIP architecture for evaluation. BiomedCoOp used its official BiomedCLIP-based 16-shot prompt-learning checkpoints: the BUSI checkpoint for UDIAT and the BTMRI checkpoint for Figshare brain MRI. No additional training was performed on the evaluation CSVs. For chest CT, we evaluated OpenAI CLIP, SigLIP, MedSigLIP, BiomedCLIP, and the CT GMPO checkpoint. For the COD10K and MAS3K natural-image domains, OpenAI CLIP, SigLIP, BioCLIP, and BioCLIP 2 were evaluated zero-shot and compared with the respective GMPO global_ind checkpoint loaded into the OpenAI CLIP ViT-B/32 architecture.

취소선 섹션은 민성’s 문서 참고

Table 1

- Brest Ultrasound
    - BUS-CoT: 논문 저자 측에서 사전 분할
    - UDIAT: impactlab 측에서 사전 분할
        - https://huggingface.co/datasets/WOOJYE/udiat/tree/main
- ~~Brain MRI~~
- ~~ChestCT~~
- ~~COD10K~~
- MAS3K
    - https://huggingface.co/datasets/WOOJYE/gmpo_mas3k/tree/main
    - 논문 측에서 사전 분할 X, 임의 8:2 train:test 분할
    - https://huggingface.co/datasets/WOOJYE/gmpo_mas3k/tree/main
    - 데이터 원천: UOVSBench는 기존 수중 분할 데이터셋 MAS3K의 이미지ㆍ마스크 쌍과 범주 라벨을 open-vocabulary segmentation 형식으로 정리하여 포함한 벤치마크이며, 본 연구에서는 그 공개 MAS3K 구성의 원본 수중생물 이미지와 클래스 정보를 사용하였다. AquaOV255 및 UOVSBench의 다른 5개 데이터셋, Earth2Ocean 방법, 그리고 UOVSBench의 분할ㆍ평가 프로토콜은 사용하지 않았다. 현재 프로젝트의 MAS3K 원본 2,910장 가운데 582장(20%)을 연구 내에서 임의로 테스트셋으로 분할하였으며, 따라서 이 582장ㆍ33개 관측 클래스의 테스트 분할은 UOVSBench가 제공한 공식 분할이 아니다. 참고: https://arxiv.org/abs/2511.07923
    - English: UOVSBench incorporates the image--mask pairs and category labels of the existing underwater segmentation dataset MAS3K after converting them to an open-vocabulary segmentation format. We used only the original underwater-animal images and class information from this released MAS3K component. We did not use AquaOV255, the other five UOVSBench dataset components, the Earth2Ocean method, or the UOVSBench split and evaluation protocol. From the 2,910 MAS3K original images in this project, we randomly selected 582 images (20%) as the test set; therefore, this test split, containing 33 observed classes, is not an official UOVSBench-provided split. Reference: https://arxiv.org/abs/2511.07923

Table 2  (segmentation)

- Brest Ultrasound
- Brain MRI
- ChestCT
- COD10K
- MAS3K
    - 논문 측에서 사전 분할 X, 임의 8:2 train:test 분할
    - https://huggingface.co/datasets/WOOJYE/gmpo_mas3k/tree/main
    - 데이터 원천: UOVSBench는 기존 수중 분할 데이터셋 MAS3K의 이미지ㆍ마스크 쌍과 범주 라벨을 open-vocabulary segmentation 형식으로 정리하여 포함한 벤치마크이며, 본 연구에서는 그 공개 MAS3K 구성의 원본 수중생물 이미지와 클래스 정보를 사용하였다. AquaOV255 및 UOVSBench의 다른 5개 데이터셋, Earth2Ocean 방법, 그리고 UOVSBench의 분할ㆍ평가 프로토콜은 사용하지 않았다. 현재 프로젝트의 MAS3K 원본 2,910장 가운데 582장(20%)을 연구 내에서 임의로 테스트셋으로 분할하였으며, 따라서 이 582장ㆍ33개 관측 클래스의 테스트 분할은 UOVSBench가 제공한 공식 분할이 아니다. 참고: https://arxiv.org/abs/2511.07923
    - English: UOVSBench incorporates the image--mask pairs and category labels of the existing underwater segmentation dataset MAS3K after converting them to an open-vocabulary segmentation format. We used only the original underwater-animal images and class information from this released MAS3K component. We did not use AquaOV255, the other five UOVSBench dataset components, the Earth2Ocean method, or the UOVSBench split and evaluation protocol. From the 2,910 MAS3K original images in this project, we randomly selected 582 images (20%) as the test set; therefore, this test split, containing 33 observed classes, is not an official UOVSBench-provided split. Reference: https://arxiv.org/abs/2511.07923

Table 3  (VQA)

- Brest Ultrasound
- Brain MRI
- camofla

위 섹션 1,2에 대한 영문단 표현

---

## 1. Generation of Target-Suppressed Images I⁻ᵏ

Let I denote an original image and I⁻ᵏ its counterfactual counterpart in which the visual evidence associated with target k is selectively suppressed. Because the appearance, acquisition process, and spatial characteristics differ substantially across breast ultrasonography, brain MRI, chest CT, and natural-image datasets, I⁻ᵏ was generated using a domain-specific procedure rather than a single universal transformation.

### Breast ultrasonography

For the BUS-CoT and UDIAT breast ultrasound datasets, lesion-suppressed images were generated using the stable-diffusion-v1-5/stable-diffusion-inpainting model in half precision. A DDIMScheduler was instantiated while retaining the scheduler configuration of the pretrained model. Inference was performed for 50 denoising steps with a classifier-free guidance scale of 5.0, and the safety checker was disabled. The documented test-set generation run used a batch size of two images per GPU and was divided into three independent shards. A total of 906 images were generated, consisting of 519 images from the initial run and 387 images from the resumed run.

Instead of conditioning the diffusion model on a manually written natural-language prompt, we used a fixed normal breast ultrasound image, assets/busi_normal_1.png, as the normal-reference condition. The same reference image was used for all input images. Following the image-to-prompt conversion procedure introduced in *The CLIP Model Is Secretly an Image-to-Prompt Converter*, the 1,024-dimensional vision CLS embedding extracted by openai/clip-vit-large-patch14 was mapped to the 768-dimensional Stable Diffusion text hidden space using the closed-form transformation

$$
W_{\mathrm{map}}
=
\operatorname{pinv}\!\left(W_{\mathrm{text}}\right)
W_{\mathrm{visual}}
$$

The singular-value decomposition cutoff was set to 0.3, retaining 607 singular values. The resulting 768-dimensional image-derived token was inserted into the Stable Diffusion text-conditioning sequence. Specifically, token position 0 of the native empty prompt was retained, whereas positions 1 through 76 were filled by repeating the same normal-image-derived token. Thus, no natural-language positive prompt was used. The negative prompt was represented by the native Stable Diffusion embedding of an empty string, and the architecture and parameters of the Stable Diffusion model itself were not modified.

The original lesion mask was loaded in grayscale and dilated using dilation_px=9. In this implementation, the operation corresponds to a 19 × 19 square maximum filter and therefore expands the mask by up to nine pixels in each direction. The source image was resized to 512 × 512 using Lanczos interpolation, whereas the binary mask was resized using nearest-neighbor interpolation. After inpainting, the generated image was resized back to the original spatial resolution using Lanczos interpolation. Output filenames were preserved. Because JPEG quality and chroma-subsampling parameters were not explicitly specified in the generation script, the corresponding Pillow defaults were used.

Generation was made deterministic at the image level. For an image at global position i in the sorted input list, the random seed was set to

$$
s_i = 20260728 + i
$$

Because the seed depended on the global sorted index rather than the local shard index, the same input image received the same seed regardless of whether it was processed in the initial run, a resumed run, or a different GPU shard.

### Brain MRI

For the Figshare brain MRI test set, target-suppressed images were generated using a conditional DDPM implemented in DDPM/sample_figshare_dilated_sharded.py and executed through DDPM/run_figshare_test_dilation9_3gpu.sh. The generation set contained 3,064 image–mask pairs. Sampling used the exponential-moving-average weights stored in checkpoint/step_122000.pt, rather than the raw model parameters. The selected checkpoint corresponded to global step 122,000, and the EMA decay was 0.9999.

The conditional UNet received a three-channel input composed of the masked baseline image, the lesion mask, and a Gaussian-noise channel. It predicted a single-channel diffusion noise term ε. The model used 128 base channels, two residual blocks per resolution, channel multipliers of (1, 1, 2, 2, 4, 4), and an attention resolution corresponding to a downsampling factor of 16. Dropout was set to zero, learned variance prediction was disabled, and the model contained approximately 113.67 million parameters.

The diffusion process consisted of 1,000 steps with a linear beta schedule increasing from 0.0001 to 0.02. The model was trained to predict ε using an MSE objective with fixed-small reverse-process variance and no timestep rescaling. Generation used the full 1,000-step DDPM reverse process implemented by p_sample_loop_known; DDIM sampling was not used. Denoised predictions were clipped by setting clip_denoised=True.

The checkpoint had been trained with a batch size of 8, a learning rate of 1 × 10⁻⁴, automatic mixed precision, and a random seed of 42. Although the configured training horizon was 2,850,000 steps, the samples used in the present study were generated from the checkpoint saved at step 122,000. During training, the synthetic_healthy mask mode generated artificial holes by translating, rotating, and flipping tumor-shaped masks and placing them in non-tumor brain regions. The actual tumor was hidden from the conditioning context by setting hide_tumor_in_context=True. The tumor-region loss weight was set to 0, whereas the synthetic-hole and surrounding-context loss weights were set to 1.0 and 0.05, respectively.

At inference time, each grayscale MRI image was first clipped to its 0.1st and 99.9th intensity percentiles and then min–max normalized to the interval [0, 1]. A region of interest was defined around the bounding box of the actual tumor mask. Its side length was determined as

$$
\max\!\left(
224,\;
h_{\mathrm{bbox}}+32,\;
w_{\mathrm{bbox}}+32
\right)
$$

corresponding to a 16-pixel context margin around each side of the bounding box. The ROI was resized to 224 × 224 using bilinear interpolation, and the mask was resized using nearest-neighbor interpolation.

The final application mask was dilated using dilation_size=9, corresponding to a 9 × 9 maximum filter and an expansion of up to four pixels in each direction. This definition differs from the breast-ultrasound setting: dilation_px=9 in the ultrasound pipeline denotes a 19 × 19 filter with a nine-pixel radius, whereas dilation_size=9 in the MRI pipeline denotes a 9 × 9 filter with a four-pixel radius.

The generated prediction was smoothed using a Pillow Gaussian blur with a radius of 1.075 and resized back to the original ROI size using bilinear interpolation. Two output forms were produced. The raw output replaced the entire ROI with the generated prediction, whereas the blended output replaced pixels only inside the dilated lesion mask using a hard binary boundary. The 3,064 target-suppressed images included in the AUROC and FPR evaluation CSV were taken from the blended outputs.

For an image at sorted dataset index i, the random seed was set to 42 + i. The data were partitioned into three shards, with shard r processing indices r, r + 3, r + 6, …. According to the generation logs, shard 0 was processed on an NVIDIA RTX PRO 5000 GPU, whereas shards 1 and 2 were processed on NVIDIA RTX A6000 GPUs.

### Chest CT

For the chest CT dataset, I⁻ᵏ was constructed by directly corrupting the target anatomical region with additive Gaussian noise. No generative inpainting model, pixel replacement, or alpha blending was used. The original 512 × 512 grayscale JPEG image was first replicated across three channels to obtain an RGB representation. For each pixel p inside the mask of target structure k and each RGB channel c, noise was independently sampled as

$$
\epsilon_{p,c}
\sim
\mathcal{N}\!\left(0,50^2\right)
$$

and the target-suppressed pixel was computed as

$$
I^{-k}_{p,c}
=
\operatorname{clip}\!\left(
I_{p,c}+\epsilon_{p,c},
0,255
\right)
$$

Pixels outside the selected target mask were not explicitly perturbed. The output was converted to 8-bit unsigned integer format after clipping.

The target regions were specified by color-coded masks: blue for the lungs, green for the heart, and red for the bronchi. No mask dilation, erosion, feathering, or boundary smoothing was applied. Consequently, the transformation used an effectively hard mask boundary. A base seed of 20250728 was used. For test image index i, the seeds were defined as

$$
\begin{aligned}
s_{i,\mathrm{bronchus}} &= 20250728 + 10i,\\
s_{i,\mathrm{heart}} &= 20250728 + 10i + 1,\\
s_{i,\mathrm{lung}} &= 20250728 + 10i + 2.
\end{aligned}
$$

Thus, independently sampled noise was assigned to every image–structure combination. The resulting images were stored as RGB JPEG files with quality 95 and 4:2:0 chroma subsampling.

We additionally verified the implemented transformation using 100 paired original and target-suppressed images. Within the target masks, the residuals had an empirical mean close to zero, a standard deviation close to 50, and a variance close to 2,500, consistent with the intended additive Gaussian model. The regression slopes between the original and modified pixel intensities ranged from approximately 0.85 to 0.97. The deviation from the ideal additive-model slope of 1 was consistent with clipping at the valid 8-bit intensity limits, particularly in low-intensity regions.

Neighboring residual pixels exhibited weak correlations of approximately 0.05–0.15, supporting the use of spatially independent white noise rather than blurred or spatially correlated noise. Correlations between noise patterns from different images were approximately zero, indicating that noise patterns were not reused. Although noise was independently sampled for each RGB channel before saving, the decoded JPEG files exhibited inter-channel residual correlations of approximately 0.80–0.84 because of chroma subsampling and compression.

Changes outside the target masks were minimal. Using an absolute intensity-difference threshold of 20, the proportions of changed outside-mask pixels were approximately 0.3%, 0.5%, and 2.5% for the lung-, heart-, and bronchus-corrupted images, respectively. These changes were concentrated within approximately four pixels of mask boundaries and were attributable to JPEG block-transform and chroma-subsampling artifacts rather than to intentional noise application.

### Natural images

The current documentation identifies the released COD10K and MAS3K datasets but does not specify the procedure used to transform an original image I into its no-object counterpart I⁻ᵏ. A reproducible manuscript description therefore cannot yet be written for this domain from the available record alone. The final appendix should report the editing or inpainting method, target-mask preprocessing, spatial resizing, replacement or blending rule, random seed, and output encoding used to generate the COD10K and MAS3K no-object images.

## 2. Construction of Positive and Target-Suppressed Captions T and T⁻ᵏ

Let T denote the positive or base caption associated with the original image I, and let T⁻ᵏ denote the corresponding caption after suppressing the semantic information associated with target k. The caption-generation strategy was selected according to the available labels and the intended evaluation protocol. Most classwise evaluation captions were deterministic class-level templates rather than image-specific free-form descriptions.

### Breast ultrasonography

#### BUS-CoT DPO caption pairs

For BUS-CoT DPO training, GPT was used to generate 100 distinct English base captions that were neutral with respect to lesion status. Each base caption begins with “Breast ultrasound image” and describes only general breast tissue, parenchyma, background echotexture, tissue composition, sonographic appearance, or anatomy. The 100 generated captions were assigned cyclically. The positive slot was taken directly from the BUS-CoT metadata, whereas the same base caption was retained for the paired negative caption. Thus, each pair had the form “[positive metadata] {base caption}” and “[negative slot] {same base caption}”.

The GPT prompt used to generate the base captions was:

> Generate exactly 100 diverse, one-sentence English base captions for breast ultrasound images.
>
> Each caption must begin with “Breast ultrasound image” and describe only general breast tissue, parenchyma, background echotexture, tissue composition, sonographic appearance, or anatomy.
>
> The captions will be shared by both positive and negative DPO pairs. Therefore, do not mention any lesion, mass, tumor, cyst, malignancy, diagnosis, pathology, BI-RADS category, measurement, location, or patient information.
>
> Use neutral professional wording. End every caption with a period.
>
> Return only the 100 captions, one caption per line, without numbering, bullet points, quotation marks, or explanations.

GPT was also used to generate 50 distinct short negative slots, each expressing a normal, unremarkable, preserved, or lesion-absent breast ultrasound finding. These slots were inserted inside square brackets before the shared base caption and were assigned uniformly at random across the training/validation pairs.

The GPT prompt used to generate the negative slots was:

> Generate exactly 50 distinct short English negative slot phrases for breast-ultrasound DPO caption pairs.
>
> Each phrase will be inserted verbatim inside square brackets before a shared generic breast-ultrasound base caption. Produce only the slot phrase itself; do not include square brackets, numbers, bullets, quotation marks, or explanations.
>
> Each slot should express a normal, unremarkable, preserved, or lesion-absent breast ultrasound finding. Use concise clinical fragments rather than full sentences. Use lower case except for standard notation such as “BI-RADS 1 negative”.
>
> Include varied wording based on expressions such as no suspicious finding, no focal lesion, normal-appearing tissue, preserved tissue/parenchymal appearance, unremarkable sonographic appearance, and negative ultrasound finding. Do not name a positive lesion or pathology.
>
> Return exactly 50 phrases, one per line.

For the UDIAT dataset, class labels were derived from the original Benign and Malignant directories and from the directory containing the generated lesion-suppressed images. These sources were mapped to the labels benign, malignant, and normal, respectively. The evaluation table contained 326 rows: 163 original tumor images and 163 generated target-suppressed images.

Each class was assigned one fixed caption:

- benign → “There is a benign tumor.”
- malignant → “There is malignant tumor.”
- normal → “There is no visible tumor.”

These strings were directly defined by the CLASSES constant in a1_inference/build_udiat_classwise_eval_csv.py. No language model was queried during construction of this classwise evaluation CSV. The current documentation does not separately specify the caption-construction procedure used for BUS-CoT.

The prompt used to specify the fixed template was:

> Generate fixed English captions for a three-class breast-ultrasound evaluation.
Return exactly three lines only, in this order: benign, malignant, normal. Do not print labels, numbering, quotation marks, or explanations.
For benign and malignant, use the literal form “There is [class] tumor.”, replacing [class] only with benign or malignant. Do not add, remove, or grammar-correct any article.
For normal, output “There is no visible tumor.”
> 

### Brain MRI

For the Figshare brain MRI evaluation set, the tumor type of each of the 3,064 original images was obtained from the label_name field in data/Figshare/test_figshare3064/figshare_tumor_type_metadata0.json. The 3,064 DDPM-generated blended images with matching identifiers were assigned the class normal, resulting in a total of 6,128 evaluation rows.

One fixed caption was assigned to each class:

- meningioma → “There is a meningioma tumor.”
- glioma → “There is a glioma tumor.”
- pituitary → “There is a pituitary tumor.”
- normal → “There is no visible tumor.”

The four strings were defined in the CAPTIONS dictionary of a1_inference/build_figshare_classwise_eval_csv.py. Tumor-type metadata were used only to select the corresponding fixed template; no sequence type, tumor location, grade, enhancement pattern, or other image-specific finding was incorporated into the caption.

The prompt used to specify the fixed template was:

> Generate fixed English captions for a four-class brain-MRI evaluation.
Return exactly four lines only, in this order: meningioma, glioma, pituitary, normal. Do not print labels, numbering, quotation marks, or explanations.
For each tumor class, use exactly “There is a [class] tumor.” and replace [class] only with the supplied class name.
For normal, output exactly “There is no visible tumor.”
> 

### Chest CT

The Chest CT AUROC/FPR evaluation used a deterministic four-class caption set, defined directly in the CLASS_SPECS constant of a1_inference/build_chest_ct_classwise_eval_csv.py. The evaluation CSV contained 800 rows: 200 images for each of bronchus noise, heart noise, lung noise, and normal. The fixed captions were:

- bronchus noise → “Chest CT image showing preserved cardiac contour and normal-appearing bilateral lung parenchyma.”
- heart noise → “Chest CT image showing preserved bilateral lung aeration and patent central airways.”
- lung noise → “Chest CT image showing preserved cardiac contour and patent central tracheobronchial structures.”
- normal → “Chest CT image showing preserved cardiac contour, preserved bilateral lung aeration, and patent central airways.”

The normal caption serves as the all-structure baseline T. For each regional-noise class, the target-suppressed caption T⁻ᵏ describes only the two structures left uncorrupted by Gaussian noise and omits the corrupted target structure. The same deterministic captioning rule was used when constructing the Chest CT DPO training data; no separate prompt or caption-generation procedure was introduced for the AUROC/FPR evaluation. No per-image caption generation, random sampling, or additional language-model inference was used in this classwise evaluation.

### Natural images

For COD10K, the original class labels were obtained from the 69 camouflage subclasses provided in the test metadata. A further no camouflaged animal class was assigned to the generated no-object images, resulting in 70 evaluation classes. For MAS3K, the 33 original object classes were supplemented with the same no-object class, resulting in 34 evaluation classes.

For an original image containing an annotated camouflaged object, the caption was deterministically constructed as

> “There is an [class name] present in the scene.”
> 

where [class name] was replaced by the corresponding dataset class. The article an was intentionally retained for every class and was not grammatically adjusted according to the initial sound of the class name. For a generated no-object image, the fixed target-suppressed caption was

> “There is no camouflaged animal present in the scene.”
> 

No habitat, color, size, background description, or image-specific attribute was appended. The same deterministic template rule was used for the COD10K and MAS3K DPO training-data construction and was reused for classwise AUROC/FPR evaluation; no separate prompt or caption-generation procedure was introduced for either dataset.

The COD10K evaluation file contained 2,026 original images and 2,026 no-object images, yielding 4,052 rows across 70 classes. The corresponding file was

data/COD10K/test/cod10k_original2026_no_object2026_subclass70_scene_caption_pairs_paths_fixed.csv.

The MAS3K evaluation file contained 582 original images and 582 no-object images, yielding 1,164 rows across 34 classes. The corresponding file was

data/MAS3K/mas3k_original582_no_object582_classwise_scene_caption_pairs_paths_fixed.csv
