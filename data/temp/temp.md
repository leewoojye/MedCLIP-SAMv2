- 논문 측에서 사전 분할 X, 임의 8:2 train:test 분할
- https://huggingface.co/datasets/WOOJYE/gmpo_mas3k/tree/main
- 데이터 원천: UOVSBench는 기존 수중 분할 데이터셋 MAS3K의 이미지ㆍ마스크 쌍과 범주 라벨을 open-vocabulary segmentation 형식으로 정리하여 포함한 벤치마크이며, 본 연구에서는 그 공개 MAS3K 구성의 원본 수중생물 이미지와 클래스 정보를 사용하였다. AquaOV255 및 UOVSBench의 다른 5개 데이터셋, Earth2Ocean 방법, 그리고 UOVSBench의 분할ㆍ평가 프로토콜은 사용하지 않았다. 현재 프로젝트의 MAS3K 원본 2,910장 가운데 582장(20%)을 연구 내에서 임의로 테스트셋으로 분할하였으며, 따라서 이 582장ㆍ33개 관측 클래스의 테스트 분할은 UOVSBench가 제공한 공식 분할이 아니다. 참고: https://arxiv.org/abs/2511.07923

- English: No author-provided train--test split was used; we created a random 8:2 train--test split. UOVSBench incorporates the image--mask pairs and category labels of the existing underwater segmentation dataset MAS3K after converting them to an open-vocabulary segmentation format. We used only the original underwater-animal images and class information from this released MAS3K component. We did not use AquaOV255, the other five UOVSBench dataset components, the Earth2Ocean method, or the UOVSBench split and evaluation protocol. From the 2,910 MAS3K original images in this project, we randomly selected 582 images (20%) as the test set; therefore, this test split, containing 33 observed classes, is not an official UOVSBench-provided split. Reference: https://arxiv.org/abs/2511.07923

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