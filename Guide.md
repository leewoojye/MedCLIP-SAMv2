Biomedclip direction loss를 통한 음성 샘플 생성 
- zero_shot_translation/vector_injection.py
- run_generate_neg_udiat_vector_injection.sh

요청하신 대로 UDIAT 데이터셋의 원본 이미지와 마스크 이미지를 vector_injection.py 스크립트를 통해 처리하는 쉘 파일(run_generate_neg_udiat_vector_injection.sh)을 생성해 드렸습니다.

생성된 파일 경로: /home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/run_generate_neg_udiat_vector_injection.sh
구조 특성상 UDIAT 하층에 있는 Benign(양성), Malignant(악성) 폴더와 각각의 짝이 되는 Benign_mask, Malignant_mask 폴더를 모두 순회하며 처리하도록 작성했습니다.
결과 저장 위치: /home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/generated_neg_output/udiat_vector_injection/ 밑에 자동으로 저장됩니다.
실행 권한(chmod +x)도 이미 부여해 두었으므로 바로 터미널에서 실행하시면 됩니다.
터미널에서 아래 명령어로 바로 실행하실 수 있습니다.

bash
./run_generate_neg_udiat_vector_injection.sh

(참고: vector_injection.py 내부 로직에 하드코딩된 Vector Anchor 단어 구문(healthy breast, tumor breast, tumor brain)과 --alpha 값(기본 1.0)은 추후 결과물의 퀄리티에 따라 해당 파이썬 스크립트나 쉘 스크립트 내에서 별도로 튜닝하실 수 있습니다.)