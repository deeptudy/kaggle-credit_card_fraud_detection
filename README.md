# kaggle-credit_card_fraud_detection
Kaggle, Credit card fraud detection 2023, Binary classification

# 1. Intro

- 데이터의 선택 이유
    - 데이터 양이 많고, 비교적 clean한 데이터 셋을 선정함.
        - 55만 개 + rows, 30 + features
- 진행 방향
    - 데이터 특징 파악 및 간단한 시각화를 이용한 EDA 및 전처리 진행
    - Scratch부터 시작하는 것을 가정하고, 일단 최대한 간단하고 빠르게 모델 학습 & 평가 진행
        - 기존에 만들어 둔 Pipeline에 데이터셋을 넣는 것이 아닌, 기존 자료, 구글링 등이 없다고 가정하고 빠르게 1회차 학습/평가를 돌리는 것이 우선이었습니다.
    - 이후, 성능 향상을 위해 많은 실험을 돌릴 수 있는 Pipeline 구축하기
- 프로젝트의 목적
    - 1) 거래 내역을 보고 카드 사기인지 아닌지를 판단할 수 있는 모델 만들기
        - Player 1 집중
    - 2) 어떠한 특성을 가진 거래 내역이 카드 사기일 확률이 높을 지 (다만 특성이 익명처리되어서 직접적인 해석은 어려움)
- 실행 계획
    - 1) Data Exploration
    - 2) 데이터 전처리 (Scaling, Train/Test Split, etc.)
    - 3) ANN 모델 학습/평가
    - 4) 머신러닝 모델 학습/평가

# 2. Dataset 설명

- Total: 31 columns
    
    
    | no. | column name | description | type | note |
    | --- | --- | --- | --- | --- |
    | 0 | id | 각 거래를 구분해주는 고유값 | int |  |
    | 1 | V1 ~ V28 | 익명처리된 거래 관련 특성들 (거래 발생 시간, 위치 등) | float |  |
    | 2 | Amount | 거래 금액 | float |  |
    | 3 | Class | 카드 사기인지(1) 아닌지(0) | int |  |
    - 데이터 소개 원본 from kaggle
        - **id**: Unique identifier for each transaction
        - **V1-V28**: Anonymized features representing various transaction attributes (e.g., time, location, etc.)
        - **Amount**: The transaction amount
        - **Class**: Binary label indicating whether the transaction is fraudulent (1) or not (0)

# 3. Outro

<aside>
💡 Player 1의 코멘트
: 정돈되어 있고 (normally distributed) 따로 전처리가 필요하지 않은 깨끗한 데이터였습니다. 
실험 또한, epochs = 10번 이내로 0.999 Accuracy가 나와서 간단하게 ANN을 사용한 결과를 제시합니다. (다른 kaggle 참여자도 비슷한 결과를 보여줌)
따라서, 최소한의 모듈화만 진행하였고, kaggle에 올릴 수 있도록  eda부터 모델 성능 결과까지 하나의 파일에서 볼 수 있는 ipynb를 추가하였습니다.
</aside>

### 1) 데이터 살펴보기

- 기초 통계량, 데이터 타입
    - Anonymized(익명 처리), normalized(정규화)된 데이터
    - id와 class를 제외하고는 전부 numerical한 데이터였음.
- 클래스의 분포 확인 → 불균형 문제 없음.
    
    
    | Class | Count |
    | --- | --- |
    | 0 (Not a fraud) | 284,315 |
    | 1 (Fraud) | 284,315 |
- 결측값, 중복값 → 없음.
- 이상치(Outliers)
    - lower, upper bound 기준을 넉넉히 잡았음에도 불구하고, IQR 기준 outliers가 많다. 약 13%
    - 확인해보면, 평균값이 1 미만인 column에서 max가 100이 넘는 column들도 존재한다.
    - 다만, card fraud 특성 상, 오히려 fraud = true 인 케이스들의 값이 유독 크거나 작을 수 있음. 따라서, 무작정 outliers를 제거하거나 처리하는 것 보다는 일단 포함한채로 학습을 진행해본다.
    - Neural network model을 메인으로 쓸 것이기 때문에, 어느 정도 outliers 패턴을 잡아줄 것이라는 기대도 있고, 그렇지 않다면 robust model들을 쓰는 방법도 있음.

### 2) 데이터 시각화 (Visualization)

- Amount
    - min: 50.35, max: 24039.93, 이 range 안에서는 골고루 분포하고 있음.
    - 신용카드 사용량이기 때문에, 적게 사용하는 사람과 많이 사용하는 사람이 모두 골고루 있는게 자연스러움. 
    - Normally distributed

![image](https://github.com/deeptudy/kaggle-credit_card_fraud_detection/assets/76639910/bae7441b-de1e-4cfc-8d32-b294dd0eebb3)

- Correlation
    - class에 대해서 correlation이 0.5이상인 변수는
        - id, v1, v3, v4, v9, v10, v11, v12, v14, v16
![image](https://github.com/deeptudy/kaggle-credit_card_fraud_detection/assets/76639910/d347767d-a4f4-49a3-9b55-11756f7c5f1b)

### 3) ANN 모델 학습 및 평가

- StandardScaler 사용
- batch size = 128
- hidden dimension = 64
- dropout ratio = 0.3
- activation = Relu
    - return with Sigmoid(x)
- optimizer = Adam
- Loss = Binary Cross Entropy
- learning rate = 0.001
- epoch = 10

![image](https://github.com/deeptudy/kaggle-credit_card_fraud_detection/assets/76639910/a669a43d-c13a-4472-ad54-9d3ce2342171)

- Confusion matrix
  
![image](https://github.com/deeptudy/kaggle-credit_card_fraud_detection/assets/76639910/2d9a90f2-a98e-4b5c-98f2-8dce8114d89e)

# 4. Links

- Kaggle
    - [Credit Card Fraud Detection Data](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/data)
