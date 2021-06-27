---
layout: post
title: "[Paper review] Session-based Recommendation with Graph Neural Networks"
date: 2021-06-20 20:20:20 +0900
category: sample
---

## 초보자를 위한 쉽게 보는 논문 리뷰 - 1편.

## Abstract
- anonymous user에 대한 추천 예측이 어려움
- 이전의 추천 방식:session을 sequence로 모델링함item representation 외에도 user representation을 진행함
    - 이는 정확한 유저 벡터를 표현하기에는 insufficient
    - 충분한 유저-아이템 interaction에 의존함
- 정확한 item embedding과 complex transitions of items(복잡한 아이템 전환?) 을 고려하기 위해서 새로운 방식을 고안함
- 새로운 추천 방식
    - session sequences를 graphstructured data로 표현
    - session graph를 통해 GNN은 complex transitions of items을 고려가능
    - 각각의 세션은 attention network를 통해 composition of global preference와 해당 세션의 current interest로 표현됨

## Introduction

### 1.1 Session-based Recommendation

- 기존 추천 시스템은 대부분의 유저 프로필과 과거 행동 데이터들이 끊임없이 저장된다고 가정한다.
- 하지만 실제 서비스에서는 유저의 신원을 모를 수도 있고, 진행 중인 세션 내의 유저 행동 기록만을 사용해야 할 수도⇒ "따라서 한 세션에서 제한된 행동 기록들을 모델링하고 그에 맞춰 추천을 생성해야한다."

**기존 모델**

- 이전의 RNN 계열의 모델에서는 유저의 선호도를 반영하는 user representation이 따로 존재하지 않고, RNN의 hidden vector를 user representation으로 삼고 다음 아이템을 예측해 왔다.그러나 각 세션은 user-specific하지 않고, 세션 클릭과 관련된 유저의 행동은 대부분 한정되어 있다.⇒ 각 세션에서 user representation을 정확하게 추정하는 것은 어렵다.
- 기존 모델들은 연속적으로 선택되는 아이템들의 single-way Transition을 모델링하고, 세션 내의 다른 여러 아이템들 간의 복잡한 Trainstion은 무시한다.⇒ 멀리 떨어진 아이템들 사이의 복잡한 Transition은 간과되어 왔다.

### 1.2 SR-GNN (Session-based Recommendation with Graph Neural Networks)

> Graph Neural Networks를 사용한 Session-based 추천시스템기존 모델들의 단점들을 보완하기 위해 본 논문에서 제안된 모델

- Graph Neural Networks를 통해 아이템 간의 관계를 파악하고, 아이템 임베딩에 대해 정확한 값을 만들어낸다이는 Markov Chain & RNN 기반 모델 같은 전통적인 시퀀셜 모델에서는 구현되기 어렵다.
- 아이템 임베딩에 대한 정확한 값들을 기반으로 global preference와 current interest를 구하여 예측에 사용한다.
- User-specific 하지는 않지만, 해당 세션의 시퀀스가 반영된 임베딩이 들어가기 때문에 session-specific하게예측할 수 있다.

- 이전의 추천 시스템은 유저의 프로필과 past activities가 계속 저장된다고 예상함
- 그러나 실제 서비스에서는 anonymous user이거나 진행중인 세션 내의 유저 행동 기록만 사용가능
- **한 세션에서 제한된 행동 기록을 모델링하고 이에 맞는 추천을 생성해야함**
- 결국 GNN의 가장 큰 장점은 다음과 같다
    - capture transition of item
    - generate accurate item embedding vectors
    - 위의 두 개는 전통적인 sequential method(Marcov Chain, RNN-based model)에서는 발견하기 어려움

# 2. Related Works

### 2.1 Conventional recommendation methods

### 1) Matrix Factorization

Target 값(implicit, explicit)을 저차원 아이템 임베딩 행렬과임베딩 행렬로 분해하는 방법론

- 세션 기반 추천에서는 적절하지 못하다.→ 왜냐하면 유저 선호도 일부가 positive 클릭에 의해서만 제공되며, 아이템 시퀀셜 순서를 고려하는데 어려움이 있고, 단지 마지막 클릭을 바탕으로 예측을 수행하기 때문

### 2.2 Deep learning-based recommendation methods

### 1) Improved recurrent neural networks for session-based recommendations

: 유저 행동의 시간적 변화를 고려하고 적절한 data augmentation기법을사용하여 순환 모델의 성능을 높임.

### 2) When recurrent neural networks meet the neighborhood for session-based recommendation

: 시퀀셜 패턴과 동시에 발생 신호를 혼합하기 위해 이웃 기반 기법과 순환기법을 결합함.

### 3) 3D convolutional networks for session-based recommendation with content features

: 추천을 수행하기 위해 3c CNN을 사용하여 아이템 카테고리 & 설명 같은컨텐츠 features와 세션 클릭을 통합함.

### 4) List-wise DNN

: 각 세션 내 제한된 유저 행동을 모델링하고, 각 세션에 대한 추천을 수행하기 위해 List-wise 랭킹 모델을 사용함.

### 5) NARM(A Neural Attentive Recommendation Machine with an encoder-decoder architecture)

: 유저의 시퀀셜 행동의 features와 주요 목적을 포착하기 위해 RNN 어텐션 메커니즘을 사용함.

### 6) STAMP(Short-Term Attention Priority model)

: 유저의 일반적인 흥미와 현재 흥미를 효율적으로 포착하기 위해 제시됨.

### 2.3 Graph Neural Networks

### 1) Graph Neural Nerworks

- 그래프 구조에서 사용하는 인공 경망이다.
- 인공 신경망들을 보통 벡터나 행렬 형태로 input이 주어지는 데 반해서 GNN의 경우 input이 그래프 구조라는 특징이 있다.
- 관계, 상호작용과 같은 추상적인 개념을 다루기에 적합하다.
- 일반적인 그래프는 𝐺 = (𝑉, 𝐸)로 정의하며 𝑉는 점 집합이고 𝐸는 두 점을 잇는 선 집합이다. 아래 그래프는 다음과 같이 정의할 수 있다.𝐺 = ( 1,2,3 ,{ 1,2 , 2,3 ,{1,3}})

    ![https://media.vlpt.us/images/99ktxx/post/78b3e56a-6ce3-4ec2-a9c7-bd4668ca3755/image.png](https://media.vlpt.us/images/99ktxx/post/78b3e56a-6ce3-4ec2-a9c7-bd4668ca3755/image.png)

### 2) Gated Graph Neural Networks

- GNN의 수정 버전으로 노드 벡터를 업데이트할 때 GRU cell을 이용한다.
- 시간 경과에 따른 역전파(BPTT, Back-Propagation Through Time)를 사용하여 Gradients를 계산한다.

# 3. The proposed Methods

### 3.1 The workflow of the proposed SR-GNN

![Session-Based%20Recommendation%20with%20Graph%20Neural%20Net%20a9b78233221a4e0f8fb0265fcf205376/Screen_Shot_2021-06-27_at_10.08.18_PM.png](Session-Based%20Recommendation%20with%20Graph%20Neural%20Net%20a9b78233221a4e0f8fb0265fcf205376/Screen_Shot_2021-06-27_at_10.08.18_PM.png)

> 세션 기반 추천은 장기 선호도 프로필 접근 없이 유저의 현재 시퀀셜 세션 데이터(current sequence session data)만을 기반으로 유저가 다음에 클릭할 아이템이 무엇인지 예측하는 것에 초점을 둔다.

### 3.2 Notations

![https://media.vlpt.us/images/99ktxx/post/50596d07-eb4d-47b7-a4de-37dbfe1b07ef/image.png](https://media.vlpt.us/images/99ktxx/post/50596d07-eb4d-47b7-a4de-37dbfe1b07ef/image.png)

# 4. Experiments and Analysis

### 4.1 Datasets

![https://media.vlpt.us/images/99ktxx/post/d5a2a572-74cb-46ea-88be-10c700d8b449/image.png](https://media.vlpt.us/images/99ktxx/post/d5a2a572-74cb-46ea-88be-10c700d8b449/image.png)

- Yoochoose, Diginetica dataset: e-commerce 웹사이트에 유저들의 클릭 기록
- 세션 길이 1짜리는 제거
- Cold-start problem 방지를 위해 최소 등장 횟수 5회로 설정
- Input 세션 *s*=[*vs*,1,*vs*,2,...,*vs*,*n*]의 경우 시퀀스와 레이블 시리즈를 다음과 같이 생성한다.

    ![https://media.vlpt.us/images/99ktxx/post/59e0afee-4693-44f6-b042-665df0c4505a/image.png](https://media.vlpt.us/images/99ktxx/post/59e0afee-4693-44f6-b042-665df0c4505a/image.png)

### 4.2 Baseline Algorithms

제시된 모델의 성능을 평가하기 위해 아래와 같은 대표적인 베이스라인들과 비교한다.

- **POP & S-POP:** 훈련집합과 현재 세션에서 각각 빈번하게 등장한 top-N개의 아이템들을 추천
- **Item-KNN:** 세션에서 이전에 클릭된 것과 유사한 아이템을 추천 (유사도는 세션 벡터 사이의 코사인 유사도로 정의)
- **BPR-MF:** SGD(Stochastic Gradient Descent)를 통해 Pairwise 랭킹 목적 함수 최적화
- **FPMC:** 마르코프 연쇄 기반 시퀀셜 예측 기법
- **GRU4REC:** 세션 기반 추천에 대해 유저 시퀀스를 모델링하는 RNN을 사용
- **NARM:** 유저의 핵심 목적과 시퀀셜 행동을 포착하는 어텐션 메커니즘 RNN을 사용
- **STAMP:** 유저의 일반적인 선호도와 현재 세션에서의 마지막 클릭을 통해 현재 관심사를 포착

### 4.3 Evaluation Metrics

- P@20 (Precision): 예측 정확도 측정치로 널리 사용됨.top-20 아이템들 중에서 정확하게 추천된 아이템의 비율을 나타냄.
- MARR@20 (Mean Reciprocal Rank): 정확하게 추천된 아이템의 역순위 평균임.역순위가 20을 초과할 때는 0으로 설정됨.MRR 측청치는 추천 랭킹 순서를 고려함.→ MRR 값이 크면 TOP 랭킨 리스트에 정확한 추천이 나타남.

# 5. Conclusions

- 세션 기반 추천은 유저 선호도와 기록 내역을 얻기 어려울 때 필수적이다.
- 본 논문은 그래프 모델로 세션 시퀀스를 표현하기 위한 새로운 아키텍처를 제시하였다.

    > SR-GNN• 세션 시퀀스 아이템 간의 복잡한 구조와 Transition을 고려할 뿐만 아니라 유저의 다음 행동을 더 잘 예측하기 위해 장기 선호도와 현재 세션에서의 관심사를 결합하여 사용하였다.• 종합적인 실험들은 제시된 알고리즘이 다른 최신 기법들보다 일관적으로 좋은 성능을 내는 것을 입증하였다.

## References

- [[Paper Review] (2019, AAAI) Session-Based Recommendation with Graph Neural Networks] ([https://velog.io/@tobigs-recsys/Session-Based-Recommendation-with-Graph-Neural-Networks#11-session-based-recommendation](https://velog.io/@tobigs-recsys/Session-Based-Recommendation-with-Graph-Neural-Networks#11-session-based-recommendation))

Usage

How to apply?

Furthermore
