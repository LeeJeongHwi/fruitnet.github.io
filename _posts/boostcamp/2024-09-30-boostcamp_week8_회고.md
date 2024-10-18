---
layout: post
title: Boostcamp AI Tech 7기 Week8
description: >
  Naver Boostcamp AI Tech 7기 Week8 회고록
sitemap: false
categories: [boostcamp]
hide_last_modified: true
related_posts:
  - ./2024-09-02-boostcamp_week5_회고.md
  - ./2024-09-10-boostcamp_week6_회고.md
  - ./2024-09-23-boostcamp_week7_회고.md
---

# Naver Boostcamp AI Tech 7기 Week 8 회고록

## 2024-09-30

### MRC는 무엇인가

Machine Reading Comprehension (기계독해) 기술을 일컫음

흔히 우리가 인터넷에 검색하고 그에 대한 결과를 받는 “검색엔진”을 말함

즉 **Question & Answering (QA) 문제**라고도 할 수 있다.

### ODQA(Open-Domain Question Answering)

QA 문제 중에서 Open-Domain이 추가된 시스템을 말한다.

지문이 따로 주어지지 않은 질의에 대해서 지식베이스 기반으로 답변을 찾아내는 Task

사전 구축된 Knowledge Resource 구축이 필요

- 본 과제에서는 현재 주어진 데이터는 Wikipedia Documents

본 과제의 Baseline은 2-Stage로 구성되어 있음

1. 질문에 관련된 문서를 찾아주는 **Retriever** 단계
2. 관련된 문서를 읽고 적절한 답변을 찾거나 Generation하는 **Reader** 단계

또한, 이번 Baseline은 Extraction-based MRC로 구성되어 있다.

Extraction-based MRC는 start point와 end point를 기반으로 context의 단어를 찾아내는 것을 목적으로 하고

Generation-based MRC는 말 그대로 GPT처럼 문장을 생성해내는 것을 목적으로 한다.

## 2024-10-01, 10-02

배운내용 : Sparse Embedding, Dense Embedding

Sparse Embedding은 Passage Retrieve 단계에서 사용되는데, BoW를 구성하는 n-gram 방식이다.

Term(단어, 토큰)이 Context에서 등장하는지, 몇번 등장하는지를 embedding vector로 구성한다.

근데 Sparse Embedding의 경우 등장하는 모든 단어들에 대해 처리하므로 Embedding Vector의 Dimension은 단어 갯수에 따라 증가하게 된다.

보통 TF-IDF(Term Frquency-Inverse Document Frequency)로 처리하는데, 단어의 등장 빈도와 제공하는 정보의 양에 따라 확률을 만들어 낸다.

$$IDF(t) = \log{\frac{N}{DF(t)}}$$

자주 등장하는 is나 the 같은 단어들은 사실상 의미가 없기 때문에, TF가 커도 IDF score가 0이 된다.

기본적으로 "공백"으로 단어를 분리

Dense Embedding은 말 그대로 Dense Layer를 사용해서 Embedding을 하는 것이다.

일반적으로 BERT계열의 모델(`AutoModelForSequenceClassification`)을 사용한다

Dim이 Sparse보다 더 작기 때문에 이렇게 활용이 가능하다

Sparse보다 장점은 단어의 유사성과 맥락 파악에서 장점을 보인다.

## 2024-10-10

이번 프로젝트에서의 Baseline 코드 분석을 했는데, 생각보다 더 어렵다.

우선은 Retrieval 단계와 Reader 단계를 확실히 분리해서 생각해야한다.

Reader 단계에서는 QA 모델을 사용해서 주어진 Dataset을 학습하고 Context에서 Start Point를 잘 찾을 수 있도록 하게 한다.

Retrieval은 현재 Inference 단계에서 사용되는데, Query(Qeustion)에 맞는 Context를 Top-K개 만큼 추출해서 Dataset을 만들어주는 역할을 한다.

추가로, 일단 다른 팀원들의 데이터 분석을 통해 확인해본 결과 TF-IDF의 성능이 매우 낮다는 것을 판단

따라서 Dense Embedding을 사용해서 Baseline 코드 만드는 것이 필요

데이터셋은 내생각엔 KorQuad랑 AiHub정도만 학습시켜도 괜찮지 않을까란 생각이 든다.

현재 주어진 시간에 AIHub까지 학습시킨다면... 얼마나 많은시간이 소요될지 감당되지 않는다.

그리고 만약에 사용한다면 과연 유의미한지도 TF-IDF로도 먼저 확인해야하는게 아닐까?란 생각이 든다.

