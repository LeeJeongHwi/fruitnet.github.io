---
layout: post
title: Boostcamp AI Tech 7기 Week3
description: >
  Naver Boostcamp AI Tech 7기 Week3 회고록
sitemap: false
categories: [boostcamp]
hide_last_modified: true
related_posts:
  - ./2024-08-05-boostcamp_week1_회고.md
  - ./2024-08-12-boostcamp_week2_회고.md


---

# Naver Boostcamp AI Tech 7기 Week 3 회고록

### 2024-08-20 회고

>  Transformer 논문을 리뷰 완료

이전에 프로젝트들을 하면서 대부분 다 Monitoring System 위주로 만들어서 Model 결과를 시각화하는 것으로 만들었었다.

하지만 아직까지도 결과에 따른 chart 유형에 대해서 잘 몰랐고.. 색도 그만큼 중요하다라는 것을 잘 몰랐다. (그래서 예전에 아무런 색으로 했더니 가시화가 그렇게 잘안된거였다..)

이전에 했던 프로젝트들 (토이플젝 포함)

1. 서울 지역별 따릉이 보유 현황(지도맵 활용)
2. 전력 데이터 실제 사용량 예측량 비교
3. Pose Skeleton 각도 변화량에 따른 경고
4. (학교 과제) 포켓몬.. 능력치..?ㅋㅋ
5. Attention Score에 따른 ECG 구간 highlighting

그리고 하면서 느낀거지만... 시각화 코드를 만들어 내는 것도 어렵다.

예를 들어 4번 Attention Score를 가장 Score가 높은 지점에만 Highlighting 하고 싶었는데, Transpose하고.. interpolation도 하고... 하다보니 쉽지않았다.

이 기회에 어떻게 개선해야될지 간략하게나마 정리해보면 좋을 것 같다.



### 2024-08-21 회고

