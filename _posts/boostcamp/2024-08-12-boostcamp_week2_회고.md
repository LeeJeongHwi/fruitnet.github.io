---
layout: post
title: Boostcamp AI Tech 7기 Week2
description: >
  Naver Boostcamp AI Tech 7기 Week2 회고록
sitemap: false
categories: [boostcamp]
hide_last_modified: true
related_posts:
  - ./2024-07-24-boostcamp_지원후기.md
  - ./2024-08-05-boostcamp_week1_회고.md

---

# Naver Boostcamp AI Tech 7기 Week 2 회고록

### 2024-08-12 회고

Numerical Instability in Softmax

우선 softmax의 함수는 다음과 같다.

$p_i = \frac{e^{z_i}}{\sum_{j=1}^k e^{z_j}}, \quad \text{i = class}$

* 여기서 exponential한 함수이기 때문에 매우 큰 값이 될 수 있고, 그 큰 값으로 나누게 될 수도 있다.
* 즉, 모든 클래스가 동일한 확률을 가지게 될 수도 있다는 것이다. (Unstable)

그래서 이를 방지하고자, 상수 $C$ 를 분모와 분자에 곱해서 shift하고, 이를 mapping하는 방법으로 문제를 해결한다.

여기서 $C$를 곱해주는 방법은 $\log{C}$로 곱해주는데, 이는 $-\max(x)$로 표현이 가능하고, 결론적으로는 입력값의 최댓값을 빼준다.

> 근데 왜 Log C가 -max(x)인지에 대해서는 잘 모르겠다.

```python
import numpy as np
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

x = np.array([1.2, 2000, -4000, 0.0])
softmax(x)
# >> 0, nan, 0, 0

def modified_softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

x = np.array([1.2, 2000, -4000, 0.0])
modified_softmax(x)
# >> 0, 1, 0, 0
```

> Reference
> https://jaykmody.com/blog/stable-softmax/
> https://dev-jm.tistory.com/30

**선형회귀의 가정에 대한 이야기**

이번 강의에서 선형회귀의 가정에 대해서 배웠는데, 좀 더 추가적으로 내 자신이 이해하기위한 내용을 작성하려고한다.

우선 선형회귀의 목적을 먼저 말하면, 알려진 데이터를 기반으로 알려지지 않은 데이터에 대한 값을 예측하기 위함이다. 예측하기 위해서 종속변수와 독립변수간의 선형관계에 대해 모델링이 필요하다.

근데 여기서 선형회귀모델을 정의할 수 있는 몇 가지 가정이 필요하다.

* 선형성
  * 독립변수와 종속변수간의 선형성 존재
    * 어찌보면 가장 당연한 이야기다. 선형성이 존재하지 않으면 의미가 없다.
* 독립성
  * 독립변수 간의 독립성
    * 다중회귀분석의 경우처럼 독립변수가 여러 개인 경우, A라는 독립변수가 B라는 독립변수와 상관관계가 존재한다면 종속변수와의 상관관계를 정확히 파악하기 어려워 모델링이 제대로 되지 않는 문제점이 있기 때문 (다중공선성)
  * 잔차 간의 독립성
    * 잔차간의 독립성은 단순하다. 잔차는 자체가 무작위성을 가지고 있다. 만약에 이전 잔차가 현재의 잔차에 영향을 미치면, 잔차에 대한 자기상관을 갖게되므로 모델링이 되지 않는다.
* 등분산성
  * 잔차의 분산들은 일정해야함
  * 이분산성 : 오차의 분산이 독립 변수 값에 영향을 받아서 변하는 경우를 의미함, 즉 독립변수의 값에 관계없이 오차의 크기가 일정해야함
* 정규성
  * 잔차들이 정규분포를 따라야함
    * 위 등분산성과 비슷한 이야기임
    * 신뢰구간 추정에 어려움이 있음 (신뢰성의 문제가 있음)



### 2024-08-13 회고



