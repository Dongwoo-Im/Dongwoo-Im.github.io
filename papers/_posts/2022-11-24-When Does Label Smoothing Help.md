---
layout: post
title:  "When Does Label Smoothing Help?"
date:   2022-11-24 19:00:00 +0900
categories: review
comments: true
use_math: true
sitemap :
    changefreq: daily
    priority: 1.0
---

# [논문리뷰] When Does Label Smoothing Help? (NIPS ‘19)

---

## 1. Introduction

### Motivation

Label smoothing (LS)은 [Inception-v3 논문](https://arxiv.org/abs/1512.00567)에서 처음 제안되었고, 이후 image classification, speech recognition, machine translation 등의 여러 task에서 hard target을 그대로 사용하는 것보다 효과적이라는 연구들이 있었습니다. 하지만 LS이 널리 사용되는 효과적인 trick임에도 왜 효과적인지, 어떻게 작동하는지와 같은 근본적인 연구가 부족했습니다. 이에 저자들이 연구를 시작하게 됩니다.

### Contribution

- Visualization method 제안 : penultimate layer activation에 linear projection을 활용한 방법으로, 학습된 representation 사이에서 직관적인 비교 가능

- Calibrate model : model의 prediction에 대한 confidence와 accuracy를 보다 align하게 함

- Impair distillation : teacher가 LS로 학습된 경우 hard target으로 학습했을 때 보다 distillation 성능 하락, 이는 logit 상에서 정보를 잃기 때문 (distillation에 유리한 representation을 잃어버린다(?))

### 1.1. Preliminaries

논문 이해를 위한 수학적 설명입니다.

- LS 적용의 효과를 시각적으로 보여주기 위해 $p_k$를 활용했습니다.

    - $p_k = \frac {\exp(x^{T}w_{k})} {\Sigma^{L}_{l = 1}{\exp(x^{T}w_{l})}}$

        - $p_k$ : class k에 대한 model이 할당한 likelihood

        - $w_k$ : 마지막 layer의 weights and biases

        - $x$ : 마지막에서 2번째인 layer에 (bias를 설명하기 위한) 1을 concat한 vector (weight + bias = linear(?))

- 또한, LS 적용을 위한 soft label 계산식은 다음과 같습니다.

    - $y^{LS}_{k} = y_k(1-\alpha) + \alpha/K$

        - $y_k$ : hard target

---

## 2. Penultimate layer representations

LS를 적용하면 정답 class logit과 오답 class logit 사이에 상수 $\alpha$ 만큼의 거리를 유지하도록 모델이 학습합니다. 반면, hard target을 활용하면 정답 class logit은 커지도록, 오답 class logit은 작아지도록 학습할 뿐입니다. 

---

## 3. Implicit model calibration

### Image classification

### Machine translation

---

## 4. Knowledge distillation

---

## 5. Related work

---

## 6. Conclusion and future work

---

## Reference

- [Paper](https://arxiv.org/abs/1906.02629)