---
layout: post
title:  "Barlow Twins: Self-Supervised Learning via Redundancy Reduction"
date:   2023-04-24 19:00:00 +0900
categories: review
comments: true
use_math: true
sitemap :
    changefreq: daily
    priority: 1.0
---

# [논문리뷰] Barlow Twins: Self-Supervised Learning via Redundancy Reduction (ICML '21 Spotlight)

[arXiv link](https://arxiv.org/abs/2103.03230)

[github link](https://github.com/facebookresearch/barlowtwins)

---

### Abstract

Contrastive self-supervised learning(SSL) 문제점 : collapsing -> 이 단점을 제거한 objective 제안 = Barlow Twins

Contrastive objective와의 차이점 : 뇌 과학자 Barlow의 redundancy-reduction 이론을 적용하여 임베딩 벡터 간 중복 감소

어떤 이점이 있나?

- large batch 필요 X
- predictor network와 같은 비대칭성 필요 X (predictor는 BYOL에서 적용 및 제안한 이후로 쭉 사용해온 모듈)
- stop gradient 필요 X
- EMA 필요 X

즉, SSL 학습 안정성을 높이기 위해 적용했던 여러 기법들 없이도 collapsing 현상을 막을 수 있다!

### Method

![Figure 1](https://dongwoo-im.github.io/assets/img/posts/2023-04-24-BarlowTwins/1-fig1.png){: .align-center}

![Objetive](https://dongwoo-im.github.io/assets/img/posts/2023-04-24-BarlowTwins/2-objective.png){: .align-center}

Barlow Twin's objective 설명

- batch 단위로 정규화 (mean=0, std=1) 적용
- Invariance term : cross-correlation의 대각 행렬을 1로
- Redundancy reduction term : cross-correlation의 대각 행렬이 아닌 값을 0으로

Information theory 측면에서 SSL 해석

- Information Bottleneck objective
    - minimize I(input, latent) + maximize I(latent, target)
- Augmentation (distortion)을 통해 다양한 input으로부터 robust한 latent를 학습한다.
    - minimize I(distort, latent) + maximize I(image, latent)

    ![Figure 6](https://dongwoo-im.github.io/assets/img/posts/2023-04-24-BarlowTwins/7-fig6.png){: .align-center}

Barlow Twin's objective VS infoNCE objective

- 유사한 부분
    - BT redundancy reduction term = infoNCE contrastive term

- BT의 장점
    - not require a large number of negative samples
    - benefits from very high-demensional embeddings

Barlow Twin's objective VS whitening

- BT는 soft-whitening으로 해석될 수 있다. (추후 서술)

### Results

순서대로 linear probing, fine-tune with semi-supervised learning, trasnfer on image classification, transfer on other task 성능입니다.

![Table 1](https://dongwoo-im.github.io/assets/img/posts/2023-04-24-BarlowTwins/3-tab1.png){: .align-center}

Table 1: linear probing 성능

![Table 2](https://dongwoo-im.github.io/assets/img/posts/2023-04-24-BarlowTwins/4-tab2.png){: .align-center}

Table 2: semi-supervised learning 방식으로 fine-tune

![Table 3](https://dongwoo-im.github.io/assets/img/posts/2023-04-24-BarlowTwins/5-tab3.png){: .align-center}

Table 3: 다른 이미지 분류 dataset으로 fine-tune (transfer learning)

![Table 4](https://dongwoo-im.github.io/assets/img/posts/2023-04-24-BarlowTwins/6-tab4.png){: .align-center}

Table 4: 다른 task로 fine-tune (transfer learning)

### Ablations

### Discussion

### Appendix