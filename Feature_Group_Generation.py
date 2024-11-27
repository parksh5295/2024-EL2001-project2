# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 데이터 로드
df = pd.read_csv('./clustering_data.csv')  # CSV 파일 경로

# 2. 클러스터링 결과를 제외한 특성들만 선택 (클러스터 열 제외)
features = df.drop(columns=['cluster', 'label'])

# 3. 데이터 스케일링 (PCA에 적합하게 스케일링)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# 4. PCA 수행 (2개의 주성분으로 차원 축소)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# 5. PCA 결과를 원본 데이터에 추가
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

'''
# 6. 차원 축소된 결과를 클러스터별로 색을 다르게 시각화
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['PCA1'], df['PCA2'], c=df['cluster'], cmap='viridis')
plt.colorbar(scatter)  # 클러스터 색상 바 표시
plt.title('PCA of Clustering Results')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
'''

# 7. PCA로 설명된 분산 비율 출력
print(f'Explained variance ratio: {pca.explained_variance_ratio_}')
print(f'Total variance explained: {sum(pca.explained_variance_ratio_):.2f}')

# 각 주성분에 대한 설명된 분산 비율
explained_variance_ratio = pca.explained_variance_ratio_

# 주성분에 대한 기여도 출력
components = pd.DataFrame(pca.components_, columns=features.columns)

# 각 특성의 기여도 (각 주성분에 대한 기여도 합산)
feature_contribution = components.abs().sum(axis=0)

# 중요도가 큰 순서대로 출력
feature_contribution = feature_contribution.sort_values(ascending=False)

print(feature_contribution)
feature_contribution.to_csv('feature_contribution.csv')
