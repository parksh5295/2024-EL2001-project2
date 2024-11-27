import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# 데이터 불러오기
df = pd.read_csv('./quantized_train_data.csv')

# 데이터 전처리 (숫자형 데이터만 추출하여 사용)
df_encoded = df.select_dtypes(include=['float64', 'int64'])

# 계층적 군집화 (Agglomerative Clustering)
agg_clustering = AgglomerativeClustering(n_clusters=2)  # 2개의 군집으로 클러스터링
df_encoded['cluster'] = agg_clustering.fit_predict(df_encoded)

# 군집화 결과 확인
print(df_encoded.head())

df_encoded.to_csv('./clustering_data.csv', index=False)

# 데이터 전처리: 숫자형 데이터만 추출
df_encoded = df.select_dtypes(include=['float64', 'int64'])

# 데이터를 표준화(Standardization)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_encoded)

# 계층적 군집화 (linkage 함수 사용)
Z = linkage(scaled_data, method='ward')  # ward method는 군집 간 분산 최소화 방법

# 덴드로그램 시각화
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
