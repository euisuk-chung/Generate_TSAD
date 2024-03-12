import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

def generate_time_series_with_anomalies(num_points=1000, 
                                        anomaly_rate=0.01, 
                                        normal_range=(0, 1), 
                                        anomaly_range=(2, 3), 
                                        include_sine=False, 
                                        include_cosine=False,
                                        anomaly_length=6,
                                        additional_anomaly_prob=0.4,
                                        freq=20,
                                        amp=0.5,
                                        slope=0):
    """
    시계열 데이터와 이상치를 생성하는 함수입니다.

    Parameters:
    - num_points (int): 생성할 데이터 포인트의 총 개수입니다.
    - anomaly_rate (float): 전체 데이터 포인트 중 이상치의 비율입니다.
    - normal_range (tuple): 정상 데이터 포인트의 값 범위입니다.
    - anomaly_range (tuple): 이상치 데이터 포인트의 값 범위입니다.
    - include_sine (bool): 사인 파형을 데이터에 포함시킬지 여부입니다.
    - include_cosine (bool): 코사인 파형을 데이터에 포함시킬지 여부입니다.
    - anomaly_length (int): 이상치가 연속으로 나타나는 길이입니다.
    - additional_anomaly_prob (float): 추가 이상치가 발생할 확률입니다.
    - freq (int): 사인/코사인 파형의 주파수입니다.
    - amp (float): 사인/코사인 파형의 진폭입니다.
    - slope (float): 데이터의 기울기로, 시계열의 경향성을 결정합니다.

    Returns:
    - time_series (numpy.ndarray): 생성된 시계열 데이터입니다.
    - anomaly_indices (list): 이상치의 인덱스 목록입니다.
    """
    
    # 기본 시계열 데이터 생성
    time_series = np.random.uniform(low=normal_range[0], high=normal_range[1], size=num_points)

    # 기울기 적용
    time_series += np.arange(num_points) * slope

    # 사인 파형 추가
    if include_sine:
        time_series += amp * np.sin(2 * np.pi * freq * np.linspace(0, 20, num_points))

    # 코사인 파형 추가
    if include_cosine:
        time_series += amp * np.cos(2 * np.pi * freq * np.linspace(0, 20, num_points))

    # 이상 데이터 포인트 수 계산
    num_anomalies = int(num_points * anomaly_rate)

    # 이상 데이터 포인트의 인덱스 선택
    anomaly_indices = []

    # 이상 데이터 포인트가 골고루 분포하도록 생성
    interval = num_points // num_anomalies
    for i in range(num_anomalies):
        start_idx = i * interval
        end_idx = (i+1) * interval

        # 이전 이상치와의 거리에 따라 인덱스 선택
        idx = np.random.randint(start_idx, end_idx)

        # 이미 선택된 인덱스인 경우 다시 선택
        while idx in anomaly_indices:
            idx = np.random.randint(start_idx, end_idx)

        # 연속적인 이상 데이터 포인트 생성
        for j in range(idx, idx + anomaly_length):
            if np.random.choice([True, False], p=[additional_anomaly_prob, 1-additional_anomaly_prob]):
                time_series[j]+= np.random.uniform(low=anomaly_range[0], high=anomaly_range[1])
                anomaly_indices.append(j)

    return time_series, anomaly_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate time series data with anomalies')
    parser.add_argument('--num_points', type=int, default=1000, help='Number of data points')
    parser.add_argument('--anomaly_rate', type=float, default=0.01, help='Anomaly rate')
    parser.add_argument('--normal_range', type=tuple, default=(0, 1), help='Range for normal data')
    parser.add_argument('--anomaly_range', type=tuple, default=(2, 3), help='Range for anomaly data')
    parser.add_argument('--include_sine', type=bool, default=False, help='Include sine wave')
    parser.add_argument('--include_cosine', type=bool, default=False, help='Include cosine wave')
    parser.add_argument('--anomaly_length', type=int, default=6, help='Length of anomaly')
    parser.add_argument('--additional_anomaly_prob', type=float, default=0.5, help='Probability of sequential anomalies')
    parser.add_argument('--freq', type=int, default=20, help='Frequency of sine/cosine wave')
    parser.add_argument('--amp', type=float, default=0.4, help='Amplitude of sine/cosine wave')
    parser.add_argument('--slope', type=float, default=0, help='Slope of the time series trend')

    args = parser.parse_args()

    time_series, anomaly_indices = generate_time_series_with_anomalies(args.num_points, 
                                                                       args.anomaly_rate, 
                                                                       args.normal_range, 
                                                                       args.anomaly_range, 
                                                                       args.include_sine, 
                                                                       args.include_cosine,
                                                                       args.anomaly_length,
                                                                       args.additional_anomaly_prob,
                                                                       args.freq,
                                                                       args.amp,
                                                                       args.slope)

    print('Generated!')
    # 데이터프레임 생성
    df = pd.DataFrame({'value': time_series})
    df['anomaly']= 0
    df.loc[anomaly_indices, 'anomaly']= 1

    # 실행된 날짜와 시간 추가
    current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 데이터프레임 저장
    filename = f'./Results/data/{current_datetime}_num{args.num_points}_rate_{int(args.anomaly_rate*100)}_slope_{int(args.slope*100)}.csv'
    df.to_csv(filename, index=False)

    # 결과 시각화
    plt.figure(figsize=(15, 5))
    plt.plot(df['value'], label='Time Series')
    plt.scatter(df[df['anomaly']== 1].index, df[df['anomaly']== 1]['value'], color='red', label='Anomalies')
    plt.legend()
    plt.savefig(f'./Results/image/{current_datetime}_num{args.num_points}_rate_{int(args.anomaly_rate*100)}_slope_{int(args.slope*100)}.png')
    print('Complete!')