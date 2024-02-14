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
                                        amp=0.5):
    # 기본 시계열 데이터 생성
    time_series = np.random.uniform(low=normal_range[0], high=normal_range[1], size=num_points)

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
    parser.add_argument('--additional_anomaly_prob', type=float, default=0.5, help='Probability of additional anomalies')
    parser.add_argument('--freq', type=int, default=20, help='Frequency of sine/cosine wave')
    parser.add_argument('--amp', type=float, default=0.4, help='Amplitude of sine/cosine wave')

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
                                                                       args.amp)

    print('Generated!')
    # 데이터프레임 생성
    df = pd.DataFrame({'value': time_series})
    df['anomaly']= 0
    df.loc[anomaly_indices, 'anomaly']= 1

    # 실행된 날짜와 시간 추가
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # 데이터프레임 저장
    filename = f'./Result/generated_{current_datetime}.csv'
    df.to_csv(filename, index=False)

    # 결과 시각화
    plt.figure(figsize=(15, 5))
    plt.plot(df['value'], label='Time Series')
    plt.scatter(df[df['anomaly']== 1].index, df[df['anomaly']== 1]['value'], color='red', label='Anomalies')
    plt.legend()
    plt.savefig(f'./Result/image_{current_datetime}.png')
    print('Complete!')
