import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import numpy as np
import csv
import os
import time
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import chi2, f
from scipy.signal import welch
from scipy.interpolate import interp1d
import datetime


# Firebase 인증 및 초기화
cred = credentials.Certificate("firebase/hrvdataset-firebase-adminsdk-oof96-146efebb50.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://hrvdataset-default-rtdb.firebaseio.com/'
})

# 결과 저장 디렉토리 및 파일명 생성
if not os.path.exists('./result'):
    os.makedirs('./result')

current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILE_PATH = f"result/drowsy_result_{current_time_str}.csv"
TEMP_CSV_FILE_PATH = f"./result/temp_drowsy_result_{current_time_str}.csv"


# Time domain (from the table)
# mean_nni, median_nni, range_nni, sdnn, sdsd, rmssd, nni_50, pnni_50, nni_20, pnni_20, cvsd, cvnni, mean_hr, min_hr, max_hr, std_hr
time_domain_cols = [
    "mean_nni", "median_nni", "range_nni", "sdnn", "sdsd", "rmssd",
     "cvsd", "cvnni",
    "mean_hr", "min_hr", "max_hr", "std_hr"
]

# Frequency domain
# power_vlf, power_lf, power_hf, total_power, lf_hf_ratio
freq_domain_cols = ["power_vlf", "power_lf", "power_hf", "total_power", "lf_hf_ratio"]

# Nonlinear domain
# csi, cvi, modified_csi, sampen
nonlinear_domain_cols = ["csi", "cvi", "modified_csi", "sampen"]

pca_cols = [
    "Time_T2", "Time_SPE",
    "Frequency_T2", "Frequency_SPE",
    "Nonlinear_T2", "Nonlinear_SPE",
    "Drowsy"
]

header = [
    "StartTimestamp", "EndTimestamp", "ErrorCount", "RRCount"
] + time_domain_cols + freq_domain_cols + nonlinear_domain_cols + pca_cols

with open(CSV_FILE_PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

# Firebase에서 데이터 참조
ref = db.reference('HeartRateData')

# 슬라이딩 윈도우 데이터
rr_window = deque(maxlen=120)  # 최근 120초 데이터만 유지
last_processed_timestamp = None

hrv_history = []

def fetch_initial_rr_intervals():
    """
    Firebase에서 가장 최근 2분 데이터를 가져옵니다.
    """
    global last_processed_timestamp

    query = ref.order_by_child("timestamp").limit_to_last(120)
    data = query.get()

    if not data:
        print("No data found.")
        return []

    rr_data = []
    for key, value in sorted(data.items(), key=lambda item: item[1]['timestamp']):
        timestamp = value.get("timestamp")
        rr_interval = value.get("rrInterval", 0)
        is_error = value.get("isError", True)

        # 에러 데이터도 포함하되, RR 간격은 0으로 설정
        rr_data.append((timestamp, 0 if is_error else rr_interval))

    if rr_data:
        last_processed_timestamp = rr_data[-1][0]  # 가장 마지막 타임스탬프 업데이트

    return rr_data


def fetch_new_rr_intervals():
    """
    마지막 처리된 타임스탬프 이후의 데이터를 Firebase에서 가져옵니다.
    """
    global last_processed_timestamp

    if not last_processed_timestamp:
        print("Fetching initial data...")
        return fetch_initial_rr_intervals()

    query = ref.order_by_child("timestamp").start_at(last_processed_timestamp)
    data = query.get()

    if not data:
        print("No new data found.")
        return []

    rr_data = []
    for key, value in sorted(data.items(), key=lambda item: item[1]['timestamp']):
        timestamp = value.get("timestamp")
        rr_interval = value.get("rrInterval", 0)
        is_error = value.get("isError", True)
        
        # 동일한 timestamp 데이터는 제외
        if timestamp == last_processed_timestamp:
            continue

        # 에러 데이터도 포함하되, RR 간격은 0으로 설정
        rr_data.append((timestamp, 0 if is_error else rr_interval))

    if rr_data:
        last_processed_timestamp = rr_data[-1][0]  # 가장 마지막 타임스탬프 업데이트

    return rr_data


def calculate_time_domain_features(rr_intervals_ms):
    rr = np.array(rr_intervals_ms)
    if len(rr) < 2:
        return {col: None for col in time_domain_cols}

    diff_rr = np.diff(rr)
    mean_nni = np.mean(rr)
    median_nni = np.median(rr)
    range_nni = np.max(rr) - np.min(rr)
    sdnn = np.std(rr, ddof=1)
    sdsd = np.std(diff_rr, ddof=1)
    rmssd = np.sqrt(np.mean(diff_rr**2))

    nni_50 = np.sum(np.abs(diff_rr) > 50)
    pnni_50 = (nni_50 / len(diff_rr))*100 if len(diff_rr) > 0 else None

    nni_20 = np.sum(np.abs(diff_rr) > 20)
    pnni_20 = (nni_20 / len(diff_rr))*100 if len(diff_rr) > 0 else None

    cvsd = rmssd / mean_nni if mean_nni != 0 else None
    cvnni = sdnn / mean_nni if mean_nni != 0 else None

    # HR(bpm) = 60000 / RR(ms)
    hr = 60000.0 / rr
    mean_hr = np.mean(hr)
    min_hr = np.min(hr)
    max_hr = np.max(hr)
    std_hr = np.std(hr, ddof=1)

    return {
        "mean_nni": mean_nni,
        "median_nni": median_nni,
        "range_nni": range_nni,
        "sdnn": sdnn,
        "sdsd": sdsd,
        "rmssd": rmssd,
        "nni_50": nni_50,
        "pnni_50": pnni_50,
        "nni_20": nni_20,
        "pnni_20": pnni_20,
        "cvsd": cvsd,
        "cvnni": cvnni,
        "mean_hr": mean_hr,
        "min_hr": min_hr,
        "max_hr": max_hr,
        "std_hr": std_hr
    }


def calculate_frequency_domain_features(rr_intervals_ms, fs=4.0):
    """
    rr_intervals_ms: RR 간격(ms 단위) 리스트
    fs: 보간 후 샘플링 주파수(Hz)

    반환되는 지표:
    power_vlf: VLF 대역 파워(0.0033~0.04 Hz)
    power_lf: LF 대역 파워(0.04~0.15 Hz)
    power_hf: HF 대역 파워(0.15~0.4 Hz)
    total_power: (VLF+LF+HF)
    lf_hf_ratio: LF/HF 비율
    """
    
    # RR 간격이 리스트인 경우, 넘파이 배열로 변환
    rr_intervals_ms = np.array(rr_intervals_ms)
    
    # 최소 길이 조건 (여기서는 30개 이상을 예)
    if len(rr_intervals_ms) < 30:
        return {
            "power_vlf": None,
            "power_lf": None,
            "power_hf": None,
            "total_power": None,
            "lf_hf_ratio": None
        }

    # RR(ms) → 누적 시간(초)
    t = np.cumsum(rr_intervals_ms) / 1000.0
    duration = t[-1]

    # RR → HR 변환 (HR(bpm) = 60000 / RR(ms))
    hr = 60000.0 / rr_intervals_ms

    # 균일 시간축으로 보간
    t_uniform = np.arange(0, duration, 1.0/fs)
    interp_func = interp1d(t, hr, kind='cubic', fill_value="extrapolate")
    hr_interpolated = interp_func(t_uniform)

    # Welch PSD 계산
    # nperseg는 데이터 길이에 맞게 조정 가능
    freq, power = welch(hr_interpolated, fs=fs, nperseg=min(len(hr_interpolated), 256))

    def band_power(band):
        mask = (freq >= band[0]) & (freq < band[1])
        return np.trapz(power[mask], freq[mask]) if np.any(mask) else 0.0

    # 주파수 대역 정의 (단기 분석 기준)
    vlf_band = (0.0033, 0.04)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)

    power_vlf = band_power(vlf_band)
    power_lf = band_power(lf_band)
    power_hf = band_power(hf_band)
    total_power = power_vlf + power_lf + power_hf
    lf_hf_ratio = (power_lf / power_hf) if power_hf > 0 else None

    return {
        "power_vlf": power_vlf,
        "power_lf": power_lf,
        "power_hf": power_hf,
        "total_power": total_power,
        "lf_hf_ratio": lf_hf_ratio
    }


def calculate_nonlinear_features(rr_intervals_ms):
    rr = np.array(rr_intervals_ms)
    if len(rr) < 2:
        return {col: None for col in nonlinear_domain_cols}

    diff_rr = np.diff(rr)
    sdnn = np.std(rr, ddof=1)
    sd1 = np.std(diff_rr, ddof=1)/np.sqrt(2)
    sd2 = np.sqrt(2*sdnn**2 - sd1**2) if sdnn > 0 else None

    csi = sd2/sd1 if (sd2 is not None and sd1 > 0) else None
    cvi = np.log(sd1*sd2) if (sd2 is not None and sd1>0 and sd2>0) else None
    modified_csi = (sd2**2)/sd1 if (sd2 is not None and sd1>0) else None

    se = sample_entropy(rr_intervals_ms, m=2, r=0.2)

    return {
        "csi": csi,
        "cvi": cvi,
        "modified_csi": modified_csi,
        "sampen": se
    }

def sample_entropy(rr_intervals_ms, m=2, r=0.2):
    rr = np.array(rr_intervals_ms)
    if len(rr) < m+1:
        return None
    std_rr = np.std(rr, ddof=1)
    if std_rr == 0:
        return None
    r = r * std_rr

    def _phi(m):
        x = np.array([rr[i:i+m] for i in range(len(rr)-m+1)])
        C = np.zeros(len(rr)-m+1)
        for i in range(len(rr)-m+1):
            dist = np.max(np.abs(x - x[i]), axis=1)
            C[i] = np.sum(dist < r) - 1
        return np.sum(C)/(len(rr)-m+1)/(len(rr)-m)

    phi_m = _phi(m)
    phi_m1 = _phi(m+1)
    if phi_m == 0 or phi_m1 == 0:
        return None
    return -np.log(phi_m1/phi_m)

# 결과를 csv파일로 기록하는 내용
def write_to_csv(start_timestamp, end_timestamp, error_count, rr_count, hrv_data):
    with open(TEMP_CSV_FILE_PATH, mode='w', newline='') as temp_file:
        writer = csv.writer(temp_file)
        
        # 기존 파일 복사
        if os.path.exists(CSV_FILE_PATH):
            with open(CSV_FILE_PATH, mode='r') as original_file:
                reader = csv.reader(original_file)
                for row in reader:
                    writer.writerow(row)
        
        row_data = [
            start_timestamp,
            end_timestamp,
            error_count,
            rr_count
        ]
        # time domain cols
        for col in time_domain_cols:
            row_data.append(hrv_data.get(col))
        # freq domain cols
        for col in freq_domain_cols:
            row_data.append(hrv_data.get(col))
        # nonlinear domain cols
        for col in nonlinear_domain_cols:
            row_data.append(hrv_data.get(col))
        # pca cols
        for col in pca_cols:
            row_data.append(hrv_data.get(col))

        writer.writerow(row_data)
    
    # 기존 CSV 삭제 후 임시파일 rename
    if os.path.exists(CSV_FILE_PATH):
        os.remove(CSV_FILE_PATH)
    os.rename(TEMP_CSV_FILE_PATH, CSV_FILE_PATH)
    
    
def load_training_hrv_data(csv_path, limit=None):
    """
    학습된 HRV 데이터를 CSV에서 로드합니다.
    Args:
        csv_path (str): CSV 파일 경로
        limit (int, optional): 가져올 데이터 개수 (None이면 모든 데이터를 로드)
    Returns:
        list: 유의미한 HRV 데이터를 포함한 리스트
    """
    if not os.path.exists(csv_path):
        print(f"CSV 파일이 존재하지 않습니다: {csv_path}")
        return []

    meaningful_cols = time_domain_cols + freq_domain_cols + nonlinear_domain_cols
    training_hrv = []
    
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            
            # 필요한 키들 포함
            hrv_entry = {
                "StartTimestamp": row["StartTimestamp"],
                "EndTimestamp": row["EndTimestamp"]
            }
            
            for col in meaningful_cols:
                hrv_entry[col] = float(row[col]) if row[col] else None
            
            training_hrv.append(hrv_entry)

    print(f"{len(training_hrv)}개의 학습 HRV 데이터를 로드했습니다.")
    return training_hrv


# 초기에 학습된 HRV 데이터를 로드
TRAINING_CSV_PATH = "data/data_set.csv"
training_hrv_data = load_training_hrv_data(TRAINING_CSV_PATH)


# 업데이트된 process_hrv 함수
def process_hrv():
    """
    새로운 데이터를 가져와 슬라이딩 윈도우 방식으로 HRV를 계산합니다.
    """
    global rr_window

    # 초기 데이터 로드
    if not rr_window:
        rr_window.extend(fetch_initial_rr_intervals())

    # 새로운 데이터 가져오기
    new_rr_data = fetch_new_rr_intervals()

    if not new_rr_data:
        print("No new data to process.")
        return

    # 새로운 데이터 추가 및 슬라이딩 윈도우 유지
    for timestamp, rr_interval in new_rr_data:
        rr_window.append((timestamp, rr_interval))

    # 슬라이딩 윈도우가 가득 찼을 때 HRV 계산
    if len(rr_window) == 120:
        start_timestamp = rr_window[0][0]
        end_timestamp = rr_window[-1][0]

        # 에러 데이터를 제외한 RR 간격 추출
        rr_list = [item[1] for item in rr_window if item[1] > 0]
        error_count = 120 - len(rr_list)

        if error_count >= 48:
            print(f"Too many errors ({error_count}). Skipping HRV calculation for window {start_timestamp} to {end_timestamp}.")
        else:
            print(f"Calculating HRV for window {start_timestamp} to {end_timestamp}...")

            time_feats = calculate_time_domain_features(rr_list)
            freq_feats = calculate_frequency_domain_features(rr_list)
            nonlinear_feats = calculate_nonlinear_features(rr_list)

            hrv_data = {}
            hrv_data.update(time_feats)
            hrv_data.update(freq_feats)
            hrv_data.update(nonlinear_feats)

            hrv_data["Time_T2"] = None
            hrv_data["Time_SPE"] = None
            hrv_data["Frequency_T2"] = None
            hrv_data["Frequency_SPE"] = None
            hrv_data["Nonlinear_T2"] = None
            hrv_data["Nonlinear_SPE"] = None
            hrv_data["Drowsy"] = None  # 졸음 여부도 나중에 결정

            hrv_history.append((start_timestamp, end_timestamp, error_count, len(rr_list), hrv_data))

            # 학습된 HRV 데이터를 포함하여 이상탐지 수행
            if len(hrv_history) >= 1:
                perform_pca_and_detect_anomaly(training_hrv_data)
            else:
                write_to_csv(start_timestamp, end_timestamp, error_count, len(rr_list), hrv_data)

# PCA 수행 함수
def perform_domain_pca(domain_data):
    if domain_data is None or domain_data.shape[0] < 2:
        return None, None, None, None
    
    scaler = StandardScaler()
    domain_data_standardized = scaler.fit_transform(domain_data)
    
    n_components = min(3, domain_data_standardized.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(domain_data_standardized)
    eigenvalues = pca.explained_variance_
    loadings = pca.components_
    
    return pca, eigenvalues, scores, loadings
                  
def perform_pca_and_detect_anomaly(training_data):
    global hrv_history

    # 도메인별 feature keys
    time_domain_keys = time_domain_cols
    frequency_domain_keys = freq_domain_cols
    nonlinear_domain_keys = nonlinear_domain_cols

    def get_valid_domain_data(feature_keys):
        valid_entries = [(st, et, ec, rc, hd) for (st, et, ec, rc, hd) in hrv_history if all(hd.get(k) is not None for k in feature_keys)]
        for entry in training_data:
            if all(entry.get(k) is not None for k in feature_keys):
                valid_entries.append((entry["StartTimestamp"], entry["EndTimestamp"], None, None, entry))
        if len(valid_entries) < 2:
            return None, None
        data_matrix = np.array([[hd[k] for k in feature_keys] for (_, _, _, _, hd) in valid_entries])
        return valid_entries, data_matrix

    # PCA 수행 및 이상탐지 로직은 기존과 동일
    time_entries, time_data = get_valid_domain_data(time_domain_keys)
    freq_entries, freq_data = get_valid_domain_data(frequency_domain_keys)
    nonlin_entries, nonlin_data = get_valid_domain_data(nonlinear_domain_keys)

    # 도메인별 PCA 및 이상탐지 수행
    time_pca, time_eig, time_scores, time_loadings = perform_domain_pca(time_data)
    freq_pca, freq_eig, freq_scores, freq_loadings = perform_domain_pca(freq_data)
    nonlin_pca, nonlin_eig, nonlin_scores, nonlin_loadings = perform_domain_pca(nonlin_data)

    # 임계값 계산 함수
    def calculate_limits(scores, loadings, input_data, level_confidence=0.9):
        # T² limit 계산 (기존 동일)
        k = scores.shape[1]
        n = scores.shape[0]
        t2_limit = ((k * (n - 1) * (n + 1)) / (n * (n - k))) * f.ppf(level_confidence, k, n - k)

        # SPE limit 계산 시 해당 도메인에 맞는 데이터 사용
        q_stat = q_statistic(input_data, loadings, scores)
        d1 = q_stat.var() / (2 * q_stat.mean())
        df = (2 * q_stat.mean()**2) / q_stat.var()
        spe_limit = d1 * chi2.ppf(level_confidence, df)

        return t2_limit, spe_limit

    
    def q_statistic(input_features, loadings, scores):
        """
        SPE (Squared Prediction Error)를 계산하는 함수.
        input_features: 원본 입력 데이터 (n_samples x n_features)
        loadings: PCA 로딩 행렬 (n_components x n_features)
        scores: PCA 점수 행렬 (n_samples x n_components)
        """
        # PCA를 통해 재구성된 데이터
        estimation_x = np.dot(scores, loadings)
        # 재구성 오차 계산
        error = input_features - estimation_x
        # 각 샘플에 대해 Squared Error를 계산
        q_statistic = np.sum(error**2, axis=1)
        return q_statistic

    
    # T²와 SPE 계산 함수
    def calculate_t2_spe(scores, loadings, eigenvalues, input_data):
        t_squared = np.sum((scores**2) / eigenvalues, axis=1)
        spe = q_statistic(input_data, loadings, scores)
        return t_squared, spe

    # 도메인별 임계값 계산
    time_t2_limit, time_spe_limit = calculate_limits(time_scores, time_loadings, time_data)
    freq_t2_limit, freq_spe_limit = calculate_limits(freq_scores, freq_loadings, freq_data)
    nonlin_t2_limit, nonlin_spe_limit = calculate_limits(nonlin_scores, nonlin_loadings, nonlin_data)


    # 마지막 HRV 데이터
    last_entry_index = len(hrv_history) - 1
    last_entry = hrv_history[last_entry_index]
    last_hrv_data = last_entry[4]

    def domain_t2_spe(pca, eigenvalues, scores, loadings, input_data):
        if pca is None or scores is None:
            return None, None
        t_squared, spe = calculate_t2_spe(scores, loadings, eigenvalues, input_data)
        return t_squared[-1], spe[-1]

    # Time Domain
    time_t2, time_spe = domain_t2_spe(time_pca, time_eig, time_scores, time_loadings, time_data)
    # Frequency Domain
    freq_t2, freq_spe = domain_t2_spe(freq_pca, freq_eig, freq_scores, freq_loadings, freq_data)
    # Nonlinear Domain
    nonlin_t2, nonlin_spe = domain_t2_spe(nonlin_pca, nonlin_eig, nonlin_scores, nonlin_loadings, nonlin_data)

    # 졸음 여부 판단
    drowsy = 0
    if (time_t2 and time_t2 > time_t2_limit) or (time_spe and time_spe > time_spe_limit):
        drowsy = 1
    elif (freq_t2 and freq_t2 > freq_t2_limit) or (freq_spe and freq_spe > freq_spe_limit):
        drowsy = 1
    elif (nonlin_t2 and nonlin_t2 > nonlin_t2_limit) or (nonlin_spe and nonlin_spe > nonlin_spe_limit):
        drowsy = 1
        
    # hrv_history 및 CSV 업데이트
    updated_hrv_data = last_hrv_data.copy()
    updated_hrv_data["Time_T2"] = time_t2
    updated_hrv_data["Time_SPE"] = time_spe
    updated_hrv_data["Frequency_T2"] = freq_t2
    updated_hrv_data["Frequency_SPE"] = freq_spe
    updated_hrv_data["Nonlinear_T2"] = nonlin_t2
    updated_hrv_data["Nonlinear_SPE"] = nonlin_spe
    updated_hrv_data["Drowsy"] = drowsy
    
    # 기록 업데이트
    hrv_history[last_entry_index] = (last_entry[0], last_entry[1], last_entry[2], last_entry[3], updated_hrv_data)
    write_to_csv(last_entry[0], last_entry[1], last_entry[2], last_entry[3], updated_hrv_data)

    # 디버깅 메시지 출력
    print(f"Time: T²={time_t2}/{time_t2_limit}, SPE={time_spe}/{time_spe_limit}")
    print(f"Frequency: T²={freq_t2}/{freq_t2_limit}, SPE={freq_spe}/{freq_spe_limit}")
    print(f"Nonlinear: T²={nonlin_t2}/{nonlin_t2_limit}, SPE={nonlin_spe}/{nonlin_spe_limit}")
    print(f"Drowsy={drowsy}")
    
    print("Explained Variance Ratio (Time Domain):", time_pca.explained_variance_ratio_)
    print("Explained Variance Ratio (Frequency Domain):", freq_pca.explained_variance_ratio_)
    print("Explained Variance Ratio (Nonlinear Domain):", nonlin_pca.explained_variance_ratio_)

def save_drowsy_result(drowsy, current_hr, current_rr_size):
    """
    졸음 여부와 현재 HR(심박수)을 텍스트 파일에 저장합니다.
    """
    with open("drowsy_result.txt", "w") as file:
        file.write(f"{drowsy}\n")
        file.write(f"{current_hr:.2f}\n")
        file.write(f"{current_rr_size}\n")



if __name__ == "__main__":
    while True:
        # HRV 데이터 처리
        process_hrv()

        # 현재 RR Interval을 HR로 변환
        current_rr_interval = rr_window[-1][1] if rr_window else None
        if current_rr_interval and current_rr_interval > 0:
            # 현재 RR Interval이 유효한 경우 HR 계산
            current_hr = 60000.0 / current_rr_interval
        else:
            # RR 데이터가 없거나 유효하지 않은 경우
            current_hr = 0
            
        # RR 간격 크기 계산
        rr_size = len(rr_window)

        # 졸음 상태 (HRV 결과가 없으면 초기값 1 사용)
        if hrv_history:
            last_hrv_entry = hrv_history[-1][4]  # 마지막 HRV 데이터
            drowsy = last_hrv_entry.get("Drowsy", 1)  # 기본값: 졸음 상태로 판단
        else:
            drowsy = 1  # 초기 상태에서 졸음 상태로 판단

        # 결과 저장
        save_drowsy_result(drowsy, current_hr, rr_size)

        print(f"Drowsy: {drowsy}, Current HR: {current_hr:.2f}, RR Size: {rr_size}")
        print("Waiting for the next interval...")
        time.sleep(1)
