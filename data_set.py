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
if not os.path.exists('./data'):
    os.makedirs('./data')

current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILE_PATH = f"data/hrv_data_{current_time_str}.csv"
TEMP_CSV_FILE_PATH = f"./data/temp_hrv_datta_{current_time_str}.csv"


# Time domain (from the table)
# mean_nni, median_nni, range_nni, sdnn, sdsd, rmssd, nni_50, pnni_50, nni_20, pnni_20, cvsd, cvnni, mean_hr, min_hr, max_hr, std_hr
time_domain_cols = [
    "mean_nni", "median_nni", "range_nni", "sdnn", "sdsd", "rmssd",
    "nni_50", "pnni_50", "nni_20", "pnni_20", "cvsd", "cvnni",
    "mean_hr", "min_hr", "max_hr", "std_hr"
]

# Frequency domain
# power_vlf, power_lf, power_hf, total_power, lf_hf_ratio
freq_domain_cols = ["power_vlf", "power_lf", "power_hf", "total_power", "lf_hf_ratio"]

# Nonlinear domain
# csi, cvi, modified_csi, sampen
nonlinear_domain_cols = ["csi", "cvi", "modified_csi", "sampen"]


header = [
    "StartTimestamp", "EndTimestamp", "ErrorCount", "RRCount"
] + time_domain_cols + freq_domain_cols + nonlinear_domain_cols

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

        writer.writerow(row_data)
    
    # 기존 CSV 삭제 후 임시파일 rename
    if os.path.exists(CSV_FILE_PATH):
        os.remove(CSV_FILE_PATH)
    os.rename(TEMP_CSV_FILE_PATH, CSV_FILE_PATH)
    


def process_hrv():
    """
    새로운 데이터를 가져와 슬라이딩 윈도우 방식으로 HRV를 계산합니다.
    """
    global rr_window

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

        if error_count >= 120:
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

            
            
            hrv_history.append((start_timestamp, end_timestamp, error_count, len(rr_list), hrv_data))

            write_to_csv(start_timestamp, end_timestamp, error_count, len(rr_list), hrv_data)
            
            # HRV 계산을 한 번 수행한 후 슬라이딩 윈도우를 초기화 한다.
            rr_window.clear()
                
if __name__ == "__main__":
    # 초기 데이터 로드
    if not rr_window:
        rr_window.extend(fetch_initial_rr_intervals())
    while True:
        process_hrv()
        print("Waiting for the next interval...")
        time.sleep(1)