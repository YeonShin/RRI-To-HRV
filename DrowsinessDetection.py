import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import numpy as np
import csv
import os
import time
from collections import deque

# Firebase 인증 및 초기화
cred = credentials.Certificate("firebase/hrvdataset-firebase-adminsdk-oof96-146efebb50.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://hrvdataset-default-rtdb.firebaseio.com/'
})

# CSV 파일 초기화
CSV_FILE_PATH = "hrv_results.csv"
TEMP_CSV_FILE_PATH = "temp_hrv_results.csv"

if not os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["StartTimestamp", "EndTimestamp", "ErrorCount", "RRCount", "SDNN", "LF", "HF", "LF/HF", "SD1", "SD2", "HotellingT2", "SPE"])

# Firebase에서 데이터 참조
ref = db.reference('HeartRateData')

# 슬라이딩 윈도우 데이터
rr_window = deque(maxlen=120)  # 최근 120초 데이터만 유지
last_processed_timestamp = None

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


def calculate_time_domain_hrv(rr_intervals):
    if len(rr_intervals) < 2:
        return {"SDNN": None}
    
    sdnn = np.std(rr_intervals, ddof=1)
    return {"SDNN": sdnn}

def calculate_frequency_domain_hrv(rr_intervals):
    if len(rr_intervals) < 30:  # 데이터 부족
        return {"LF": None, "HF": None, "LF/HF": None}
    
    from scipy.signal import welch
    freq, power = welch(rr_intervals, fs=4.0, nperseg=len(rr_intervals))

    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)

    lf_power = np.trapz(power[(freq >= lf_band[0]) & (freq < lf_band[1])], freq[(freq >= lf_band[0]) & (freq < lf_band[1])])
    hf_power = np.trapz(power[(freq >= hf_band[0]) & (freq < hf_band[1])], freq[(freq >= hf_band[0]) & (freq < hf_band[1])])
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else None

    return {"LF": lf_power, "HF": hf_power, "LF/HF": lf_hf_ratio}

def calculate_nonlinear_domain_hrv(rr_intervals):
    if len(rr_intervals) < 2:
        return {"SD1": None, "SD2": None}
    
    diff_rr = np.diff(rr_intervals)
    sd1 = np.std(diff_rr) / np.sqrt(2)
    sd2 = np.sqrt(2 * np.std(rr_intervals, ddof=1)**2 - sd1**2)
    return {"SD1": sd1, "SD2": sd2}

def calculate_hotelling_t2_spe(hrv_data):
    """
    Hotelling T² 및 SPE 값을 계산합니다.
    """
    try:
        hrv_values = np.array([v for v in hrv_data.values() if v is not None])
        if len(hrv_values) == 0:
            return {"HotellingT2": None, "SPE": None}
        
        mean = np.mean(hrv_values)
        cov_matrix = np.cov(hrv_values)
        t2_stat = np.dot(hrv_values - mean, np.linalg.inv(cov_matrix)).dot(hrv_values - mean)
        spe = np.sum((hrv_values - mean)**2)
        
        return {"HotellingT2": t2_stat, "SPE": spe}
    except Exception as e:
        print(f"Error calculating Hotelling T² and SPE: {e}")
        return {"HotellingT2": None, "SPE": None}

def write_to_csv(start_timestamp, end_timestamp, error_count, rr_count, hrv_data):
    # 기존 파일에서 데이터를 읽어와 임시 파일에 복사
    with open(TEMP_CSV_FILE_PATH, mode='w', newline='') as temp_file:
        writer = csv.writer(temp_file)
        
        # 기존 파일이 존재하면 내용을 복사
        if os.path.exists(CSV_FILE_PATH):
            with open(CSV_FILE_PATH, mode='r') as original_file:
                reader = csv.reader(original_file)
                for row in reader:
                    writer.writerow(row)
        
        # 새로운 데이터 추가
        writer.writerow([
            start_timestamp,
            end_timestamp,
            error_count,
            rr_count,
            hrv_data.get("SDNN"),
            hrv_data.get("LF"),
            hrv_data.get("HF"),
            hrv_data.get("LF/HF"),
            hrv_data.get("SD1"),
            hrv_data.get("SD2"),
            hrv_data.get("HotellingT2"),
            hrv_data.get("SPE")
        ])
    
    # 기존 CSV 파일 삭제 후 임시 파일을 새 파일로 이동
    if os.path.exists(CSV_FILE_PATH):
        os.remove(CSV_FILE_PATH)  # 기존 파일 삭제
    os.rename(TEMP_CSV_FILE_PATH, CSV_FILE_PATH)  # 임시 파일을 새 파일로 교체

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

        if error_count >= 120:
            print(f"Too many errors ({error_count}). Skipping HRV calculation for window {start_timestamp} to {end_timestamp}.")
        else:
            print(f"Calculating HRV for window {start_timestamp} to {end_timestamp}...")
            # HRV 계산
            time_domain_hrv = calculate_time_domain_hrv(rr_list)
            frequency_domain_hrv = calculate_frequency_domain_hrv(rr_list)
            nonlinear_domain_hrv = calculate_nonlinear_domain_hrv(rr_list)

            hrv_data = {**time_domain_hrv, **frequency_domain_hrv, **nonlinear_domain_hrv}
            t2_spe = calculate_hotelling_t2_spe(hrv_data)
            hrv_data.update(t2_spe)

            print(f"HRV Calculated: {hrv_data}")
            write_to_csv(start_timestamp, end_timestamp, error_count, len(rr_list), hrv_data)

if __name__ == "__main__":
    while True:
        process_hrv()
        print("Waiting for the next interval...")
        time.sleep(1)
