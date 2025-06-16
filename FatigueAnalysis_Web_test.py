import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import plotly.graph_objects as go
import streamlit as st
from fastdtw import fastdtw
from dtaidistance import dtw
from streamlit_plotly_events import plotly_events
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import find_peaks
from supabase import create_client
import io

# Supabase設定
SUPABASE_URL = "https://sqyludydesosumixzuzf.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNxeWx1ZHlkZXNvc3VtaXh6dXpmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk5MTEzODQsImV4cCI6MjA2NTQ4NzM4NH0.fkQTHATCLR0RQdEDL-LCsg6Vlgal4WkogfQPVi9o1_c"
BUCKET_NAME = "fatigue-data"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 設定 ---
distance_type = 'DTW'
plot_type = "before_peak"

# --- データセット一覧取得 ---
all_files = supabase.storage.from_(BUCKET_NAME).list("", {"recursive": True})
existing_datasets = sorted(list({
    f["name"].split("/")[0]
    for f in all_files
    if "/" in f["name"] and not f["name"].endswith(".csv") and not f["name"].split("/")[0].endswith("_SpeedRoll")
}))

options = existing_datasets + ["新規作成"]
selected_option = st.selectbox("使用するデータセットを選択", options)

if selected_option == "新規作成":
    new_dataset = st.text_input("新しいデータセット名を入力", "")
    if new_dataset:
        selected_dataset = new_dataset
        path = f"{selected_dataset}"
        speed_path = f"{selected_dataset}_SpeedRoll"

        # ダミーファイルをアップロードして"フォルダ"を作る
        dummy = b"init"
        supabase.storage.from_(BUCKET_NAME).upload(f"{path}/__init__.txt", dummy, {"content-type": "text/plain"})
        supabase.storage.from_(BUCKET_NAME).upload(f"{speed_path}/__init__.txt", dummy, {"content-type": "text/plain"})
    else:
        st.warning("データセット名を入力してください。")
        st.stop()
else:
    selected_dataset = selected_option
    path = f"{selected_dataset}"
    speed_path = f"{selected_dataset}_SpeedRoll"

# --- サブフォルダ取得関数 ---
def get_subfolders(base_path):
    files = supabase.storage.from_(BUCKET_NAME).list(base_path + "/", {"recursive": True})
    return sorted(list({f["name"].split("/")[1] for f in files if f["name"].count("/") >= 2}))

folders = get_subfolders(path)
folders.append("新規フォルダを作成")
selected_subfolder = st.selectbox(f"'{selected_dataset}': サブフォルダ選択", folders)

if selected_subfolder == "新規フォルダを作成":
    new_folder_name = st.text_input(f"'{selected_dataset}': 新しいフォルダ名を入力")
    if new_folder_name:
        selected_subfolder = new_folder_name
        dummy = b"init"
        supabase.storage.from_(BUCKET_NAME).upload(f"{path}/{selected_subfolder}/__init__.txt", dummy, {"content-type": "text/plain"})
    else:
        st.stop()

# --- アップロード処理 ---
st.markdown(f"#### '{selected_dataset}/{selected_subfolder}': CSVアップロード")
uploaded_files = st.file_uploader("CSVファイルを選択", type="csv", accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        supa_path = f"{path}/{selected_subfolder}/{file.name}"
        try:
            supabase.storage.from_(BUCKET_NAME).remove(supa_path)
        except:
            pass
        supabase.storage.from_(BUCKET_NAME).upload(supa_path, file, {"content-type": "text/csv"})
    st.success(f"{len(uploaded_files)} ファイルをアップロードしました。")

# --- 削除処理 ---
files = supabase.storage.from_(BUCKET_NAME).list(f"{path}/{selected_subfolder}")
csv_files = [f["name"] for f in files if f["name"].endswith(".csv")]
if csv_files:
    selected_file = st.selectbox("削除したいCSVファイルを選択", csv_files)
    if st.button("このCSVファイルを削除"):
        supabase.storage.from_(BUCKET_NAME).remove(f"{selected_file}")
        st.success(f"{selected_file} を削除しました。")
        st.experimental_rerun()

# --- SpeedRoll 側 ---
speed_folders = get_subfolders(speed_path)
speed_folders.append("新規フォルダを作成")
selected_speed_subfolder = st.selectbox(f"'{selected_dataset}_SpeedRoll': サブフォルダ選択", speed_folders)

if selected_speed_subfolder == "新規フォルダを作成":
    new_folder_name = st.text_input(f"'{selected_dataset}_SpeedRoll': 新しいフォルダ名を入力", key="speed_new_folder")
    if new_folder_name:
        selected_speed_subfolder = new_folder_name
        dummy = b"init"
        supabase.storage.from_(BUCKET_NAME).upload(f"{speed_path}/{selected_speed_subfolder}/__init__.txt", dummy, {"content-type": "text/plain"})
    else:
        st.stop()

st.markdown(f"#### '{selected_dataset}_SpeedRoll/{selected_speed_subfolder}': CSVアップロード")
uploaded_speed_files = st.file_uploader("CSVファイルを選択", type="csv", accept_multiple_files=True, key="speed_upload")
if uploaded_speed_files:
    for file in uploaded_speed_files:
        supa_path = f"{speed_path}/{selected_speed_subfolder}/{file.name}"
        try:
            supabase.storage.from_(BUCKET_NAME).remove(supa_path)
        except:
            pass
        supabase.storage.from_(BUCKET_NAME).upload(supa_path, file, {"content-type": "text/csv"})
    st.success(f"{len(uploaded_speed_files)} ファイルをアップロードしました。")

files = supabase.storage.from_(BUCKET_NAME).list(f"{speed_path}/{selected_speed_subfolder}")
csv_files = [f["name"] for f in files if f["name"].endswith(".csv")]
if csv_files:
    selected_file = st.selectbox("削除したいCSVファイルを選択", csv_files, key="delete_speed")
    if st.button("このCSVファイルを削除", key="btn_delete_speed"):
        supabase.storage.from_(BUCKET_NAME).remove(f"{selected_file}")
        st.success(f"{selected_file} を削除しました。")
        st.experimental_rerun()



# ピーク検出とデータ整形のためのパラメータ
peak_threshold = 1000  # ピーク検出の閾値
desired_length = 100
change_threshold = 10
value_threshold = 500

split_list = []  # データを格納するリスト
split_list_acc = []  # 加速度データを格納するリスト
split_list_gyro = []  # ジャイロデータ（X, Y, Z合成）を格納するリスト

#極端に長いデータや短いデータを補完する関数
def pad_or_interpolate(series, target_length=100):
    series = np.array(series, dtype=np.float64)

    if len(series) >= target_length:
        return series  # すでに十分な長さがある場合はそのまま返す

    if len(series) == 1:
        return np.full(target_length, series[0])  # 1点しかない場合は同じ値で埋める

    # 線形補間
    x_old = np.linspace(0, 1, len(series))
    x_new = np.linspace(0, 1, target_length)
    f = interp1d(x_old, series, kind='linear', fill_value='extrapolate')
    return f(x_new)

#スプライン補完,移動平均＋補完
def pad_or_interpolate_smooth(series, target_length=100):
    series = np.array(series, dtype=np.float64)

    if len(series) >= target_length:
        return series[:target_length]

    if len(series) == 1:
        return np.full(target_length, series[0])

    x_old = np.linspace(0, 1, len(series))
    x_new = np.linspace(0, 1, target_length)

    # スプライン補間でなめらかなカーブを作る
    spline = UnivariateSpline(x_old, series, k=3, s=0.5)  # `s`はスムージング係数
    return spline(x_new)


# フォルダ名を数値でソートするための関数
def extract_number(folder_name):
    match = re.search(r'\d+', folder_name)
    if match:
        return int(match.group())
    return float('inf')

# ピーク後のセグメント終了位置を検出する関数
def find_end_of_segment_by_change_and_value(data_segment, peak_index):
    for j in range(peak_index, len(data_segment) - 1):
        if abs(data_segment[j + 1] - data_segment[j]) < change_threshold and data_segment[j] < value_threshold:
            return j + 1
    return len(data_segment)

# 加速度データのリストを作成し、指定の長さに揃える関数
def make_split_list_acc(one_person_data):
    for i, data_segment in enumerate(one_person_data):
        if len(data_segment) >= 3:
            peak_index = -1
            for j in range(1, len(data_segment) - 1):
                if data_segment[j] > peak_threshold and data_segment[j] > data_segment[j - 1] and data_segment[j] > data_segment[j + 1]:
                    peak_index = j
                    print(f"Peak detected at {peak_index} in segment {i+1}")
                    break
                    
            
            if peak_index != -1:
            #if 0 == 0:
                end_index = find_end_of_segment_by_change_and_value(data_segment, peak_index)
                trimmed_data = data_segment[:end_index]
                split_list_acc.append(trimmed_data)
            else:
                split_list_acc.append(one_person_data[i])#欠損値は欠損値のまま
                #split_list_acc.append(0) #0埋め
                print(f"Peak cannot detected in segment {i+1}")
                
        else:
            print(f"Peak cannot detected in segment {i+1}")

            
def make_split_list_gyro(one_person_data):
    """
    ジャイロデータ（X, Y, Z合成）をリストにし、指定の長さに揃える。
    """
    for i, data_segment in enumerate(one_person_data):
        combined_gyro = data_segment

        # 5000以上で、傾きが1以下のデータを取り除く
        valid_data = []
        for k in range(1, len(combined_gyro)):
            value = combined_gyro[k]
            prev_value = combined_gyro[k - 1]
            slope = (value - prev_value)  # 傾きを計算
            if not (value >= 5000 and slope <= 500):
                valid_data.append(value)

        trimmed_data = np.array(valid_data)

        # データがdesired_lengthより短い場合、前に同じ値を追加して長さを揃える
        if len(trimmed_data) < desired_length:
            if len(trimmed_data) > 0:
                padding_value = trimmed_data[0]  # 最初の値を使う
            else:
                padding_value = 0  # データが空なら0を使う
            padding = [padding_value] * (desired_length - len(trimmed_data))
            trimmed_data = np.array(padding + list(trimmed_data))  # 前にパディングを追加

        # データがdesired_lengthより長い場合、後ろの部分を削除して長さを揃える
        if len(trimmed_data) > desired_length:
            trimmed_data = trimmed_data[:desired_length]  # desired_lengthに合わせて後ろをカット

        # 整形したデータをsplit_list_gyroに格納
        split_list_gyro.append(trimmed_data)
        # plt.plot(trimmed_data)
        # plt.show()

# データの形状を確認
if split_list_acc and split_list_gyro:
    print("split_list_accの形状:", len(split_list_acc), "x", len(split_list_acc[0]))
    print("split_list_gyroの形状:", len(split_list_gyro), "x", len(split_list_gyro[0]))
else:
    print("split_list_accまたはsplit_list_gyroが空です")

# フォルダを数値順にソートして処理
sorted_folders = sorted(folders, key=extract_number)

for folder in sorted_folders:
    folder_path = f"{path}/{folder}"
    files = supabase.storage.from_(BUCKET_NAME).list(folder_path, {"recursive": True})
    csv_files = [f["name"] for f in files if f["name"].endswith(".csv")]

    for csv_file in csv_files:
        print(csv_file)
        res = supabase.storage.from_(BUCKET_NAME).download(csv_file)
        df = pd.read_csv(io.BytesIO(res), encoding='utf-8')

        list_num = -1
        one_person_data_acc = []
        one_person_data_gyro = []
        row_num = 0
        count = 0

        for index, row in df.iterrows():
            if not pd.isna(row.iloc[5]) and row.iloc[4] != 0:
                if count == 0:
                    list_num += 1
                    one_person_data_acc.append([])
                    one_person_data_gyro.append([])
                if row_num < 200:
                    count = 1
                    row_num += 1
                    acc_data = float(row.iloc[26])**2 + float(row.iloc[27])**2 + float(row.iloc[28])**2
                    gyro_data = float(row.iloc[29])**2 + float(row.iloc[30])**2 + float(row.iloc[31])**2
                    one_person_data_acc[list_num].append(acc_data)
                    one_person_data_gyro[list_num].append(gyro_data)
            elif row.iloc[4] == 0:
                count = 0
                row_num = 0

        print(len(one_person_data_acc))
        make_split_list_acc(one_person_data_acc)
        make_split_list_gyro(one_person_data_gyro)
        print(len(split_list_acc))


# DTW距離を計算する関数
def dtw_distance(series1, series2):
    distance, _ = fastdtw(series1, series2)
    return distance

# ユークリッド距離を計算する関数
def euclidean_distance(series1, series2):
    if len(series1) != len(series2):
        raise ValueError("The series must have the same length for Euclidean distance.")
    return np.sqrt(np.sum((np.array(series1) - np.array(series2)) ** 2))

# コサイン類似度を計算する関数
def cosine_similarity(series1, series2):
    series1 = np.array(series1)
    series2 = np.array(series2)
    dot_product = np.dot(series1, series2)
    norm_series1 = np.linalg.norm(series1)
    norm_series2 = np.linalg.norm(series2)
    return dot_product / (norm_series1 * norm_series2)

# ピアソン相関係数を計算する関数
def pearson_correlation(series1, series2):
    series1 = np.array(series1)
    series2 = np.array(series2)
    return np.corrcoef(series1, series2)[0, 1]

# ピークを検出する関数
def find_peak(data_segment):
    return np.argmax(data_segment)




# セグメントを比較し、距離行列を計算する関数
def compare_segments(split_list, distance_type='DTW', mode='before_peak'):
    n = len(split_list)
    print(n)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        peak_i = find_peak(split_list[i])

        for j in range(n):
            if i != j:
                try:
                    peak_j = find_peak(split_list[j])
                    if mode == 'after_peak':
                        # ピーク以降のデータを比較
                        if peak_i > 0 and peak_j > 0:
                            series_i = split_list[i][peak_i:]
                            series_j = split_list[j][peak_j:]

                        # データの長さを短い方に合わせる
                        min_len = min(len(series_i), len(series_j))
                        series_i = series_i[:min_len]  # 前の部分を切り取る
                        series_j = series_j[:min_len]
                    
                    elif mode == 'before_peak':
                        # ピークまでのデータを比較
                        series_i = split_list[i][:peak_i]
                        series_j = split_list[j][:peak_j]

                        # データの長さを短い方に合わせる
                        min_len = min(len(series_i), len(series_j))
                        series_i = series_i[-min_len:]  # 後ろの部分を切り取る
                        series_j = series_j[-min_len:]

                        
                        #角速度データの時のみ
                        #series_i = series_i[-10:]  # ピーク前の最後の10サンプル
                        #series_j = series_j[-10:]  # ピーク前の最後の10サンプル
                        
                        
                    elif mode == 'entire_series':
                        # 全体のデータを比較
                        series_i = split_list[i]
                        series_j = split_list[j]

                        # データの長さを短い方に合わせる
                        min_len = min(len(series_i), len(series_j))
                        series_i = series_i[-min_len:]  # 頭の部分を切り取る
                        series_j = series_j[-min_len:]

                    distance_matrix[i, j] = dtw_distance(series_i, series_j)

                except Exception as e:
                    # ピーク検出に失敗した場合や計算エラーがあった場合
                    print(f"Error in distance calculation between {i} and {j}: {e}")
                    # DTW距離を0に設定
                    distance_matrix[i, j] = 0
    #dtw_matrix_plot(distance_matrix)
    print(len(distance_matrix))
    return distance_matrix

# 距離行列のヒートマップをプロットする関数
def plot_heatmap(matrix, title, colormap='hot', vmin=None, vmax=None):
    st.subheader("ヒートマップ")
    fig = go.Figure(
        data=go.Heatmap(z=matrix, colorscale=colormap, zmin=vmin, zmax=vmax),
    )
    fig.update_layout(
        title=title,
        xaxis_title="Segment Index",
        yaxis_title="Segment Index",
        yaxis=dict(autorange="reversed"),  # Y軸を反転
        width=600,  # 正方形にするための幅
        height=750,  # 正方形にするための高さ
    )
    return fig

#棒グラフを表示する関数
def show_bar_chart(matrix):
    st.subheader("DTW距離")
    # 列ごとに平均を計算（縦方向の平均）
    column_means = np.mean(matrix, axis=0)
    num_cols = matrix.shape[1]  # ← ここで列数を取得！

    # 折れ線グラフとして表示
    #df_line = pd.DataFrame({'Column Index': np.arange(num_cols), 'Mean': column_means})
    #df_line.set_index('Column Index', inplace=True)
    #st.line_chart(df_line)

    # DataFrame に変換
    df_bar = pd.DataFrame({'Column Index': np.arange(num_cols), 'Mean': column_means})
    df_bar.set_index('Column Index', inplace=True)

    # 棒グラフとして表示
    st.bar_chart(df_bar)


@st.cache_resource
def analyze_segments_cached(split_list, distance_type='DTW', mode='after_peak'):
    return analyze_segments(split_list, distance_type, mode)

# 距離行列と滑らかさの評価結果を取得
def analyze_segments(split_list, distance_type='DTW', mode='after_peak'):
    distance_matrix = compare_segments(split_list, distance_type, mode)
    return distance_matrix

# 波形を表示する関数
def plot_waveforms(series1, series2, label1, label2):
    plt.figure(figsize=(10, 8))
    plt.plot(series1, color="blue",  label=label1)
    plt.plot(series2, color="green",label=label2)
    plt.legend()
    plt.title('Waveforms')
    plt.xlabel('Index')
    plt.ylabel('Value')
    st.pyplot(plt.gcf())
    plot_dtw_matrix(series1, series2, label1, label2)

#パス行列を描画する関数
def plot_dtw_matrix(series1, series2, label1, label2):
    n, m = len(series1), len(series2)
    path = np.array(dtw.warping_path(series1, series2))

    #fig = plt.figure(figsize=(6, 6))
    fig = plt.figure(figsize=(max(6, m/10), max(6, n/10)))  # データ長に応じてサイズ調整

    #行列の上に Time Series 2 をプロット
    ax_top = plt.subplot2grid((4, 4), (0, 1), colspan=3)
    ax_top.plot(series2, color="green", linewidth=2)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_xlim(-0.5, m-0.5)
    
    #行列の左に Time Series 1 をプロット
    ax_left = plt.subplot2grid((4, 4), (1, 0), rowspan=3)
    ax_left.plot(series1, range(len(series1)), color="blue", linewidth=2)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.set_ylim(-0.5, n-0.5)
    ax_left.invert_xaxis()  # 左を大きい値にするため反転

    #DTW行列とパスを描画
    ax_matrix = plt.subplot2grid((4, 4), (1, 1), rowspan=3, colspan=3)
    ax_matrix.set_xticks(np.arange(m+1)-0.5, minor=True)
    ax_matrix.set_yticks(np.arange(n+1)-0.5, minor=True)
    ax_matrix.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax_matrix.tick_params(which="both", bottom=False, left=False, labelbottom=True, labelleft=True)
    ax_matrix.scatter(path[:, 1], path[:, 0], color='red', s=50)

    #軸ラベル
    ax_matrix.set_xlabel("series2")
    ax_matrix.set_ylabel("series1")
    ax_matrix.set_title("DTW Path Matrix")

    #plt.tight_layout()
    #plt.show()
    st.pyplot(plt.gcf())

# 球速回転数CSVファイル読み込み関数
@st.cache_data
def load_pitch_data(df_new):
    df_new.columns = ['球速(km/h)', '回転数(rpm)', '縦変化量(cm)', '横変化量(cm)']

    # 各列ごとに配列化
    speed_array = df_new['球速(km/h)'].to_numpy()
    spin_array = df_new['回転数(rpm)'].to_numpy()
    vertical_movement_array = df_new['縦変化量(cm)'].to_numpy()
    horizontal_movement_array = df_new['横変化量(cm)'].to_numpy()

    return speed_array, spin_array, vertical_movement_array, horizontal_movement_array



# 実行関数
def run_analysis_and_plot(split_list, distance_type='DTW', mode='before_peak', df_speed = None, vmin=None, vmax=None,):
    distance_matrix = analyze_segments_cached(split_list, distance_type, mode)
    #print(distance_matrix)
    if vmin is None:
        vmin = np.min(distance_matrix)
    if vmax is None:
        vmax = np.max(distance_matrix)
    fig = plot_heatmap(distance_matrix, f"Distance Matrix - Mode: {mode}", vmin=vmin, vmax=vmax)
    st.plotly_chart(fig)
    show_bar_chart(distance_matrix)

    #キャッシュの強制削除で,追加データの読み込み
    if st.button("データを再読み込み"):
        st.cache_data.clear()

    
    # セル手動選択用のUI
    st.subheader("比較する投球を選択してください")
    idx1 = st.selectbox("series1のインデックス", list(range(len(split_list_acc))), index=0)
    idx2 = st.selectbox("series2のインデックス", list(range(len(split_list_acc))), index=1)

    # 選択に応じて波形を表示
    if st.button("波形とDTWパスを表示"):
        series_i = split_list[idx1]
        series_j = split_list[idx2]
        plot_waveforms(series_i, series_j, f"{idx1}", f"{idx2}")




    st.subheader("データフレーム表示")
    st.dataframe(df_speed)

    # 棒グラフで球速表示
    st.subheader("球速")
    st.line_chart(pd.DataFrame({'球速': speed_array}))

    # 棒グラフで回転数表示
    st.subheader("回転数")
    st.line_chart(pd.DataFrame({'回転数': spin_array}))

df_speed = pd.DataFrame()
speed_array = []
spin_array = []
vertical_movement_array = []
horizontal_movement_array = []


if not speed_folders:
    st.warning("指定フォルダが見つかりません。")
else:
    sorted_speed_folders = sorted(speed_folders, key=extract_number)
    for folder in sorted_speed_folders:
        folder_path = f"{speed_path}/{folder}"
        files = supabase.storage.from_(BUCKET_NAME).list(folder_path)
        csv_files = [f["name"] for f in files if f["name"].endswith(".csv")]

        if not csv_files:
            st.warning("指定フォルダにCSVファイルが見つかりません。")
        else:
            for csv_file in csv_files:
                res = supabase.storage.from_(BUCKET_NAME).download(f"{folder_path}/{csv_file}")
                df_new = pd.read_csv(io.BytesIO(res), usecols=[12, 13, 16, 17], encoding='shift_jis')

                s_new, sp_new, vm_new, hm_new = load_pitch_data(df_new)
                df_speed = pd.concat([df_speed, df_new], ignore_index=True)
                speed_array += list(s_new)
                spin_array += list(sp_new)
                vertical_movement_array += list(vm_new)
                horizontal_movement_array += list(hm_new)

# 実行
split_list = split_list_acc #または　split_list_gyro
run_analysis_and_plot(split_list, distance_type='DTW', mode='before_peak', df_speed = df_speed)
