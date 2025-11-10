import json
import sys
import argparse
from typing import List, Dict, Any, Union

def find_kernel_durations(
    file_path: str, 
    kernel_name_substring: str = "paged_attention_ll4mi_QKV_mfma16_kernel"
) -> List[float]:
    """
    從 Chrome Tracing JSON 檔案中讀取數據，篩選出特定 Kernel 的執行持續時間 (dur)。

    Args:
        file_path (str): JSON 檔案的路徑。
        kernel_name_substring (str): 要查找的 Kernel 名稱子字符串。

    Returns:
        List[float]: 包含所有匹配 Kernel 的持續時間列表。
    """
    try:
        # 1. 讀取 JSON 檔案
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{file_path}'。", file=sys.stderr)
        return []
    except json.JSONDecodeError:
        print(f"錯誤：檔案 '{file_path}' 不是有效的 JSON 格式。", file=sys.stderr)
        return []

    if not isinstance(data, dict) or 'traceEvents' not in data:
        print("警告：JSON 結構不包含 'traceEvents' 鍵或格式錯誤。", file=sys.stderr)
        return []
        
    durations = []
    
    # 2. 遍歷 traceEvents 列表
    duration_500=0
    for event in data['traceEvents']:
        # 3. 篩選條件：'X' 事件, 必須有 'name', 'dur' 字段, 且名稱匹配
        if (event.get('ph') == 'X' and 
            'name' in event and 
            'dur' in event and
            kernel_name_substring in event['name']):
            
            # 4. 記錄持續時間
            try:
                # 確保轉換為浮點數
                duration = float(event['dur'])
                durations.append(duration)
                if duration>500:
                    duration_500+=1
                    print(f"duration={duration}, duration_500={duration_500}")
            except ValueError:
                print(f"警告：發現非數值 'dur' 字段: {event['dur']}，已跳過。", file=sys.stderr)


    # 5. 輸出結果摘要
    print(f"--- 分析結果摘要 ({kernel_name_substring}) ---")
    print(f"找到 {len(durations)} 個匹配的 Kernel 執行記錄。")
    if durations:
        print(f"最短持續時間 (min dur): {min(durations):.4f}")
        print(f"最長持續時間 (max dur): {max(durations):.4f}")
        print(f"平均持續時間 (avg dur): {sum(durations) / len(durations):.4f}")
    print("----------------------------------------")
    
    return durations

# --- 使用 argparse 處理命令列參數 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="分析 Chrome Tracing JSON 檔案，提取特定 Kernel 的持續時間。",
        # 預設 target_name
        epilog="預設 Kernel 子字符串為: paged_attention_ll4mi_QKV_mfma16_kernel"
    )
    
    # -i 或 --input 參數，用於指定檔案路徑 (必需)
    parser.add_argument(
        '-i', '--input', 
        type=str, 
        required=True,
        help="輸入的 Chrome Tracing JSON 檔案路徑。"
    )
    
    # -k 或 --kernel 參數，用於指定要查找的 Kernel 名稱子字符串 (可選)
    parser.add_argument(
        '-k', '--kernel', 
        type=str, 
        default="paged_attention_ll4mi_QKV_mfma16_kernel",
        help="要篩選的 Kernel 名稱子字符串。"
    )

    args = parser.parse_args()
    
    file_path = args.input
    target_name = args.kernel
    
    # 執行分析
    all_durations = find_kernel_durations(file_path, target_name)

    # 打印所有找到的持續時間
    if all_durations:
        print(f"\n所有 {target_name} 的持續時間 (dur):")
        # 為了清晰，只打印前 10 個
        print(all_durations[:10], "..." if len(all_durations) > 10 else "")