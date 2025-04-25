def extract_hr(block):
    """
    Trích xuất giá trị HR@10 từ một block.
    Block mẫu:
    ['lr=0.005, dropout=0.1, neg_weight=0.001',
     'Final HR@10: 0.3356953055801594  NDCG@10: 0.18038808828983668']
    """
    try:
        # Tách lấy phần sau "HR@10:" và lấy số đầu tiên
        hr_str = block[1].split("HR@10:")[1].split()[0]
        return float(hr_str)
    except Exception as e:
        return 0.0

# Đọc file kết quả
with open('ENSFM_hyperparam_results.txt', 'r') as f:
    content = f.read()

# Tách nội dung file thành các block dựa trên dòng trống
blocks = [block.strip().split('\n') for block in content.strip().split('\n\n') if block.strip()]

# Sắp xếp các block theo giá trị HR@10 giảm dần
sorted_blocks = sorted(blocks, key=extract_hr, reverse=True)

# Ghi kết quả đã sắp xếp vào file mới
with open('ENSFM_hyperparam_results_sorted.txt', 'w') as f:
    for block in sorted_blocks:
        f.write('\n'.join(block) + "\n\n")

print("Đã lưu kết quả sắp xếp vào file ENSFM_hyperparam_results_sorted.txt")
