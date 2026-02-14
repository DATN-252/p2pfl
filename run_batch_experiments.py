import os
import time
import subprocess
from datetime import datetime

def run_experiments(config_dir="batch_configs"):
    # 1. Kiểm tra thư mục chứa configs
    if not os.path.exists(config_dir):
        print(f"❌ Thư mục '{config_dir}' không tồn tại. Vui lòng tạo và bỏ các file .yaml vào đó.")
        return

    # 2. Lấy danh sách các file yaml
    config_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml') or f.endswith('.yml')]
    config_files.sort()

    if not config_files:
        print(f"⚠️ Không tìm thấy file cấu hình nào trong '{config_dir}'.")
        return

    print(f"🚀 Tìm thấy {len(config_files)} thí nghiệm. Bắt đầu chạy...")

    for i, config_file in enumerate(config_files):
        config_path = os.path.join(config_dir, config_file)
        print("\n" + "="*60)
        print(f"🧪 [{i+1}/{len(config_files)}] Đang chạy: {config_file}")
        print(f"⏰ Bắt đầu lúc: {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)

        try:
            # Sử dụng lệnh p2pfl run đã có sẵn trong dự án
            # Chúng ta dùng subprocess.run để đợi thí nghiệm này xong mới chạy cái tiếp theo
            result = subprocess.run(
                ["p2pfl", "run", config_path],
                check=True,
                text=True
            )
            print(f"✅ Hoàn thành thí nghiệm: {config_file}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Lỗi khi chạy {config_file}: {e}")
        except KeyboardInterrupt:
            print("\n🛑 Đã dừng bởi người dùng. Thoát...")
            break

    print("\n" + "="*60)
    print("🎉 TẤT CẢ THÍ NGHIỆM ĐÃ HOÀN TẤT!")
    print("="*60)

if __name__ == "__main__":
    # Bạn có thể đổi tên thư mục chứa các file yaml ở đây
    run_experiments("batch_configs")
