#!/usr/bin/env python3
"""
整理 SR 验证数据集为统一格式

目标格式（和 DIV2K 一致）:
  Data/{Dataset}/HR/{name}.png
  Data/{Dataset}/LR_X4/{name}.png

支持的数据集:
  - RealSR V3 (Canon + Nikon)
  - DRealSR
  - Urban100

用法:
  python reorganize_datasets.py --all
  python reorganize_datasets.py --realsr --drealsr --urban100
"""

import os
import shutil
import argparse
import zipfile
from pathlib import Path


def reorganize_realsr_v3(src_base, dst_base, scale=4, copy=True):
    """
    整理 RealSR V3 数据集
    
    原始格式:
      RealSR(V3)/Canon/Test/4/Canon_001_HR.png
      RealSR(V3)/Canon/Test/4/Canon_001_LR4.png
      RealSR(V3)/Nikon/Test/4/Nikon_001_HR.png
      RealSR(V3)/Nikon/Test/4/Nikon_001_LR4.png
    
    目标格式:
      HR/Canon_001.png, Nikon_001.png, ...
      LR_X4/Canon_001.png, Nikon_001.png, ...
    """
    src_base = Path(src_base)
    dst_base = Path(dst_base)
    
    # 查找 RealSR(V3) 目录
    realsr_dir = None
    for d in src_base.rglob('*'):
        if d.is_dir() and 'RealSR' in d.name and 'V3' in d.name:
            realsr_dir = d
            break
    
    if realsr_dir is None:
        # 尝试直接使用 src_base
        if (src_base / 'Canon').exists() or (src_base / 'Nikon').exists():
            realsr_dir = src_base
        else:
            raise FileNotFoundError(f"找不到 RealSR(V3) 目录 in {src_base}")
    
    print(f"[RealSR V3] 源目录: {realsr_dir}")
    
    # 创建目标目录
    hr_dst = dst_base / 'HR'
    lr_dst = dst_base / f'LR_X{scale}'
    hr_dst.mkdir(parents=True, exist_ok=True)
    lr_dst.mkdir(parents=True, exist_ok=True)
    
    op = shutil.copy2 if copy else shutil.move
    count = 0
    
    # 处理 Canon 和 Nikon
    for camera in ['Canon', 'Nikon']:
        test_dir = realsr_dir / camera / 'Test' / str(scale)
        if not test_dir.exists():
            print(f"  警告: {test_dir} 不存在，跳过")
            continue
        
        hr_files = sorted([f for f in test_dir.iterdir() if '_HR' in f.name])
        print(f"  {camera}: 找到 {len(hr_files)} 张 HR 图片")
        
        for hr_file in hr_files:
            # Canon_001_HR.png -> Canon_001
            base_name = hr_file.stem.replace('_HR', '')
            
            # 找对应的 LR
            lr_name = f"{base_name}_LR{scale}{hr_file.suffix}"
            lr_file = test_dir / lr_name
            
            if not lr_file.exists():
                print(f"    警告: 找不到 LR 文件 {lr_name}")
                continue
            
            # 复制
            new_name = f"{base_name}{hr_file.suffix}"
            op(hr_file, hr_dst / new_name)
            op(lr_file, lr_dst / new_name)
            count += 1
    
    print(f"[RealSR V3] 完成! 共 {count} 对图片")
    print(f"  HR: {hr_dst}")
    print(f"  LR: {lr_dst}")
    return count


def reorganize_drealsr(src_base, dst_base, scale=4, copy=True):
    """
    整理 DRealSR 数据集
    
    原始格式:
      DRealSR/DRealSR/x4/Test_x4/Test_x4/test_HR/Canon_10_x4.png
      DRealSR/DRealSR/x4/Test_x4/Test_x4/test_LR/Canon_10_x1.png
    
    目标格式:
      HR/Canon_10.png
      LR_X4/Canon_10.png
    """
    src_base = Path(src_base)
    dst_base = Path(dst_base)
    
    # 查找 test_HR 目录
    hr_src = None
    lr_src = None
    
    for d in src_base.rglob('test_HR'):
        if d.is_dir():
            hr_src = d
            break
    
    for d in src_base.rglob('test_LR'):
        if d.is_dir():
            lr_src = d
            break
    
    if hr_src is None or lr_src is None:
        raise FileNotFoundError(f"找不到 test_HR/test_LR 目录 in {src_base}")
    
    print(f"[DRealSR] HR 源目录: {hr_src}")
    print(f"[DRealSR] LR 源目录: {lr_src}")
    
    # 创建目标目录
    hr_dst = dst_base / 'HR'
    lr_dst = dst_base / f'LR_X{scale}'
    hr_dst.mkdir(parents=True, exist_ok=True)
    lr_dst.mkdir(parents=True, exist_ok=True)
    
    op = shutil.copy2 if copy else shutil.move
    
    # 获取 HR 文件列表
    hr_files = sorted([f for f in hr_src.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    print(f"[DRealSR] 找到 {len(hr_files)} 张 HR 图片")
    
    count = 0
    for hr_file in hr_files:
        # DRealSR 命名: Canon_10_x4.png (HR) -> Canon_10_x1.png (LR)
        # 提取 base name: Canon_10
        base_name = hr_file.stem
        
        # 去掉 _x4 后缀得到 base
        if '_x4' in base_name:
            base = base_name.replace('_x4', '')
        elif '_x' in base_name:
            # 处理其他 scale
            import re
            base = re.sub(r'_x\d+$', '', base_name)
        else:
            base = base_name
        
        # LR 文件名: Canon_10_x1.png
        lr_name = f"{base}_x1{hr_file.suffix}"
        lr_file = lr_src / lr_name
        
        if not lr_file.exists():
            # 尝试其他模式
            for f in lr_src.iterdir():
                if f.stem.startswith(base):
                    lr_file = f
                    break
        
        if not lr_file.exists():
            print(f"  警告: 找不到 {hr_file.name} 对应的 LR 文件 ({lr_name})")
            continue
        
        # 统一命名
        new_name = f"{base}{hr_file.suffix}"
        
        op(hr_file, hr_dst / new_name)
        op(lr_file, lr_dst / new_name)
        count += 1
    
    print(f"[DRealSR] 完成! 共 {count} 对图片")
    print(f"  HR: {hr_dst}")
    print(f"  LR: {lr_dst}")
    return count


def reorganize_urban100(src_base, dst_base, scale=4, copy=True):
    """
    整理 Urban100 数据集
    
    原始格式:
      X4_Urban100/X4/HIGH_x4_URban100/img_001_SRF_4_HR.png
      X4_Urban100/X4/LOW_x4_URban100/img_001_SRF_4_LR.png
    
    目标格式:
      HR/img_001.png
      LR_X4/img_001.png
    """
    src_base = Path(src_base)
    dst_base = Path(dst_base)
    
    # 1) 优先使用已经整理好的标准目录，避免误选 X2 数据。
    hr_src = src_base / 'HR'
    lr_src = src_base / f'LR_X{scale}'
    if not (hr_src.is_dir() and lr_src.is_dir()):
        # 2) 回退到原始 Urban100 目录结构，按 scale 精确匹配。
        hr_src = None
        lr_src = None
        high_candidates = []
        low_candidates = []
        scale_token = f"X{scale}"

        for d in src_base.rglob('*'):
            if not d.is_dir():
                continue
            name_upper = d.name.upper()
            path_upper = str(d).upper()
            if 'HIGH' in name_upper:
                high_candidates.append(d)
            elif 'LOW' in name_upper:
                low_candidates.append(d)

        def _pick_scale_dir(candidates, kind):
            # Strong match: both path and folder name include target scale token.
            strong = [c for c in candidates if scale_token in str(c).upper() and scale_token in c.name.upper()]
            if strong:
                return sorted(strong)[0]
            # Medium match: path includes target scale token.
            medium = [c for c in candidates if scale_token in str(c).upper()]
            if medium:
                return sorted(medium)[0]
            # Fallback: deterministic first candidate (for backward compatibility).
            if candidates:
                picked = sorted(candidates)[0]
                print(f"  警告: 未找到包含 {scale_token} 的 {kind} 目录，回退到 {picked}")
                return picked
            return None

        hr_src = _pick_scale_dir(high_candidates, "HIGH")
        lr_src = _pick_scale_dir(low_candidates, "LOW")

    if hr_src is None or lr_src is None:
        raise FileNotFoundError(f"找不到 Urban100 对应的 HR/LR 源目录 in {src_base} (scale={scale})")
    
    print(f"[Urban100] HR 源目录: {hr_src}")
    print(f"[Urban100] LR 源目录: {lr_src}")
    
    # 创建目标目录
    hr_dst = dst_base / 'HR'
    lr_dst = dst_base / f'LR_X{scale}'
    hr_dst.mkdir(parents=True, exist_ok=True)
    lr_dst.mkdir(parents=True, exist_ok=True)
    
    op = shutil.copy2 if copy else shutil.move
    
    # 处理 HR 图片
    hr_files = sorted([f for f in hr_src.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    print(f"[Urban100] 找到 {len(hr_files)} 张 HR 图片")
    
    count = 0
    for hr_file in hr_files:
        # img_001_SRF_4_HR.png -> img_001
        name = hr_file.stem
        if '_SRF_' in name:
            base = name.split('_SRF_')[0]
        elif '_HR' in name:
            base = name.replace('_HR', '')
        else:
            base = name
        
        # 找对应的 LR
        lr_name = name.replace('_HR', '_LR') + hr_file.suffix
        lr_file = lr_src / lr_name
        
        if not lr_file.exists():
            # 尝试其他模式
            for f in lr_src.iterdir():
                if base in f.stem and 'LR' in f.stem:
                    lr_file = f
                    break
        
        if not lr_file.exists():
            print(f"  警告: 找不到 {hr_file.name} 对应的 LR 文件")
            continue
        
        new_name = f"{base}{hr_file.suffix}"
        op(hr_file, hr_dst / new_name)
        op(lr_file, lr_dst / new_name)
        count += 1
    
    print(f"[Urban100] 完成! 共 {count} 对图片")
    print(f"  HR: {hr_dst}")
    print(f"  LR: {lr_dst}")
    return count


def main():
    parser = argparse.ArgumentParser(description='整理 SR 验证数据集')
    
    # 数据集选择
    parser.add_argument('--all', action='store_true', help='处理所有数据集')
    parser.add_argument('--realsr', action='store_true', help='处理 RealSR V3')
    parser.add_argument('--drealsr', action='store_true', help='处理 DRealSR')
    parser.add_argument('--urban100', action='store_true', help='处理 Urban100')
    
    # 路径配置
    parser.add_argument('--data_root', type=str, default='Data',
                        help='数据根目录 (默认: Data)')
    parser.add_argument('--scale', type=int, default=4, help='放大倍数 (默认: 4)')
    parser.add_argument('--move', action='store_true', help='移动文件而不是复制')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    if args.all:
        args.realsr = args.drealsr = args.urban100 = True
    
    if not any([args.realsr, args.drealsr, args.urban100]):
        print("请指定要处理的数据集: --all, --realsr, --drealsr, --urban100")
        return
    
    print("=" * 60)
    print("SR 数据集整理工具")
    print("=" * 60)
    print(f"数据根目录: {data_root}")
    print(f"放大倍数: {args.scale}")
    print(f"操作模式: {'移动' if args.move else '复制'}")
    print("=" * 60)
    
    results = {}
    
    # RealSR V3
    if args.realsr:
        try:
            src = data_root / 'RealSR'
            dst = data_root / 'RealSR_test'
            count = reorganize_realsr_v3(src, dst, args.scale, copy=not args.move)
            results['RealSR'] = count
        except Exception as e:
            print(f"[RealSR V3] 错误: {e}")
            results['RealSR'] = 0
        print()
    
    # DRealSR
    if args.drealsr:
        try:
            src = data_root / 'DRealSR'
            dst = data_root / 'DRealSR_test'
            count = reorganize_drealsr(src, dst, args.scale, copy=not args.move)
            results['DRealSR'] = count
        except Exception as e:
            print(f"[DRealSR] 错误: {e}")
            results['DRealSR'] = 0
        print()
    
    # Urban100
    if args.urban100:
        try:
            src = data_root / 'Urban100'
            dst = data_root / 'Urban100_test'
            count = reorganize_urban100(src, dst, args.scale, copy=not args.move)
            results['Urban100'] = count
        except Exception as e:
            print(f"[Urban100] 错误: {e}")
            results['Urban100'] = 0
        print()
    
    # 汇总
    print("=" * 60)
    print("汇总")
    print("=" * 60)
    for name, count in results.items():
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {name}: {count} 对图片")
    
    print("\n评估命令示例:")
    print("-" * 60)
    
    if results.get('RealSR', 0) > 0:
        print(f"""
# RealSR
python evaluate_clear_control.py \\
    --checkpoint ./checkpoints/xxx/best_model.pt \\
    --hr_dir {data_root}/RealSR_test/HR \\
    --lr_dir {data_root}/RealSR_test/LR_X{args.scale} \\
    --dataset RealSR
""")
    
    if results.get('DRealSR', 0) > 0:
        print(f"""
# DRealSR  
python evaluate_clear_control.py \\
    --checkpoint ./checkpoints/xxx/best_model.pt \\
    --hr_dir {data_root}/DRealSR_test/HR \\
    --lr_dir {data_root}/DRealSR_test/LR_X{args.scale} \\
    --dataset DRealSR
""")
    
    if results.get('Urban100', 0) > 0:
        print(f"""
# Urban100
python evaluate_clear_control.py \\
    --checkpoint ./checkpoints/xxx/best_model.pt \\
    --hr_dir {data_root}/Urban100_test/HR \\
    --lr_dir {data_root}/Urban100_test/LR_X{args.scale} \\
    --dataset Urban100
""")


if __name__ == '__main__':
    main()
