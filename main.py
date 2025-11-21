"""
深度学习模型性能测试 - 主入口
包含四个测试任务：
1. 对比3个模型的显存占用差异
2. 找出模型能用的最大批大小
3. 找到让GPU使用率最平稳的worker数量
4. 用Profiler找瓶颈（进阶）
"""

import sys


def print_menu():
    """打印任务选择菜单"""
    print("\n" + "="*80)
    print(" " * 20 + "深度学习模型性能测试")
    print("="*80)
    print("\n请选择要执行的任务：\n")
    print("  [1] 任务1：对比3个模型的显存占用差异")
    print("      - 测试 SimpleCNN、ResNet18、MobileNetV2")
    print("      - 批大小固定为32")
    print("      - 记录参数量、显存占用、训练时间\n")

    print("  [2] 任务2：找出模型能用的最大批大小")
    print("      - 使用 MobileNetV2 模型")
    print("      - 批大小从8开始翻倍测试")
    print("      - 记录显存占用直到OOM")
    print("      - 生成显存占用对比图\n")

    print("  [3] 任务3：找到让GPU使用率最平稳的worker数量")
    print("      - 使用 MobileNetV2 模型，批大小64")
    print("      - 测试 num_workers = 0, 2, 4, 8")
    print("      - 实时监控GPU利用率")
    print("      - 生成GPU利用率对比图\n")

    print("  [4] 任务4：用Profiler找瓶颈（进阶）")
    print("      - 使用 PyTorch Profiler 分析训练过程")
    print("      - 找出最耗时的操作")
    print("      - 生成Chrome trace文件和TensorBoard日志")
    print("      - 分析CPU/GPU时间分布\n")

    print("  [0] 退出程序\n")
    print("="*80)


def main():
    """主函数"""
    while True:
        print_menu()

        try:
            choice = input("请输入任务编号 [0-4]: ").strip()

            if choice == "0":
                print("\n退出程序。再见！\n")
                sys.exit(0)

            elif choice == "1":
                print("\n正在启动任务1...")
                from task1_compare_models import run_task1
                run_task1()

            elif choice == "2":
                print("\n正在启动任务2...")
                from task2_max_batch_size import main as run_task2
                run_task2()

            elif choice == "3":
                print("\n正在启动任务3...")
                from task3_gpu_workers import main as run_task3
                run_task3()

            elif choice == "4":
                print("\n正在启动任务4...")
                from task4_profiler import run_task4
                run_task4()

            else:
                print("\n❌ 无效的选择，请输入 0-4 之间的数字。")
                continue

            # 任务完成后询问是否继续
            print("\n" + "="*80)
            continue_choice = input("任务完成！是否继续执行其他任务？(y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes', '是']:
                print("\n退出程序。再见！\n")
                break

        except KeyboardInterrupt:
            print("\n\n检测到 Ctrl+C，退出程序。\n")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ 执行出错: {e}")
            print("请检查错误信息并重试。\n")
            continue


if __name__ == "__main__":
    main()
