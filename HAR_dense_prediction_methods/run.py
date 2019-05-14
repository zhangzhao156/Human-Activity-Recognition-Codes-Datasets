import os
# unet subsequence analysis
# os.system("python main.py --dataset WISDMar --subseq 96")
# os.system("python main.py --dataset WISDMar --subseq 160")
# os.system("python main.py --dataset WISDMar --subseq 224")
# os.system("python main.py --dataset WISDMar --subseq 288")



# FCN 0318
# os.system("python main.py --dataset WISDMar --subseq 224")
# os.system("python main.py --dataset Sanitation --subseq 224")
# os.system("python main.py --dataset UCI_HAPT --subseq 224")
# os.system("python main.py --dataset UCI_Opportunity --subseq 224")

# SegNet 0427
# os.system("python main.py --dataset WISDMar --subseq 224")
# os.system("python main.py --dataset Sanitation --subseq 224")
# os.system("python main.py --dataset UCI_HAPT --subseq 224")
# os.system("python main.py --dataset UCI_Opportunity --subseq 224")

# SegNet 0427 different block
# os.system("python main.py --dataset WISDMar --subseq 224")

# SegNet 0427 different subsequence
# os.system("python main.py --dataset WISDMar --subseq 96")
# os.system("python main.py --dataset WISDMar --subseq 160")
# os.system("python main.py --dataset WISDMar --subseq 288")

# unet 0429 subsequence analysis different dataset
# os.system("python main.py --dataset Sanitation --subseq 96")
# os.system("python main.py --dataset Sanitation --subseq 160")
# os.system("python main.py --dataset Sanitation --subseq 288")
#
# os.system("python main.py --dataset UCI_HAPT --subseq 96")
# os.system("python main.py --dataset UCI_HAPT --subseq 160")
# os.system("python main.py --dataset UCI_HAPT --subseq 288")
#
# os.system("python main.py --dataset UCI_Opportunity --subseq 96")
# os.system("python main.py --dataset UCI_Opportunity --subseq 160")
# os.system("python main.py --dataset UCI_Opportunity --subseq 288")

# # Mask RCNN 0501
# os.system("python main.py --dataset WISDMar --subseq 28")
# os.system("python main.py --dataset Sanitation --subseq 28")
# os.system("python main.py --dataset UCI_HAPT --subseq 28")
# os.system("python main.py --dataset UCI_Opportunity --subseq 28")

# UNET different block on different datasets
# os.system("python main.py --dataset WISDMar --subseq 224 --block 4")
# os.system("python main.py --dataset WISDMar --subseq 224 --block 3")
# os.system("python main.py --dataset WISDMar --subseq 224 --block 2")

# os.system("python main.py --dataset Sanitation --subseq 224 --block 4")
# os.system("python main.py --dataset Sanitation --subseq 224 --block 3")
# os.system("python main.py --dataset Sanitation --subseq 224 --block 2")
#
# os.system("python main.py --dataset UCI_HAPT --subseq 224 --block 4")
# os.system("python main.py --dataset UCI_HAPT --subseq 224 --block 3")
# os.system("python main.py --dataset UCI_HAPT --subseq 224 --block 2")

# os.system("python main.py --dataset UCI_Opportunity --subseq 224 --block 4")
# os.system("python main.py --dataset UCI_Opportunity --subseq 224 --block 3")
# os.system("python main.py --dataset UCI_Opportunity --subseq 224 --block 2")

# label_gd for the win data non overlap
# os.system("python main.py --dataset WISDMar --subseq 96")
# os.system("python main.py --dataset Sanitation --subseq 96")
# os.system("python main.py --dataset UCI_HAPT --subseq 96")
# os.system("python main.py --dataset UCI_Opportunity --subseq 96")
# os.system("python main.py --dataset WISDMar --subseq 160")
# os.system("python main.py --dataset WISDMar --subseq 224")
# os.system("python main.py --dataset WISDMar --subseq 288")

# FCN subsequence analysis
# os.system("python main.py --dataset WISDMar --subseq 96")
# os.system("python main.py --dataset WISDMar --subseq 160")
# os.system("python main.py --dataset WISDMar --subseq 224")
# os.system("python main.py --dataset WISDMar --subseq 288")

# u-net different depth (different number block)
# os.system("python main.py --dataset WISDMar --subseq 224")

# 20190503
os.system("python main.py --dataset WISDMar --subseq 224 --block 5 --net segnet")
os.system("python main.py --dataset Sanitation --subseq 224 --block 5 --net segnet")
os.system("python main.py --dataset UCI_HAPT --subseq 224 --block 5 --net segnet")
os.system("python main.py --dataset UCI_Opportunity --subseq 224 --block 5 --net segnet")

