import os
os.system("python com_main.py --method lstm --dataset WISDMar --subseq 96")
os.system("python com_main.py --method cnn --dataset WISDMar --subseq 96")
os.system("python com_main.py --method cnnlstm --dataset WISDMar --subseq 96")
os.system("python com_main.py --method lstm --dataset Sanitation --subseq 96")
os.system("python com_main.py --method cnn --dataset Sanitation --subseq 96")
os.system("python com_main.py --method cnnlstm --dataset Sanitation --subseq 96")

os.system("python com_main.py --method lstm --dataset UCI_HAPT --subseq 96")
os.system("python com_main.py --method cnn --dataset UCI_HAPT --subseq 96")
os.system("python com_main.py --method cnnlstm --dataset UCI_HAPT --subseq 96")

os.system("python com_main.py --method lstm --dataset UCI_Opportunity --subseq 96")
os.system("python com_main.py --method cnn --dataset UCI_Opportunity --subseq 96")
os.system("python com_main.py --method cnnlstm --dataset UCI_Opportunity --subseq 96")

# os.system("python com_main.py --method cnn --dataset WISDMar --subseq 160")
# os.system("python com_main.py --method cnn --dataset WISDMar --subseq 224")
# os.system("python com_main.py --method cnn --dataset WISDMar --subseq 288")
