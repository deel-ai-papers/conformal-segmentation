echo "Cityscapes -- binary_loss" && python experiments/postprocess_expes.py --input-dir experiments/outputs/Cityscapes/binary_loss/
echo "Cityscapes -- miscoverage_loss" && python experiments/postprocess_expes.py --input-dir experiments/outputs/Cityscapes/miscoverage_loss/
echo "ADE20K -- binary_loss" && python experiments/postprocess_expes.py --input-dir experiments/outputs/ADE20K/binary_loss/
echo "ADE20K -- miscoverage_loss" && python experiments/postprocess_expes.py --input-dir experiments/outputs/ADE20K/miscoverage_loss/
echo "LoveDA -- binary_loss" && python experiments/postprocess_expes.py --input-dir experiments/outputs/LoveDA/binary_loss/
echo "LoveDA -- miscoverage_loss" && python experiments/postprocess_expes.py --input-dir experiments/outputs/LoveDA/miscoverage_loss/