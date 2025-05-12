# Baseline
python benchmark/B1_3dhp/00_evaluate.py -m camerahmr
python benchmark/B1_3dhp/00_evaluate.py -m tokenhmr
python benchmark/B1_3dhp/00_evaluate.py -m hmr2
python benchmark/B1_3dhp/00_evaluate.py -m nlf

# # Undist
# python benchmark/B1_3dhp/00_evaluate.py -m camerahmr -p undist_img
# python benchmark/B1_3dhp/00_evaluate.py -m tokenhmr -p undist_img
# python benchmark/B1_3dhp/00_evaluate.py -m hmr2 -p undist_img
# python benchmark/B1_3dhp/00_evaluate.py -m nlf -p undist_img

# # Known GT intrinsic
# # Undist
# python benchmark/B1_3dhp/00_evaluate.py -m camerahmr -p gt_intrinsic
# python benchmark/B1_3dhp/00_evaluate.py -m nlf -p gt_intrinsic