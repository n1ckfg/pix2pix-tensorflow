@echo off

python pix2pix.py --mode train --output_dir files/model --max_epochs 200 --input_dir files/train --which_direction AtoB --batch_size 1

call export_all.bat

@pause