mkdir files\export_quantized
cd server\tools
python export-checkpoint.py --checkpoint ../../files/model --output_file ../../files/export_quantized/output.pict

