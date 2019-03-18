./test.sh
cd files/output/images
rm *.tga
for file in *-outputs.png; do convert $file $file.tga; done
for file in *.tga; do autotrace $file -background-color=000000 -color=16 -centerline -error-threshold=10 -line-threshold=0 -line-reversion-threshold=10 -output-format=svg -output-file $file.svg; done
rm *.tga
