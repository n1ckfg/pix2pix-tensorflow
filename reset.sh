cp -r files/train files/train_bak
./clean.sh
rmdir files/train
mv files/train_bak files/train

