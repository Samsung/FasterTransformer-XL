#python ./runData.py -i ./data/data.npz -o ./data/output.npz -j ./data/xlnet_cased_L-12_H-768_A-12/xlnet_config.json -m ../../../../train/saved_model -b 4 -l 1 -f 0 -n 0 -e 4
#python ./runData.py -i ./data/data.npz -o ./data/output.npz -j ./data/xlnet_cased_L-12_H-768_A-12/xlnet_config.json -m ../../../../train/saved_model -b 1 -l 1 -f 0 -n 0 -e 1
#python ./runData.py -i ./data/data.npz -o ./data/output.npz -j ./data/xlnet_cased_L-12_H-768_A-12/xlnet_config.json -m ../../../../train/saved_model -b 2 -l 1 -f 0 -n 0 -e 1
#python ./runData.py -i ./data/data.npz -o ./data/output.npz -m ../../../../train/saved_model -b 4 -l 1 -f 0 -n 0 -e 0
#python ./runData.py -i ./data/data.npz -o ./data/output.npz -m ../../../../train/saved_model -b 4 -l 1 -f 0 -n 0 -e 16
#python ./runData.py -o ./data/output.npz -m ../../../../train/saved_model -b 1 -l 1 -f 0 -n 0 -e 16
#python ./runData.py -o ./data/output.npz -m ../../../../train/saved_model -b 4 -l 1 -f 0 -n 0 -e 16
#python ./runData.py -o ./data/output.npz -m ../../../../train/saved_model -b 1 -l 1 -f 0 -n 0 -e 3
python ./runData.py -o ./data/output.npz -m ../../../../train/saved_model -l 1 -f 0 -n 0 -b 4 -r 1 -e 16 -d 0
