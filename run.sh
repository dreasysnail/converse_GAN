# Training
python converse_gan.py -s 0.001 -l 0.00002 -m 0.1 |tee aim.txt
python converse_gan.py -s 0.001 -l 0.00002 -m 0.1 -d |tee daim.txt
# Testing
python converse_gan.py -s 0.001 -m 0.1 -d -t 
