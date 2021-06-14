# NIH Chest X-Ray With PyTorch

## Requirements
* torch
* torchvision
* numpy

To run locally:
```
python main.py --data-path=/root/dir/to/dataset --model-path=/root/dir/to/save/models
```

Run: `python main.py --help` to get further options.  

To run the script on kaggle, create a notebook with the NIH Chest XRay dataset attached to it and add the following instructions:

```
!git clone https://github.com/PaulStryck/nih-chest-x-ray.git ./nih_chest_x_ray
!git -C nih_chest_x_ray pull
!git -C nih_chest_x_ray checkout tags/1.1
!pip install torchinfo

!python nih_chest_x_ray/main.py --folds=5 --fold-id=0 --seed=2021 --device="cuda" --data-path="/kaggle/input/data" --model-path="/kaggle/working/models" --test-bs=128 --train-bs=128 --val-bs=128 --data-frac=1 --log-interval=1 --epochs=5 --lr=0.0001
```

Adjust the --data-frac option between 0 and 1 to run only on a fraction of the data.
