# DGCNN PyTorch

## To run a training on the S3DIS Dataset
```
python main_semseg_s3dis.py --exp_name=semseg_s3dis_6 --test_area=6 
```

## To run training on a custom dataset (transfer learning from S3DIS)
```
python main_semseg_ourData.py --exp_name=semseg_s3dis_6 --test_area=6 --batch_size=8 --custom_mode==True --num_features=5
```
